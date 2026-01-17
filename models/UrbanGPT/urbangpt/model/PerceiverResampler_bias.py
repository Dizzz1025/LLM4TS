import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
class HToGtDataset(Dataset):
    def __init__(self, h_cache, mask, gt):
        """
        h_cache: [N, T, H]
        mask:    [N, T]
        gt:      [N, T, D]
        """
        self.h = torch.tensor(h_cache, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.gt = torch.tensor(gt, dtype=torch.float32)

    def __len__(self):
        return self.h.shape[0]

    def __getitem__(self, idx):
        return self.h[idx], self.mask[idx], self.gt[idx]
def FeedForward(dim, mult = 4):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias = False)
    )

class PerceiverAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)
        self.beta = nn.Parameter(torch.tensor(1.0))
    def forward(self, x, latents, dynamics_bias=None, times=None):
        """
        einstein notation
        b - batch
        t - time
        n - sequence
        d - dimension
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents) # [b, n_latents, inner_dim]

        # the paper differs from Perceiver in which they also concat the key / values derived from the latents to be attended to
        # kv_input = torch.cat((x, latents), dim = -2)
        # k, v = self.to_kv(kv_input).chunk(2, dim = -1)

        k, v = self.to_kv(x).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        # attention
        # q: [b, h, n_latents, d]
        # k: [b, h, n_media, d]
        # v: [b, h, n_media, d]
        sim = einsum('... i d, ... j d  -> ... i j', q, k)
        # #### MODIFIED: Eq (12) - Inject dynamics information ####
        if dynamics_bias is not None:
            # dynamics_bias: [b, n_media] -> broadcast to [b, 1, 1, n_media]
            bias = dynamics_bias.unsqueeze(1).unsqueeze(1)
            # Ensure beta is non-negative as per paper
            beta_val = F.softplus(self.beta) 
            sim = sim + beta_val * bias
        # sim: [b, h, n_latents, n_media]
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        # #### MODIFIED: Eq (13) - Calculate Center Time ####
        center_times = None
        if times is not None:
            # times: [b, n_media]
            # attn: [b, h, n_latents, n_media]
            # Weighted average of time per head:
            t_weighted = einsum('b h i j, b j -> b h i', attn, times)
            # Average over heads to get one time per latent token
            center_times = t_weighted.mean(dim=1) # [b, n_latents]

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out), center_times

class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head = 64,
        heads = 8,
        num_latents = 5,
        # num_media_embeds = 4,
        ff_mult = 4
    ):
    # q: self.latents, (num_latents, dim)
    # k,v: x, (b, n, dim)
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        # self.media_pos_emb = nn.Parameter(torch.randn(num_media_embeds, 1, dim))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim = dim, dim_head = dim_head, heads = heads),
                FeedForward(dim = dim, mult = ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t_input=None):
        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times]
        b, t, d = x.shape
        latents = repeat(self.latents, 'n d -> b n d', b = b) # 每个样本用同一套查询模版，查询到的内容取决于输入x，latent学到的是怎么看
        # 1. Compute Dynamics Magnitude (Eq 11)
        # we approximate derivative using finite differences: ||h(t) - h(t-1)||
        if t > 1:
            # simple diff, padded at start to keep length T
            diff = x[:, 1:] - x[:, :-1]
            diff = torch.cat([torch.zeros(b, 1, d, device=x.device), diff], dim=1)
        else:
            diff = torch.zeros_like(x)
        # s_i = || h_dot(t) ||_2
        dynamics_mag = torch.norm(diff, p=2, dim=-1) # [B, T]
        # Normalize s_i within each sequence to obtain s_tilde (Eq 11 text)
        # Paper implies simple normalization, let's use division by max or sum.
        # Here we use Max scaling per batch item for stability.
        max_val = dynamics_mag.amax(dim=-1, keepdim=True).clamp(min=1e-5)
        dynamics_norm = dynamics_mag / max_val # [B, T]
        
        final_center_times = None
            
        for attn, ff in self.layers:
            attn_out, center_times = attn(x, latents, dynamics_bias=dynamics_norm, times=t_input)
            latents = attn_out + latents
            latents = ff(latents) + latents
            
            final_center_times = center_times

        return self.norm(latents), final_center_times
    
class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(dim // 2) * 0.1)

    def forward(self, t):  # t: [B, Nq]
        # expand
        t = t.unsqueeze(-1)                    # [B, Nq, 1]
        freqs = self.freq[None, None, :]       # [1, 1, D/2]
        x = t * freqs * 2 * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, q, kv):
        # q:  [B, Nq, D]
        # kv: [B, K,  D]
        B, Nq, _ = q.shape
        K = kv.shape[1]
        h = self.heads

        q = self.to_q(q)
        k, v = self.to_kv(kv).chunk(2, dim=-1)

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale
        attn = torch.softmax(torch.einsum('bhid,bhjd->bhij', q, k), dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class ODEPerceiverPredictor(nn.Module):
    def __init__(
        self,
        *,
        dim,                # hidden dim of ODE / Perceiver
        y_dim=2,              # output dimension (obs dim)
        perceiver_depth=2,
        num_latents=5,
        heads=8,
        dim_head=64,
        ff_mult=4,
    ):
        super().__init__()

        # 1. Perceiver Resampler
        self.perceiver = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            heads=heads,
            dim_head=dim_head,
            num_latents=num_latents,
            ff_mult=ff_mult
        )

        # 2. Time embedding (query)
        self.time_emb = TimeEmbedding(dim)

        # 3. Cross-attention: time query → latent memory
        self.cross_attn = CrossAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head
        )

        # 4. Projection head
        self.proj = nn.Linear(dim, y_dim)

    def forward(
        self,
        h_seq,      # [B, T, D]   ODE hidden states
        t_query,    # [B, Nq]     times to predict (obs or future)
        t_input   # [B, T]
    ):
        """
        Returns:
            y_hat: [B, Nq, y_dim]
        """

        B, T, D = h_seq.shape

        # ---- Step 1: Perceiver compress history ----
        # perceiver expects x: [B, N, D]
        latent, center_times = self.perceiver(h_seq, t_input=t_input)     # [B, K, D]

        # ---- Step 2: time embedding (center time + query) ----
        center_time_emb = self.time_emb(center_times) # [B, K, D]
        q_time = self.time_emb(t_query)    # [B, Nq, D]

        # ---- Step 3: cross attention ----
        pred_h = self.cross_attn(q_time, latent)   # [B, Nq, D]

        # ---- Step 4: project to observation space ----
        y_hat = self.proj(pred_h)          # [B, Nq, y_dim]

        return y_hat

# 问题：t_query在每个样本中长度都不一定，所以不能stack
# def build_time_queries(gt, mask):
#     """
#     gt:    [B, T, y_dim]
#     mask:  [B, T]   (1 = observed)

#     return:
#         t_query:  [B, Nq]
#         gt_query: [B, Nq, y_dim]
#     """
#     B, T = mask.shape
#     t_list, gt_list = [], []

#     for b in range(B):
#         idx = mask[b].nonzero(as_tuple=False).squeeze(-1)
#         t_list.append(idx.float())
#         gt_list.append(gt[b, idx])
#     t_query = torch.stack(t_list)
#     t_query = t_query / (T - 1)
#     return t_query, torch.stack(gt_list)

def build_t_input(T, device):
    """
    return:
        t_input: [0, 119]  -> 可 broadcast 到 batch
    """
    idx = torch.arange(0, T, device=device).float()
    t_query = idx / (T - 1)   # 归一化
    return t_query.unsqueeze(0)  # [1, 60]

def build_future_queries(T, device):
    """
    return:
        t_query: [60, 119]  -> 可 broadcast 到 batch
    """
    idx = torch.arange(T - 60, T, device=device).float()
    t_query = idx / (T - 1)   # 归一化
    return t_query.unsqueeze(0)  # [1, 60]

def masked_mse(pred, gt, mask):
    # pred, gt: [B, T, D]
    # mask:     [B, T]
    _, _, D = pred.shape
    mask = mask.unsqueeze(-1)     # [B,T,1]
    diff = (pred - gt) ** 2
    diff = diff * mask            # 只在 mask=1 的位置求损失
    # 避免除以0
    denom = mask.sum() * D
    return diff.sum() / denom

class MaskedCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        # only_attend_immediate_media = True
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim, bias = False)

        # whether for text to only attend to immediate preceding image, or all images

        # self.only_attend_immediate_media = only_attend_immediate_media

    def forward(
        self,
        x, # 文本序列中每个token的嵌入表示, shape: (batch, text_len, dim)
        media, # 预处理后的特征, shape: (batch, K, dim)
        # media_locations = None
    ):
        # b, t, m = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim = -1)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)
        attn = sim.softmax(dim = -1)
        
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class GatedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        # only_attend_immediate_media = True
    ):
        super().__init__()
        self.feat_attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads)
        self.text_attn = MaskedCrossAttention(dim = dim, dim_head = dim_head, heads = heads)
        self.attn_gate = nn.Parameter(torch.tensor([0.])) # alpha
        self.fusion_lambda = nn.Parameter(torch.tensor(0.5)) # lambda

        self.ff = FeedForward(dim, mult = ff_mult)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(
        self,
        x,
        feat_kv,                  # media tensor, encoded by perceiver resample - (batch, K, dim)
        text_kv
    ):
        # 1. 分别计算两路 Attention 的输出
        # 这两部分相当于公式里的 Delta_feat 和 Delta_text
        d_feat = self.feat_attn(x, feat_kv)
        d_text = self.text_attn(x, text_kv)

        # 2. 按照 lambda 进行融合: (lambda * d_feat + (1 - lambda) * d_text)
        l = torch.sigmoid(self.fusion_lambda)
        combined_delta = l * d_feat + (1 - l) * d_text

        # 3. 应用门控 (alpha) 并加回到主路径: Y = Y + alpha * combined_delta
        # 注意：这里的 attn_gate.tanh() 完美承担了公式中 alpha 的角色
        x = combined_delta * self.attn_gate.tanh() + x

        # 4. 这里的 FFW 同样是基于融合后的信息进行残差更新
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ODEPerceiverPredictor(dim=50, y_dim=2, perceiver_depth=2, num_latents=5).to(device)
    data = np.load('./Sim/charged_test_data.npz')
    h_cache = data["h"]      # (N, T, H)
    mask    = data["mask"]         # (N, T)
    gt      = data["gt"]           # (N, T, variable_num)
    dataset = HToGtDataset(h_cache, mask, gt)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(150):
        model.train()
        total_loss = 0.0

        for h_batch, mask_batch, gt_batch in loader:
            # h_batch:    [B, T, D]
            # t_all_batch:[B, T]
            # mask_batch: [B, T]
            # gt_batch:   [B, T, y_dim]
            B = h_batch.shape[0]
            T = h_batch.shape[1]
            h_batch = h_batch.to(device)
            mask_batch = mask_batch.to(device)
            gt_batch = gt_batch.to(device)

            # ---- 构造 time query（代替 masked loss）----
            # t_query, gt_query = build_time_queries(gt_batch, mask_batch)
            t_input = build_t_input(T=T, device=device)
            t_query = build_future_queries(T=T, device=device)
            t_input = t_input.repeat(B, 1) # [B, T]
            t_query = t_query.repeat(B, 1)
            # t_query:  [B, Nq]
            # gt_query: [B, Nq, y_dim]

            # ---- forward ----
            pred = model(h_batch, t_query, t_input)   # [B, Nq, y_dim]

            # ---- loss ----
            loss = masked_mse(pred, gt_batch[:, -60:, :], mask_batch[:, -60:])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d}  Loss = {total_loss:.6f}")
        save_path = "charged_projection_model_v2.pt"
        torch.save({"model_state_dict": model.state_dict()}, save_path)
    print("Model saved to:", save_path)