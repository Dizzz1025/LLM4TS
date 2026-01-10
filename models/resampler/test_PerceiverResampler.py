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

    def forward(self, x, latents):
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
        # sim: [b, h, n_latents, n_media]
        sim = sim - sim.amax(dim = -1, keepdim = True).detach()
        attn = sim.softmax(dim = -1)

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h = h)
        return self.to_out(out)

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

    def forward(self, x):
        # times = x.shape[1]
        # x = x + self.media_pos_emb[:times]
        latents = repeat(self.latents, 'n d -> b n d', b = x.shape[0]) # 每个样本用同一套查询模版，查询到的内容取决于输入x，latent学到的是怎么看

        for attn, ff in self.layers:
            latents = attn(x, latents) + latents
            latents = ff(latents) + latents

        return self.norm(latents)
    
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
    ):
        """
        Returns:
            y_hat: [B, Nq, y_dim]
        """

        B, T, D = h_seq.shape

        # ---- Step 1: Perceiver compress history ----
        # perceiver expects x: [B, N, D]
        latent = self.perceiver(h_seq)     # [B, K, D]

        # ---- Step 2: time embedding (query) ----
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

def build_future_queries(T, device):
    """
    return:
        t_query: [1, 60]  -> 可 broadcast 到 batch
    """
    idx = torch.arange(T - 59, T, device=device).float()
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
    return diff.sum(), denom

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ODEPerceiverPredictor(dim=50, y_dim=2, perceiver_depth=2, num_latents=5).to(device)
    data = np.load('./Sim/test_data.npz')
    h_cache = data["h"]      # (N, T, H)
    mask    = data["mask"]         # (N, T)
    gt      = data["gt"]           # (N, T, variable_num)
    ckpt = torch.load("projection_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    test_dataset = HToGtDataset(h_cache, mask, gt)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    total_se = 0.0
    total_count = 0.0
    all_preds = []
    all_gts = []
    all_masks = []
    with torch.no_grad():
        for h_batch, mask_batch, gt_batch in test_loader:
            B, T, _ = h_batch.shape

            h_batch = h_batch.to(device)
            mask_batch = mask_batch.to(device)
            gt_batch = gt_batch.to(device)

            t_query = build_future_queries(T=T, device=device)
            t_query = t_query.repeat(B, 1)

            pred = model(h_batch, t_query)   # [B,60,D]

            se_sum, count = masked_mse(
                pred,
                gt_batch[:, -59:, :],
                mask_batch[:, -59:]
            )

            total_se += se_sum.item()
            total_count += count.item()
            
            all_preds.append(pred.cpu())
            all_gts.append(gt_batch[:, -59:, :].cpu())
            all_masks.append(mask_batch[:, -59:].cpu())

    final_mse = total_se / total_count
    print("total_count:", total_count)
    print("\n===== Test Result =====")
    print(f"Masked MSE: {final_mse:.6f}")
    
    
    pred_all = torch.cat(all_preds, dim=0)   # [N, 60, D]
    gt_all   = torch.cat(all_gts, dim=0)
    mask_all = torch.cat(all_masks, dim=0)
    seq_id = 2
    pred_result = pred_all[seq_id].cpu().numpy()  # [T,D]
    gt_result = gt_all[seq_id].cpu().numpy()                    # [T,D]
    mask_result = mask_all[seq_id].cpu().numpy()                # [T]
    T_tail = 59
    # pred_tail = pred_result[-T_tail:]
    pred_tail = pred_result
    gt_tail   = gt_result[-T_tail:]
    mask_tail = mask_result[-T_tail:] 
    obs_idx = np.where(mask_tail==1)[0]
    plt.plot(gt_tail[obs_idx, 0], gt_tail[obs_idx, 1], color="red", label="Observed (mask=1)")
    plt.plot(pred_tail[obs_idx, 0], pred_tail[obs_idx, 1], color="orange", label="Predicted (mask=1)")
    plt.show()