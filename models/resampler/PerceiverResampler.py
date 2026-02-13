import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# PART 1: Core Attention Mechanisms (Token Compression)
# Reference: Section 3.4 - Token Compression
# =============================================================================

class PerceiverAttention(nn.Module):
    """
    Dynamics-aware Cross-Attention for Token Compression.
    Implements the resampling attention with dynamics bias (Eq. 12) and 
    center time calculation (Eq. 13).
    """
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        
        # Learnable scalar beta for dynamics bias (Eq. 12)
        self.beta = nn.Parameter(torch.tensor(1.0))

    def forward(self, x, latents, dynamics_bias=None, times=None):
        """
        Args:
            x: Input features (ODE latents) [B, T, D]
            latents: Learnable latent queries [B, K, D]
            dynamics_bias: Normalized dynamics magnitude s_tilde [B, T]
            times: Input timestamps t_i [B, T]
        """
        x = self.norm_media(x)
        latents = self.norm_latents(latents)

        b, m, h = *x.shape[:2], self.heads

        q = self.to_q(latents) 
        k, v = self.to_kv(x).chunk(2, dim=-1) # Key/Value from input history

        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        # Standard cross-attention logits
        # q: [b, h, K, d], k: [b, h, T, d] -> sim: [b, h, K, T]
        sim = einsum('... i d, ... j d  -> ... i j', q, k)

        # ---- Eq. 12: Inject Dynamics-Aware Bias ----
        # alpha_{k,i} = softmax(QK^T / sqrt(d) + beta * s_tilde_i)
        if dynamics_bias is not None:
            # dynamics_bias: [B, T] -> [B, 1, 1, T]
            bias = dynamics_bias.unsqueeze(1).unsqueeze(1)
            beta_val = F.softplus(self.beta) 
            sim = sim + beta_val * bias

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # ---- Eq. 13: Calculate Center Time ----
        # t_k^{center} = sum_i (alpha_{k,i} * t_i)
        center_times = None
        if times is not None:
            # times: [B, T] -> Weighted average by attention map
            t_weighted = einsum('b h i j, b j -> b h i', attn, times)
            # Average over heads to get per-token center time
            center_times = t_weighted.mean(dim=1) # [B, K]

        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        return self.to_out(out), center_times


class PerceiverResampler(nn.Module):
    """
    Compresses variable-length latent history into fixed-size memory tokens.
    Wraps PerceiverAttention to compute dynamics magnitude (Eq. 11).
    """
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        num_latents=5,
        ff_mult=4
    ):
        super().__init__()
        # Learnable latent queries z_k
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttention(dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(dim=dim, mult=ff_mult)
            ]))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, t_input=None):
        b, t, d = x.shape
        # Repeat initial queries for batch
        latents = repeat(self.latents, 'n d -> b n d', b=b) 

        # ---- Eq. 11: Compute Dynamics Magnitude ----
        # s_i = || h_dot(t_i) ||_2 approx || h(t_i) - h(t_{i-1}) ||_2
        if t > 1:
            diff = x[:, 1:] - x[:, :-1]
            diff = torch.cat([torch.zeros(b, 1, d, device=x.device), diff], dim=1)
        else:
            diff = torch.zeros_like(x)
        
        dynamics_mag = torch.norm(diff, p=2, dim=-1) # [B, T]

        # Normalize s_i to s_tilde (Normalization step in Section 3.4)
        max_val = dynamics_mag.amax(dim=-1, keepdim=True).clamp(min=1e-5)
        dynamics_norm = dynamics_mag / max_val # [B, T]
        
        final_center_times = None
            
        for attn, ff in self.layers:
            # Pass dynamics_norm as bias to attention
            attn_out, center_times = attn(x, latents, dynamics_bias=dynamics_norm, times=t_input)
            latents = attn_out + latents
            latents = ff(latents) + latents
            
            final_center_times = center_times

        # Output compressed tokens z_k and center times t_k^{center}
        return self.norm(latents), final_center_times

# =============================================================================
# PART 2: Injection Mechanism
# Reference: Section 3.5 - Dual-Source Gated Injection
# =============================================================================

class MaskedCrossAttention(nn.Module):
    """
    Standard Cross-Attention used for injecting tokens into the LLM stream.
    Used in Eq. 18 (Delta_dyn and Delta_prior).
    """
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, x, media):
        # x: LLM hidden states (Query)
        # media: Compressed tokens (Key/Value)
        h = self.heads

        x = self.norm(x)
        q = self.to_q(x)

        k, v = self.to_kv(media).chunk(2, dim=-1)
        
        q = rearrange(q, 'b n (h d) -> b h n d', h=h)
        k = rearrange(k, 'b n (h d) -> b h n d', h=h)
        v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        q = q * self.scale

        sim = einsum('... i d, ... j d -> ... i j', q, k)
        attn = sim.softmax(dim=-1)
        
        out = einsum('... i j, ... j d -> ... i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class GatedCrossAttentionBlock(nn.Module):
    """
    Gated injection block to fuse external evidence into LLM.
    Corresponds to the gating logic in Eq. 18 and Eq. 19.
    """
    def __init__(
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
    ):
        super().__init__()
        self.attn = MaskedCrossAttention(dim=dim, dim_head=dim_head, heads=heads)
        # Gating parameter alpha (Eq. 20)
        self.attn_gate = nn.Parameter(torch.tensor([0.]))

        self.ff = FeedForward(dim, mult=ff_mult)
        # Gating parameter alpha_dense (Eq. 19, 20)
        self.ff_gate = nn.Parameter(torch.tensor([0.]))

    def forward(self, x, media):
        # x: LLM hidden states Y
        # media: Evidence tokens Z or P
        
        # Eq. 18: Y = Y + alpha * CrossAttn(Y, Z)
        x = self.attn(x, media) * self.attn_gate.tanh() + x
        
        # Eq. 19: Y = Y + alpha_dense * FFN(Y)
        x = self.ff(x) * self.ff_gate.tanh() + x
        return x

# =============================================================================
# PART 3: Predictor Wrapper & Utilities
# =============================================================================

class ODEPerceiverPredictor(nn.Module):
    """
    End-to-end predictor wrapper:
    1. Compresses latent history using PerceiverResampler.
    2. Embeds center times and query times.
    3. Predicts future states via Cross-Attention and Projection.
    """
    def __init__(
        self,
        *,
        dim,
        y_dim=2,
        perceiver_depth=2,
        num_latents=5,
        heads=8,
        dim_head=64,
        ff_mult=4,
    ):
        super().__init__()

        # 1. Perceiver Resampler (Token Compression)
        self.perceiver = PerceiverResampler(
            dim=dim,
            depth=perceiver_depth,
            heads=heads,
            dim_head=dim_head,
            num_latents=num_latents,
            ff_mult=ff_mult
        )

        # 2. Time Embedding (Eq. 14 / Sec 3.4)
        self.time_emb = TimeEmbedding(dim)

        # 3. Cross-Attention (Query Slot Alignment)
        # Maps query times to memory tokens to generate predictions
        self.cross_attn = CrossAttention(
            dim=dim,
            heads=heads,
            dim_head=dim_head
        )

        # 4. Regression Head (Eq. 21)
        self.proj = nn.Linear(dim, y_dim)

    def forward(self, h_seq, t_query, t_input):
        """
        h_seq: ODE latent states [B, T, D]
        t_query: Query timestamps [B, Nq]
        t_input: Input timestamps [B, T]
        """
        B, T, D = h_seq.shape

        # Step 1: Compress history into dynamic memory tokens (Eq. 7 - 11)
        latent, center_times = self.perceiver(h_seq, t_input=t_input) # [B, K, D]

        # Step 2: Time embedding (Eq. 14, 15)
        # Embed center times (Eq. 15) and query times
        center_time_emb = self.time_emb(center_times) 
        q_time = self.time_emb(t_query) # [B, Nq, D]

        # Note: The actual code implementation here uses q_time to attend to 'latent'.
        # Theoretically, 'latent' should be fused with 'center_time_emb' (Eq. 16: z_tilde = z + W_tau * tau)
        # The cross_attn below effectively aligns query times with memory tokens.

        # Step 3: Cross attention (Predict features at query times)
        pred_h = self.cross_attn(q_time, latent)   # [B, Nq, D]

        # Step 4: Final projection (Eq. 21)
        y_hat = self.proj(pred_h)          # [B, Nq, y_dim]

        return y_hat

# =============================================================================
# PART 4: Helper Modules
# =============================================================================

def FeedForward(dim, mult=4):
    """Standard position-wise feed-forward network."""
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias=False),
        nn.GELU(),
        nn.Linear(inner_dim, dim, bias=False)
    )

class TimeEmbedding(nn.Module):
    """
    Multi-frequency sinusoidal time embedding (Eq. 14).
    phi(t) = [sin(w_1 t), cos(w_1 t), ...]
    """
    def __init__(self, dim):
        super().__init__()
        self.freq = nn.Parameter(torch.randn(dim // 2) * 0.1)

    def forward(self, t):
        t = t.unsqueeze(-1)                    # [B, N, 1]
        freqs = self.freq[None, None, :]       # [1, 1, D/2]
        x = t * freqs * 2 * math.pi
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class CrossAttention(nn.Module):
    """
    Simple Cross-Attention module for the predictor head.
    """
    def __init__(self, dim, heads=8, dim_head=64):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, q, kv):
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

class HToGtDataset(Dataset):
    def __init__(self, h_cache, mask, gt):
        """
        Dataset wrapper for cached ODE states and ground truth.
        """
        self.h = torch.tensor(h_cache, dtype=torch.float32)
        self.mask = torch.tensor(mask, dtype=torch.float32)
        self.gt = torch.tensor(gt, dtype=torch.float32)

    def __len__(self):
        return self.h.shape[0]

    def __getitem__(self, idx):
        return self.h[idx], self.mask[idx], self.gt[idx]

def build_t_input(T, device):
    """Generates normalized timestamps for the input sequence."""
    idx = torch.arange(0, T, device=device).float()
    t_query = idx / (T - 1)
    return t_query.unsqueeze(0)

def build_future_queries(T, device):
    """Generates normalized timestamps for the future prediction window."""
    idx = torch.arange(T - 60, T, device=device).float()
    t_query = idx / (T - 1)
    return t_query.unsqueeze(0)

def masked_mse(pred, gt, mask):
    """Computes Mean Squared Error only on observed values (masked)."""
    _, _, D = pred.shape
    mask = mask.unsqueeze(-1)
    diff = (pred - gt) ** 2
    diff = diff * mask
    denom = mask.sum() * D
    return diff.sum() / denom