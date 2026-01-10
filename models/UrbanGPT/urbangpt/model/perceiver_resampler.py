import torch
from torch import nn
from einops import repeat
from models.resampler.PerceiverResampler import PerceiverResampler

class PerceiverSTWrapper(nn.Module):
    def __init__(self, dim=50, num_latents=5, depth=2, dim_head=64, heads=8, ff_mult=4):
        super().__init__()
        self.perceiver = PerceiverResampler(
            dim=dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            num_latents=num_latents,
            ff_mult=ff_mult
        )

    def forward(self, h_seq):
        return self.perceiver(h_seq)

def load_perceiver_st_model(path=None, device="cuda"):
    model = PerceiverSTWrapper().to(device)
    if path is not None:
        state = torch.load(path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model
