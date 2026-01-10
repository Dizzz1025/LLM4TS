import torch
import torch.nn as nn
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

class ProjectionModel(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super().__init__()
        # self.proj = nn.Linear(hidden_dim, out_dim)
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2, bias=True),
            torch.nn.ReLU(),
            # torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_dim//2, out_dim, bias=True),
        )

    def forward(self, h):
        # h: [B, T, H]
        return self.proj(h)  # 输出 [B, T, D]
    
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

def masked_mse_for_test(pred, gt, mask, last_k=59):
    # pred, gt: [B, T, D]
    # mask:     [B, T]
    pred = pred[:, -last_k:, :]    # [B, K, D]
    gt   = gt[:, -last_k:, :]      # [B, K, D]
    mask = mask[:, -last_k:]       # [B, K]
    
    _, _, D = pred.shape
    mask = mask.unsqueeze(-1)     # [B,T,1]
    diff = (pred - gt) ** 2
    diff = diff * mask            # 只在 mask=1 的位置求损失
    # 避免除以0
    denom = mask.sum() * D
    return diff.sum() / denom    

if __name__ == "__main__":
    data = np.load('test_data.npz')
    h_cache = data["h"]      # (N, T, H)
    mask    = data["mask"]         # (N, T)
    gt      = data["gt"]           # (N, T, variable_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("projection_model.pt", map_location=device)
    H = ckpt["H"]
    D = ckpt["D"]
    model = ProjectionModel(hidden_dim=H, out_dim=D).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    h_cache = torch.tensor(h_cache, dtype=torch.float32).to(device)
    mask  = torch.tensor(mask, dtype=torch.float32).to(device)
    gt    = torch.tensor(gt, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        pred = model(h_cache)
    mse = masked_mse_for_test(pred, gt, mask).item()
    print("\n===== Test Result =====")
    print(f"Masked MSE: {mse:.6f}")
    
    seq_id = 0
    pred_result = pred[seq_id].cpu().numpy()  # [T,D]
    gt_result = gt[seq_id].cpu().numpy()                    # [T,D]
    mask_result = mask[seq_id].cpu().numpy()                # [T]
    T_tail = 60
    pred_tail = pred_result[-T_tail:]
    gt_tail   = gt_result[-T_tail:]
    mask_tail = mask_result[-T_tail:] 
    obs_idx = np.where(mask_tail==1)[0]
    plt.plot(gt_tail[obs_idx, 0], gt_tail[obs_idx, 1], color="red", label="Observed (mask=1)")
    plt.plot(pred_tail[obs_idx, 0], pred_tail[obs_idx, 1], color="orange", label="Predicted (mask=1)")
    plt.show()