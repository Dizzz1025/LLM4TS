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

if __name__ == "__main__":
    data = np.load('train_data.npz')
    h_cache = data["h"]      # (N, T, H)
    mask    = data["mask"]         # (N, T)
    gt      = data["gt"]           # (N, T, variable_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HToGtDataset(h_cache, mask, gt)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 维度信息
    N, T, H = h_cache.shape
    _, _, D = gt.shape

    model = ProjectionModel(hidden_dim=H, out_dim=D).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(50):
        model.train()
        total_loss = 0

        for h_batch, mask_batch, gt_batch in loader:
            h_batch = h_batch.to(device)
            mask_batch = mask_batch.to(device)
            gt_batch = gt_batch.to(device)

            pred = model(h_batch)          # [B,T,D]
            loss = masked_mse(pred, gt_batch, mask_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch:02d}  Loss = {total_loss:.6f}")
    save_path = "projection_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "H": H,
        "D": D,
    }, save_path)
    print("Model saved to:", save_path)
    
    with torch.no_grad():
        pred = model(h_cache)
    mse = masked_mse(pred, gt, mask).item()
    print("\n===== Test Result =====")
    print(f"Masked MSE: {mse:.6f}")
    
    seq_id = 0
    pred_result = pred[seq_id].cpu().numpy()  # [T,D]
    gt_result = gt[seq_id]                    # [T,D]
    mask_result = mask[seq_id]                # [T]
    obs_idx = np.where(mask_result==1)[0]
    plt.scatter(obs_idx, gt_result[obs_idx], color="red", s=60, label="Observed (mask=1)")
    plt.scatter(obs_idx, pred_result[obs_idx], color="orange", s=20, label="Predicted (mask=1)")
    plt.show()