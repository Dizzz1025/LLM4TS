import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os

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
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim//2, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim//2, out_dim, bias=True),
        )

    def forward(self, h):
        # 确保输入是 Tensor
        if isinstance(h, np.ndarray):
            h = torch.from_numpy(h).float().to(next(self.parameters()).device)
        return self.proj(h)
    
def masked_mse(pred, gt, mask):
    # 确保所有输入都是 Tensor 且在同一设备
    if isinstance(gt, np.ndarray):
        gt = torch.from_numpy(gt).float().to(pred.device)
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).float().to(pred.device)
        
    _, _, D = pred.shape
    mask = mask.unsqueeze(-1)     # [B,T,1]
    diff = (pred - gt) ** 2
    diff = diff * mask            
    denom = mask.sum() * D + 1e-8 # 避免除以0
    return diff.sum() / denom

if __name__ == "__main__":
    # 1. 加载数据
    data_path = 'h_mask_gt_charged.npz'
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found.")
        exit()
        
    data = np.load(data_path)
    h_cache = data["h"]      # (N, T, H)
    mask    = data["mask"]   # (N, T)
    gt      = data["gt"]     # (N, T, D)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = HToGtDataset(h_cache, mask, gt)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    N, T, H = h_cache.shape
    _, _, D = gt.shape

    # 2. 初始化模型
    model = ProjectionModel(hidden_dim=H, out_dim=D).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # 3. 训练循环
    print("Starting training...")
    for epoch in range(50):
        model.train()
        total_loss = 0
        for h_batch, mask_batch, gt_batch in loader:
            h_batch = h_batch.to(device)
            mask_batch = mask_batch.to(device)
            gt_batch = gt_batch.to(device)

            pred = model(h_batch)
            loss = masked_mse(pred, gt_batch, mask_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss = {total_loss/len(loader):.6f}")

    # 4. 保存模型
    save_path = "projection_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "H": H, "D": D,
    }, save_path)
    print("Model saved to:", save_path)
    
    # 5. 测试与可视化 (修复点)
    model.eval()
    with torch.no_grad():
        # 将整个 h_cache 转为 Tensor 再输入
        h_tensor = torch.from_numpy(h_cache).float().to(device)
        pred_all = model(h_tensor)
        
        # 计算整体 MSE
        mse_val = masked_mse(pred_all, gt, mask).item()
        print("\n===== Test Result =====")
        print(f"Masked MSE: {mse_val:.6f}")
    
    # 可视化第一个序列
    seq_id = 0
    pred_result = pred_all[seq_id].cpu().numpy()  # [T,D]
    gt_result = gt[seq_id]                        # [T,D]
    mask_result = mask[seq_id]                    # [T]
    
    obs_idx = np.where(mask_result == 1)[0]
    
    plt.figure(figsize=(10, 5))
    # 假设 D=2, 我们只画第一维作为展示，或者画轨迹
    if D >= 2:
        plt.subplot(1, 2, 1)
        plt.title("Dimension 0 (X)")
        plt.scatter(obs_idx, gt_result[obs_idx, 0], color="blue", label="GT")
        plt.scatter(obs_idx, pred_result[obs_idx, 0], color="orange", marker='x', label="Pred")
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.title("Dimension 1 (Y)")
        plt.scatter(obs_idx, gt_result[obs_idx, 1], color="green", label="GT")
        plt.scatter(obs_idx, pred_result[obs_idx, 1], color="red", marker='x', label="Pred")
        plt.legend()
    else:
        plt.scatter(obs_idx, gt_result[obs_idx], color="red", s=60, label="GT")
        plt.scatter(obs_idx, pred_result[obs_idx], color="orange", s=20, label="Pred")
        plt.legend()
        
    plt.tight_layout()
    plt.show()