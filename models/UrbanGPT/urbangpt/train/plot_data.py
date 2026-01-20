import numpy as np
import matplotlib.pyplot as plt
import os

def plot_from_npz(file_path="preds_gts_dual_taxi_supervised.npz", save_name="eval_visualization.png"):
    # 1. 读取 .npz 文件
    if not os.path.exists(file_path):
        print(f"Error: 找不到文件 {file_path}")
        return

    data = np.load(file_path)
    preds = data['preds']  # 形状: (Total_Points, 2)
    gts = data['gts']      # 形状: (Total_Points, 2)

    # 2. 还原维度
    # 假设每个样本预测 12 个时间步
    num_samples = preds.shape[0] // 12
    # 重新整理成 (样本数, 12, 2)
    preds_reshaped = preds[:num_samples*12].reshape(num_samples, 12, 2)
    gts_reshaped = gts[:num_samples*12].reshape(num_samples, 12, 2)

    print(f"数据加载成功。样本总数: {num_samples}")

    # 3. 设置绘图风格
    plt.style.use('seaborn-v0_8-muted') # 或者使用 'ggplot'
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Ground Truth vs Prediction Comparison (Random 4 Samples)', fontsize=16)

    # 随机选 4 个不同的样本索引
    indices = np.random.choice(num_samples, 4, replace=False)
    
    steps = np.arange(1, 13) # 12个时间步

    for i, idx in enumerate(indices):
        ax = axes[i // 2, i % 2]
        
        # --- 绘制第一维 (通常是 Inflow) ---
        ax.plot(steps, gts_reshaped[idx, :, 0], 'o-', label='GT Inflow', color='#1f77b4', markersize=4)
        ax.plot(steps, preds_reshaped[idx, :, 0], 'x--', label='Pred Inflow', color='#ff7f0e', markersize=6)
        
        # --- 绘制第二维 (通常是 Outflow) ---
        ax.plot(steps, gts_reshaped[idx, :, 1], 's-', label='GT Outflow', color='#2ca02c', markersize=4, alpha=0.7)
        ax.plot(steps, preds_reshaped[idx, :, 1], 'd--', label='Pred Outflow', color='#d62728', markersize=6, alpha=0.7)

        ax.set_title(f"Sample Index: {idx}")
        ax.set_xlabel("Steps (Horizon)")
        ax.set_ylabel("Normalized Value")
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best', fontsize='small', ncol=2)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_name, dpi=300)
    print(f"✅ 图像已保存至: {save_name}")
    plt.show()

if __name__ == "__main__":
    plot_from_npz("preds_gts_dual_taxi_supervised.npz")