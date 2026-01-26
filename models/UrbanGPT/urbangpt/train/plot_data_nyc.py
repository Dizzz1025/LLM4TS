import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ================= 配置路径 =================
TYPE = 'bike_supervised' 
NPZ_PATH = f'preds_gts_dual_{TYPE}.npz' 
PLOT_DIR = f"nyc_timeseries_{TYPE}"
RESULT_CSV = "nyc_flow_metrics.csv"
# ===========================================

# ================= 绘图配置 =================
PLOT_INTERVAL = 100     # 每隔多少个样本画一张图
HORIZON = 12           # 预测的时间步长 (UrbanGPT NYC 任务通常为 12)
# ===========================================

def plot_flow_timeseries(file_path=NPZ_PATH, save_dir=PLOT_DIR):
    """
    绘制 Inflow 和 Outflow 随时间变化的曲线对比图
    """
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    data = np.load(file_path)
    preds_raw = data['preds'] # (Total_Points, 2) -> [Inflow, Outflow]
    gts_raw = data['gts']     # (Total_Points, 2)

    num_samples = preds_raw.shape[0] // HORIZON
    preds = preds_raw[:num_samples * HORIZON].reshape(num_samples, HORIZON, 2)
    gts = gts_raw[:num_samples * HORIZON].reshape(num_samples, HORIZON, 2)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"开始绘制流量时序图... 总样本: {num_samples}")

    time_steps = np.arange(1, HORIZON + 1)

    for idx in range(0, num_samples, PLOT_INTERVAL):
        # 创建两个子图：上方画 Inflow，下方画 Outflow
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # --- 子图 1: Inflow (第 0 维) ---
        ax1.plot(time_steps, gts[idx, :, 0], 'o-', label='GT Inflow', color='#1f77b4', linewidth=2)
        ax1.plot(time_steps, preds[idx, :, 0], 'x--', label='Pred Inflow', color='#ff7f0e', linewidth=2)
        ax1.set_ylabel('Normalized Flow')
        ax1.set_title(f'Sample {idx} | Variable: INFLOW')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # --- 子图 2: Outflow (第 1 维) ---
        ax2.plot(time_steps, gts[idx, :, 1], 'o-', label='GT Outflow', color='#2ca02c', linewidth=2)
        ax2.plot(time_steps, preds[idx, :, 1], 'x--', label='Pred Outflow', color='#d62728', linewidth=2)
        ax2.set_ylabel('Normalized Flow')
        ax2.set_xlabel('Future Time Steps')
        ax2.set_title(f'Sample {idx} | Variable: OUTFLOW')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # 计算该样本的 MSE
        mse_in = np.mean((preds[idx, :, 0] - gts[idx, :, 0])**2)
        mse_out = np.mean((preds[idx, :, 1] - gts[idx, :, 1])**2)
        plt.suptitle(f"NYC Flow Prediction - {TYPE}\nInflow MSE: {mse_in:.6f} | Outflow MSE: {mse_out:.6f}", fontsize=12)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(save_dir, f"flow_ts_{idx:05d}.png")
        plt.savefig(save_path, dpi=120)
        plt.close()

    print(f"✅ 时序图已保存至: {save_dir}")
    
    # 统计所有样本的平均指标
    save_final_metrics(preds, gts)

def save_final_metrics(preds, gts):
    # 计算 Inflow (dim 0) 和 Outflow (dim 1) 的各自指标
    mse_in = np.mean((preds[..., 0] - gts[..., 0])**2)
    mae_in = np.mean(np.abs(preds[..., 0] - gts[..., 0]))
    
    mse_out = np.mean((preds[..., 1] - gts[..., 1])**2)
    mae_out = np.mean(np.abs(preds[..., 1] - gts[..., 1]))

    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Method', 'Variable', 'MAE', 'MSE'])
        writer.writerow([TYPE, 'Inflow', f"{mae_in:.6f}", f"{mse_in:.6f}"])
        writer.writerow([TYPE, 'Outflow', f"{mae_out:.6f}", f"{mse_out:.6f}"])
    
    print(f"📊 指标已汇总至 {RESULT_CSV}")

if __name__ == "__main__":
    plot_flow_timeseries()