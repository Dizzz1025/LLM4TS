import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# ================= 配置路径 =================
TYPE = 'charged'                 # 实验类型/模型名称
NPZ_PATH = 'preds_gts_dual_charged.npz' # 形状: (Total_Points, 2)
PLOT_DIR = f"{TYPE}_plots"            # 图片保存文件夹
RESULT_CSV = "metrics_results.csv"    # 汇总指标结果
# ===========================================

# ================= 绘图配置 =================
PLOT_INTERVAL = 1000   # 每隔多少个样本画一张图
HORIZON = 23           # 预测步长
DIM_0_LABEL = 'X'      # 维度0标签 (映射为 X 轴)
DIM_1_LABEL = 'Y'      # 维度1标签 (映射为 Y 轴)
# ===========================================

def plot_from_npz(file_path=NPZ_PATH, save_dir=PLOT_DIR):
    """
    加载NPZ文件并绘制2D轨迹对比图 (X, Y 坐标映射)。
    计算当前样本的 MSE 并显示在图表上。
    """
    # 1. 读取 .npz 文件
    if not os.path.exists(file_path):
        print(f"❌ 错误: 找不到文件 {file_path}")
        return

    print(f"正在加载数据: {file_path}...")
    data = np.load(file_path)
    
    # 兼容不同的 key 名
    preds_raw = data['preds']
    gts_raw = data['gts'] if 'gts' in data else data['labels']

    # 2. 还原维度
    num_total_points = preds_raw.shape[0]
    num_samples = num_total_points // HORIZON
    
    if num_samples == 0:
        print(f"❌ 错误: 总点数 ({num_total_points}) 不足一个步长 ({HORIZON})")
        return

    # 重新整理成 (样本数, HORIZON, 维度)
    preds = preds_raw[:num_samples * HORIZON].reshape(num_samples, HORIZON, -1)
    gts = gts_raw[:num_samples * HORIZON].reshape(num_samples, HORIZON, -1)

    print(f"数据加载成功。总点数: {num_total_points}, 样本总数: {num_samples}, 预测步长: {HORIZON}")

    # 3. 创建保存目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"已创建文件夹: {save_dir}")

    # 4. 循环绘图
    print(f"开始绘制轨迹图并计算单图 MSE，采样间隔: {PLOT_INTERVAL}...")
    
    for idx in range(0, num_samples, PLOT_INTERVAL):
        plt.figure(figsize=(8, 8))
        
        # 提取当前样本数据 (Horizon, 2)
        p_sample = preds[idx]
        g_sample = gts[idx]
        
        # 计算当前图的 MSE (均方误差)
        sample_mse = np.mean((p_sample - g_sample)**2)
        # 计算每一维度的 MSE
        mse_x = np.mean((p_sample[:, 0] - g_sample[:, 0])**2)
        mse_y = np.mean((p_sample[:, 1] - g_sample[:, 1])**2)
        
        # 真实轨迹 (X=Dim0, Y=Dim1)
        gt_x, gt_y = g_sample[:, 0], g_sample[:, 1]
        # 预测轨迹
        pr_x, pr_y = p_sample[:, 0], p_sample[:, 1]
        
        # --- 绘制真实路径 ---
        plt.plot(gt_x, gt_y, 'o-', label='Ground Truth', color='#1f77b4', linewidth=2, markersize=4, alpha=0.6)
        plt.scatter(gt_x[0], gt_y[0], c='#1f77b4', marker='s', s=100, edgecolors='black', label='GT Start', zorder=5)
        plt.scatter(gt_x[-1], gt_y[-1], c='#1f77b4', marker='X', s=100, edgecolors='black', label='GT End', zorder=5)
        
        # --- 绘制预测路径 ---
        plt.plot(pr_x, pr_y, 'x--', label='Prediction', color='#d62728', linewidth=2, markersize=6, alpha=0.8)
        plt.scatter(pr_x[-1], pr_y[-1], c='#d62728', marker='X', s=80, edgecolors='black', label='Pred End', zorder=5)

        # 在图上打印 MSE
        info_text = f"Sample MSE: {sample_mse:.6f}\nMSE_X: {mse_x:.6f}\nMSE_Y: {mse_y:.6f}"
        plt.text(0.05, 0.95, info_text, transform=plt.gca().transAxes, fontsize=10,
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

        # 图表装饰
        plt.title(f"Trajectory Comparison | Sample Index: {idx}\nMethod: {TYPE}", fontsize=14)
        plt.xlabel(f"Dimension 0 ({DIM_0_LABEL})", fontsize=12)
        plt.ylabel(f"Dimension 1 ({DIM_1_LABEL})", fontsize=12)
        
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.legend(loc='lower right', frameon=True, shadow=True) # 修改图例位置避免遮挡 MSE 文字
        
        plt.axis('equal')
        
        # 保存并关闭
        save_path = os.path.join(save_dir, f"trajectory_{idx:06d}.png")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()

    print(f"✅ 所有带 MSE 标注的轨迹图已保存至文件夹: {save_dir}")

    # 5. 计算并保存整体指标
    calculate_and_save_metrics(preds, gts)

def calculate_and_save_metrics(preds, gts):
    """计算 MAE/MSE 并追加到汇总 CSV"""
    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Method', 'Steps', 'Dimension', 'MAE', 'MSE'])
        
        for d in range(preds.shape[-1]):
            mae = np.mean(np.abs(preds[..., d] - gts[..., d]))
            mse = np.mean((preds[..., d] - gts[..., d])**2)
            writer.writerow([TYPE, HORIZON, d, mae, mse])
    print(f"📊 整体指标 (Steps={HORIZON}) 已追加至: {RESULT_CSV}")

if __name__ == "__main__":
    plot_from_npz()