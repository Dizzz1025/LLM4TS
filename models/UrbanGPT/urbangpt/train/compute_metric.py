import numpy as np
import os

def recalculate_metrics(file_path="preds_gts_dual_springs.npz"):
    # 1. 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：在以下路径找不到文件: {os.path.abspath(file_path)}")
        return

    # 2. 加载数据
    data = np.load(file_path)
    preds = data['preds']  # 形状 (N, 2)
    gts = data['gts']      # 形状 (N, 2)

    # 3. 计算总体指标
    overall_mae = np.mean(np.abs(preds - gts))
    overall_mse = np.mean((preds - gts) ** 2)

    # 4. 分维度计算 (Dim 1 为索引 0, Dim 2 为索引 1)
    # 第一维 (Dimension 1)
    mae_d1 = np.mean(np.abs(preds[:, 0] - gts[:, 0]))
    mse_d1 = np.mean((preds[:, 0] - gts[:, 0]) ** 2)

    # 第二维 (Dimension 2)
    mae_d2 = np.mean(np.abs(preds[:, 1] - gts[:, 1]))
    mse_d2 = np.mean((preds[:, 1] - gts[:, 1]) ** 2)

    # 5. 打印结果表格 (MAE 在前，4位小数)
    print("\n" + "="*55)
    print(f"📊 Metrics calculated from: {file_path}")
    print(f"Total Prediction Points: {preds.shape[0]}")
    print("-" * 55)
    print(f"{'Metric Scope':<18} | {'MAE':<12} | {'MSE':<12}")
    print("-" * 55)
    print(f"{'Overall':<18} | {overall_mae:<12.4f} | {overall_mse:<12.4f}")
    print(f"{'Dimension 1':<18} | {mae_d1:<12.4f} | {mse_d1:<12.4f}")
    print(f"{'Dimension 2':<18} | {mae_d2:<12.4f} | {mse_d2:<12.4f}")
    print("="*55 + "\n")

if __name__ == "__main__":
    # 你可以直接在这里修改文件名
    recalculate_metrics("preds_gts_dual_springs.npz")