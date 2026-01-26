import numpy as np
import pandas as pd
import os
import csv

# ================= 配置路径 =================
TYPE = 'charged'
NPZ_PATH = 'preds_gts_dual_charged.npz' 
CSV_PATH = 'charged_100.csv' 
RESULT_CSV = "charged_100_metrics_final_23steps.csv"
# ===========================================

# ================= 评估配置 =================
HORIZON = 23      
INTERVALS = [70, 80, 90, 100, 110, 120] 
# ===========================================

def run_evaluation():
    print(f"--- 正在执行后 {HORIZON} 步区间评估: {TYPE} ---")
    
    if not os.path.exists(NPZ_PATH) or not os.path.exists(CSV_PATH):
        print("❌ 错误: 找不到 NPZ 或 CSV 文件")
        return

    # 1. 加载并还原预测数据 (N, 23, 2)
    data = np.load(NPZ_PATH)
    preds_all = data['preds']
    gts_all = data['gts'] if 'gts' in data else data['labels']
    
    num_samples = preds_all.shape[0] // HORIZON
    preds = preds_all[:num_samples * HORIZON].reshape(num_samples, HORIZON, -1)
    gts = gts_all[:num_samples * HORIZON].reshape(num_samples, HORIZON, -1)
    
    # 2. 读取 CSV 并获取每个样本的最后 23 个时间戳
    df = pd.read_csv(CSV_PATH)
    # 按照 ID 排序，确保与 Dataset 顺序一致
    unique_ids = sorted(df['ID'].unique())
    
    # 获取时间矩阵 (num_samples, 23)
    time_matrix = []
    
    print("正在匹配每个样本的最后 23 个观测点时间戳...")
    for sample_id in unique_ids[:num_samples]:
        # 提取该 ID 的所有数据，按时间排序，取最后 23 行
        sample_df = df[df['ID'] == sample_id].sort_values('Time')
        last_23_times = sample_df['Time'].values[-HORIZON:]
        
        if len(last_23_times) < HORIZON:
             # 如果不足23步，用 -1 填充以防报错，统计时会自动过滤掉
             temp = np.full(HORIZON, -1.0)
             temp[-len(last_23_times):] = last_23_times
             time_matrix.append(temp)
        else:
             time_matrix.append(last_23_times)
             
    time_matrix = np.array(time_matrix)

    # 3. 按区间执行误差统计
    file_exists = os.path.isfile(RESULT_CSV)
    with open(RESULT_CSV, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Method', 'Time_Interval', 'Dimension', 'MAE', 'MSE', 'Point_Count'])

        print(f"\n{'Step_Range':<12} | {'Dim':<6} | {'MAE':<10} | {'MSE':<10} | {'Count':<8}")
        print("-" * 55)

        for end_t in INTERVALS:
            sum_abs_err = np.zeros(2)
            sum_sq_err = np.zeros(2)
            pt_count = 0

            # 遍历样本 i 和预测步 j
            for i in range(num_samples):
                for j in range(HORIZON):
                    t_curr = time_matrix[i, j]
                    
                    # 只有物理时间落在当前考察区间内才统计
                    # (这里计算的是从 0 到 end_t 的累计，符合你参考代码 <= end_t 的逻辑)
                    if 0 <= t_curr <= end_t:
                        p = preds[i, j, :]
                        g = gts[i, j, :]
                        
                        diff = p - g
                        sum_abs_err += np.abs(diff)
                        sum_sq_err += diff**2
                        pt_count += 1

            # 计算平均指标
            range_label = f"0-{end_t}"
            for d in range(2):
                mae = sum_abs_err[d] / pt_count if pt_count > 0 else 0
                mse = sum_sq_err[d] / pt_count if pt_count > 0 else 0
                
                print(f"{range_label:<12} | {f'Dim_{d}':<6} | {mae:<10.6f} | {mse:<10.6f} | {int(pt_count):<8}")
                writer.writerow([TYPE, range_label, d, mae, mse, int(pt_count)])
            print("-" * 55)

    print(f"✅ 评估完成。结果已保存至: {RESULT_CSV}")

if __name__ == "__main__":
    run_evaluation()