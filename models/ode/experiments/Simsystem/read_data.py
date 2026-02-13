import numpy as np
import pandas as pd

# ================= 配置路径 =================
TYPE = 'charged'
NPY_PATH = f"test_p_vec_{TYPE}.npy" 
CSV_PATH = f"{TYPE}_norm.csv"
# ===========================================

def run_evaluation():
    print("Loading data...")
    npy_data = np.load(NPY_PATH, allow_pickle=True)
    df = pd.read_csv(CSV_PATH)
    
    # 筛选时间范围
    test_df = df[(df['Time'] >= 61) & (df['Time'] <= 120)].copy()
    
    intervals = [70, 80, 90, 100, 110, 120]
    
    print(f"\n{'Range':<10} | {'Var':<6} | {'MAE':<10} | {'MSE':<10} | {'Count':<8}")
    print("-" * 52)

    for end_t in intervals:
        current_df = test_df[test_df['Time'] <= end_t]
        
        sum_abs_error = np.zeros(2)
        sum_sq_error = np.zeros(2)
        counts = np.zeros(2)
        
        for row in current_df.itertuples():
            t_obs = int(row.Time) - 61
            # 直接将 ID 转为索引
            sample_pos = int(row.ID) 
            
            # 为了防止越界做的简单保护
            if sample_pos >= npy_data.shape[1]:
                continue

            # 取预测值和真实值
            pred = npy_data[t_obs, sample_pos, :]
            true = [row.Value_0, row.Value_1]
            
            for d in range(2):
                diff = pred[d] - true[d]
                sum_abs_error[d] += abs(diff)
                sum_sq_error[d] += diff**2
                counts[d] += 1

        for d in range(2):
            mae = sum_abs_error[d] / counts[d] if counts[d] > 0 else 0
            mse = sum_sq_error[d] / counts[d] if counts[d] > 0 else 0
            print(f"{f'61-{end_t}':<10} | {f'Dim_{d}':<6} | {mae:<10.4f} | {mse:<10.4f} | {int(counts[d]):<8}")
        print("-" * 52)

if __name__ == "__main__":
    run_evaluation()