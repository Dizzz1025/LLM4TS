import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

raw_records = {}
x_scale = 1
for i in range(1, 4):
    with open(f'./data_trend/NYCtaxi_2017-0{i}_irregular_region_1_80.pkl', 'rb') as f:
        data = pickle.load(f)  # List of 80 arrays, each of shape (T_i, 9)
        
    R = len(data)
    for region in range(R):
        if region not in raw_records:
            raw_records[region] = []
        raw_records[region].extend(list(data[region]))
        
# ========== 第二步：用全局 mean/std 对 inflow/outflow 归一化，生成 records ==========
records = {}
seq_id_counter = 0        
for region, all_rows in raw_records.items():
    records[region] = []
    region_data = records[region]
    
    current_day = None
    current_seq_id = None
    
    for row in all_rows:
        # row: [inflow, outflow, hour, minute, week, day, month, year, interval]
        hour, minute, week, day, month, year = row[2:8]
        inflow, outflow, interval = row[0], row[1], int(row[8])
        timestamp = datetime(int(year), int(month), int(day), int(hour), int(minute))
        
        day_start = datetime(int(year), int(month), int(day), 0, 0)
        time_value = (timestamp - day_start).total_seconds() / 60.0 / x_scale # 对x缩放100倍，扩大a0
        
        if current_day != day_start:
            current_day = day_start
            current_seq_id = seq_id_counter
            seq_id_counter += 1
                                
        region_data.append({
            # 'region_ID': region,
            'ID': current_seq_id,
            'Time': time_value,
            'Value_0': inflow,
            'Value_1': outflow,
            'Mask_0': 1.0,
            'Mask_1': 1.0,
        })
            
            
rows = []
for region, region_data in records.items():
    for e in region_data:
        rows.append(e)
df = pd.DataFrame(rows)
print(df.head())
print('-'*20)
print(df.tail())
print('='*40)
# ========== 第三步：对每个 ID 进行线性拟合，计算残差 ==========
df_res = []
coeff_list = []
for ID, group in df.groupby("ID"):
    t_all = group["Time"].values
    mask_known = t_all <= 1080 / x_scale
    t_known = t_all[mask_known]

    # 对 Value_0 做线性拟合
    y0 = group["Value_0"].values
    y0_known = y0[mask_known]
    a0 = np.sum(t_known * y0_known) / np.sum(t_known ** 2)
    y0_trend = a0 * t_all
    y0_residual = y0 - y0_trend

    # 对 Value_1 做线性拟合
    y1 = group["Value_1"].values
    y1_known = y1[mask_known]
    a1 = np.sum(t_known * y1_known) / np.sum(t_known ** 2)
    y1_trend = a1 * t_all
    y1_residual = y1 - y1_trend
    
    # save coeff
    coeff_list.append({
        "ID": ID,
        "a0": a0, 
        "a1": a1,
    })

    # 构造新的 dataframe 行
    tmp = group.copy()
    tmp["Value_0"] = y0_residual
    tmp["Value_1"] = y1_residual
    tmp["Trend_0"] = y0_trend
    tmp["Trend_1"] = y1_trend

    df_res.append(tmp)

coeff_df = pd.DataFrame(coeff_list)
coeff_df.to_csv("./data_trend/NYCtaxi_trend_coeff.csv", index=False)
df_final = pd.concat(df_res).reset_index(drop=True)

print(df_final.head())
print('-'*20)
print(df_final.tail())
print('='*40)
# ========== 第四步：在 raw_records 上统计 inflow/outflow 的均值和方差 ==========        
inflow_mean = df_final["Value_0"].mean()
inflow_std = df_final["Value_0"].std()
outflow_mean = df_final["Value_1"].mean()
outflow_std = df_final["Value_1"].std()
norm_stats = pd.DataFrame({
    "var": ["Value_0", "Value_1"],
    "mean": [inflow_mean, outflow_mean],
    "std": [inflow_std, outflow_std],
})

df_final["Value_0"] = (df_final["Value_0"] - inflow_mean) / inflow_std
df_final["Value_1"] = (df_final["Value_1"] - outflow_mean) / outflow_std

# 保存结果
df_final.to_csv("./data_trend/NYCtaxi_2017_01-03_irregular_region_1_80_residual_norm.csv", index=False)
norm_stats.to_csv("./data_trend/NYCtaxi_residual_norm_stats.csv", index=False)

print(df_final.head())
print(df_final.tail())