import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

raw_records = {}
x_scale = 1
for i in range(6, 7):
    with open(f'./data_delta/NYCtaxi_2017-0{i}_irregular_region_1_80.pkl', 'rb') as f:
        data = pickle.load(f)  # List of 80 arrays, each of shape (T_i, 9)
        
    R = len(data)
    for region in range(R):
        if region not in raw_records:
            raw_records[region] = []
        raw_records[region].extend(list(data[region]))
        
# ========== 第二步 ==========
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
        if interval == 0:
            continue
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
            # 'inflow': inflow,
            # 'out_flow': outflow,
            'Value_0': inflow / interval,
            'Value_1': outflow / interval,
            'Mask_0': 1.0,
            'Mask_1': 1.0,
            'interval': interval
        })
            
            
rows = []
for region, region_data in records.items():
    for e in region_data:
        rows.append(e)
df = pd.DataFrame(rows)
# df.to_csv("./data_delta/NYCtaxi_2017-06_irregular_region_1_80.csv", index=False)
print(df.head())
print('-'*20)
print(df.tail())
print('='*40)

# ========== 第三步==========        
inflow_mean = df["Value_0"].mean()
inflow_std = df["Value_0"].std()
outflow_mean = df["Value_1"].mean()
outflow_std = df["Value_1"].std()
norm_stats = pd.DataFrame({
    "var": ["Value_0", "Value_1"],
    "mean": [inflow_mean, outflow_mean],
    "std": [inflow_std, outflow_std],
})

df["Value_0"] = (df["Value_0"] - inflow_mean) / inflow_std
df["Value_1"] = (df["Value_1"] - outflow_mean) / outflow_std

# 保存结果
df.to_csv("./data_delta/NYCtaxi_2017-06_irregular_region_1_80_norm.csv", index=False)
norm_stats.to_csv("./data_delta/NYCtaxi_2017-06_delta_norm_stats.csv", index=False)

print(df.head())
print(df.tail())