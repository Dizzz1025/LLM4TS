import pickle
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

raw_records = {}
x_scale = 1
# for i in range(1, 4):
#     with open(f'./data_delta/NYCtaxi_2017-0{i}_irregular_region_1_80.pkl', 'rb') as f:
#         data = pickle.load(f)  # List of 80 arrays, each of shape (T_i, 9)
        
#     R = len(data)
#     for region in range(R):
#         if region not in raw_records:
#             raw_records[region] = []
#         raw_records[region].extend(list(data[region]))
with open(f'./data_delta/NYCtaxi_2017_01-03_irregular_region_1_80.pkl', 'rb') as f:
    data = pickle.load(f)  # List of 80 arrays, each of shape (T_i, 9)
R = len(data)
print('num_of_regions:', R)
for region in range(R):
    if region not in raw_records:
        raw_records[region] = []
    raw_records[region].extend(list(data[region]))    
# ========== 第二步 ==========
records = {}
seq_id_counter = 0
windows_minutes = int(1.5 * 24 * 60) # 一天半的长度        
for region, all_rows in raw_records.items():         
    region_row = []
    for row in all_rows:
        # row: [inflow, outflow, hour, minute, week, day, month, year, interval]
        hour, minute, week, day, month, year = row[2:8]
        inflow, outflow, interval = row[0], row[1], int(row[8])
        if interval == 0:
            continue
        ts = datetime(int(year), int(month), int(day), int(hour), int(minute))
        region_row.append((ts, inflow, outflow, interval))
    
    all_dates = sorted({r[0].date() for r in region_row})
    region_data = []
    n = len(all_dates)
    for i in range(n - 1):
        d = all_dates[i]
        start_ts = datetime(d.year, d.month, d.day, 0, 0)
        end_ts = start_ts + timedelta(minutes=windows_minutes)
        current_id = seq_id_counter
        seq_id_counter += 1
        
        for ts, inflow, outflow, interval in region_row:
            # 只取在 [start_ts, end_ts) 里的数据
            if ts < start_ts:
                continue
            if ts >= end_ts:
                break

            delta_min = (ts - start_ts).total_seconds() / 60.0
            time_value = delta_min / x_scale

            region_data.append({
                'ID': current_id,
                'Time': time_value,
                'Value_0': inflow / interval,
                'Value_1': outflow / interval,
                'Mask_0': 1.0,
                'Mask_1': 1.0,
                'interval': interval
            })
    records[region] = region_data
            
            
rows = []
for region, region_data in records.items():
    for e in region_data:
        rows.append(e)
df = pd.DataFrame(rows)
df.to_csv("./data_delta/NYCtaxi_2017_01-03_irregular_region_1_80_less8.csv", index=False)
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
df.to_csv("./data_delta/NYCtaxi_2017_01-03_irregular_region_1_80_less8_norm.csv", index=False)
norm_stats.to_csv("./data_delta/NYCtaxi_2017_01-03_delta_less8_norm_stats.csv", index=False)

print(df.head())
print(df.tail())