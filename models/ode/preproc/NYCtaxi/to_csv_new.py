import pickle
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

raw_records = {}
x_scale = 1
with open(f'./data/NYCtaxi_2021-12_irregular_region_1_80.pkl', 'rb') as f:
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
    raw_records[region] = region_row
        
# start_time = datetime(2017, 1, 1, 0, 0)
# end_time   = datetime(2017, 3, 31, 23, 59)
start_time = datetime(2021, 12, 1, 0, 0)
end_time   = datetime(2021, 12, 31, 23, 59)
all_hours = []
t = start_time
t = t.replace(minute=0, second=0, microsecond=0)
while t <= end_time:
    all_hours.append(t)
    t += timedelta(hours=1)

region_data = []
n = len(all_hours)
print('n:', n)
idx = 0
group_id = 0
for i in range(0, n - int(windows_minutes / 60), 8):
    d = all_hours[i]
    start_ts = d
    end_ts = start_ts + timedelta(minutes=windows_minutes)
    for region, all_rows in raw_records.items():
        for ts, inflow, outflow, interval in all_rows:
            # 只取在 [start_ts, end_ts) 里的数据
            if ts < start_ts:
                continue
            if ts >= end_ts:
                break

            delta_min = (ts - start_ts).total_seconds() / 60.0
            time_value = delta_min / x_scale

            region_data.append({
                'ID': idx,
                'group_ID': group_id,
                'Time': time_value,
                'Value_0': inflow,
                'Value_1': outflow,
                'Mask_0': 1.0,
                'Mask_1': 1.0,
            })
        idx += 1
    group_id += 1
df = pd.DataFrame(region_data)

            
            
# rows = []
# for region, region_data in records.items():
#     for e in region_data:
#         rows.append(e)
# df = pd.DataFrame(rows)
df["Time"] = df["Time"] // 10
df.to_csv("./data/NYCtaxi_2021-12_irregular_region_1_80.csv", index=False)
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
df.to_csv("./data/NYCtaxi_2021-12_irregular_region_1_80_norm.csv", index=False)
norm_stats.to_csv("./data/NYCtaxi_2021-12_delta_norm_stats.csv", index=False)

print(df.head())
print(df.tail())