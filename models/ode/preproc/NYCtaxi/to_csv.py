import pickle
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

raw_records = {}
with open(f'./data_delta/NYCtaxi_2017_01-03_irregular_region_1_80.pkl', 'rb') as f:
    data = pickle.load(f)  # List of 80 arrays, each of shape (T_i, 9)
        
R = len(data)
for region in range(R):
    if region not in raw_records:
        raw_records[region] = []
    raw_records[region].extend(list(data[region]))
# 对齐时间戳到[tk-1, tk]
for region, rows in raw_records.items():
    arr = np.asarray(rows, dtype=np.float32)  # (T, 9)
    arr[:-1, 2:8] = arr[1:, 2:8]
    arr[-1, 2:8] = [23, 59, 4, 31, 3, 2017]
    raw_records[region] = arr.tolist()
        
# ========== 第二步：在 raw_records 上统计 inflow/outflow 的均值和方差 ==========        
all_inflow = []
all_outflow = []
for rows in raw_records.values():
    for row in rows:
        all_inflow.append(row[0])  # inflow
        all_outflow.append(row[1]) # outflow

all_inflow = np.array(all_inflow, dtype=float)
all_outflow = np.array(all_outflow, dtype=float)

inflow_mean = all_inflow.mean()
inflow_std  = all_inflow.std()
outflow_mean = all_outflow.mean()
outflow_std  = all_outflow.std()

print("inflow mean/std :", inflow_mean, inflow_std)
print("outflow mean/std:", outflow_mean, outflow_std)

stats_df = pd.DataFrame({
    'var':  ['inflow', 'outflow'],
    'mean': [inflow_mean, outflow_mean],
    'std':  [inflow_std,  outflow_std],
})
stats_df.to_csv('./data/NYCtaxi_2017_01-03_norm_stats.csv', index=False)

# ========== 第三步：用全局 mean/std 对 inflow/outflow 归一化，生成 records ==========
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
        
        inflow_norm = (inflow - inflow_mean) / inflow_std
        outflow_norm = (outflow - outflow_mean) / outflow_std
        
        day_start = datetime(int(year), int(month), int(day), 0, 0)
        time_value = (timestamp - day_start).total_seconds() / 60.0
        
        if current_day != day_start:
            current_day = day_start
            current_seq_id = seq_id_counter
            seq_id_counter += 1
                                
        region_data.append({
            # 'region_ID': region,
            'ID': current_seq_id,
            'Time': time_value,
            'Value_0': inflow_norm,
            'Value_1': outflow_norm,
            'Mask_0': 1.0,
            'Mask_1': 1.0,
        })
            
            
rows = []
for region, region_data in records.items():
    for e in region_data:
        rows.append(e)
df = pd.DataFrame(rows)
print(df.head())
print('---------------------')
print(df.tail())
df.to_csv('./data/NYCtaxi_2017_01-03_irregular_region_1_80_norm.csv', index=False)

# region_id = 0
# region_data_all = records[region_id]

# # 选一个具体的 day-ID，先拿第一条的 ID 例子：
# day_id = region_data_all[0]['ID']

# # 过滤出这一天的数据，并按 Time 排序
# region_data = [item for item in region_data_all if item['ID'] == day_id]
# region_data = sorted(region_data, key=lambda x: x['Time'])

# times = [item['Time'] for item in region_data]
# cum_inflow = [item['Value_0'] for item in region_data]
# cum_outflow = [item['Value_1'] for item in region_data]

# plt.figure(figsize=(14, 6))
# plt.plot(times, cum_inflow, label='Cumulative Inflow', linewidth=2)
# plt.plot(times, cum_outflow, label='Cumulative Outflow', linewidth=2)

# plt.title(f'Region {region_id}, Day-ID {day_id} - Daily Cumulative Inflow/Outflow')
# plt.xlabel('Time (minutes since 00:00)')
# plt.ylabel('Cumulative Value')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()      