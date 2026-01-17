import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os

# ================== 配置区 ==================
# 只需要修改这里即可切换数据集
DATA_PREFIX = "NYCbike"          # 数据集前缀: NYCbike 或 NYCtaxi
TIME_STR = "2017_04-06"         # 文件名中的时间标识
NUM_REGIONS = 80                # 处理的区域数量
MODE = 'train'
# 设定数据的实际起始和结束时间（根据你的文件名 2017_04-06 修改）
START_TIME = datetime(2017, 4, 1, 0, 0)
END_TIME   = datetime(2017, 6, 30, 23, 59)

INPUT_FILE = f'./data_accu/{MODE}/{DATA_PREFIX}_{TIME_STR}_irregular_region_1_{NUM_REGIONS}.pkl'
OUTPUT_CSV = f'./data_accu/{MODE}/{DATA_PREFIX}_{TIME_STR}_irregular_region_1_{NUM_REGIONS}.csv'
NORM_CSV   = f'./data_accu/{MODE}/{DATA_PREFIX}_{TIME_STR}_irregular_region_1_{NUM_REGIONS}_norm.csv'
STATS_CSV  = f'./data_accu/{MODE}/{DATA_PREFIX}_delta_norm_stats.csv'
# ============================================

# ========== 第一步：加载数据 ==========
if not os.path.exists(INPUT_FILE):
    print(f"错误: 找不到文件 {INPUT_FILE}")
    exit()

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)  # 预期: 80个数组的列表

raw_records = {}
R = len(data)
print(f'正在处理 {DATA_PREFIX} 数据, 区域数量:', R)

# 转换原始数据格式
for region in range(R):
    region_row = []
    for row in data[region]:
        # row: [inflow, outflow, hour, minute, week, day, month, year, interval]
        inflow, outflow = row[0], row[1]
        hour, minute, week, day, month, year = row[2:8]
        interval = int(row[8])
        
        if interval == 0:
            continue
            
        ts = datetime(int(year), int(month), int(day), int(hour), int(minute))
        region_row.append((ts, inflow, outflow, interval))
    raw_records[region] = region_row

# ========== 第二步：生成滑动窗口序列 ==========
x_scale = 1
windows_minutes = int(1.5 * 24 * 60) # 1.5天

# 生成时间轴
all_hours = []
t = START_TIME.replace(minute=0, second=0, microsecond=0)
while t <= END_TIME:
    all_hours.append(t)
    t += timedelta(hours=1)

region_data = []
n = len(all_hours)
idx = 0
group_id = 0

# 滑动窗口步长为 8 小时
for i in range(0, n - int(windows_minutes / 60), 8):
    start_ts = all_hours[i]
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

# 离散化时间
df["Time"] = df["Time"] // 10
df.to_csv(OUTPUT_CSV, index=False)
print(f"原始数据已保存至: {OUTPUT_CSV}")

# ========== 第三步：标准化 ==========
inflow_mean, inflow_std = df["Value_0"].mean(), df["Value_0"].std()
outflow_mean, outflow_std = df["Value_1"].mean(), df["Value_1"].std()

norm_stats = pd.DataFrame({
    "var": ["Value_0", "Value_1"],
    "mean": [inflow_mean, outflow_mean],
    "std": [inflow_std, outflow_std],
})

df["Value_0"] = (df["Value_0"] - inflow_mean) / inflow_std
df["Value_1"] = (df["Value_1"] - outflow_mean) / outflow_std

# 保存标准化结果
df.to_csv(NORM_CSV, index=False)
norm_stats.to_csv(STATS_CSV, index=False)

print(f"标准化数据已保存至: {NORM_CSV}")
print(f"统计信息已保存至: {STATS_CSV}")
print('-'*20)
print(df.head())