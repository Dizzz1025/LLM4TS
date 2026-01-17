import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import math
from pathlib import Path

# ================== 核心参数 ==================
DATA_PREFIX = "NYCcrime"
SELECTED_VALUE_INDICES = [0, 1] 

# 采样与窗口参数
WINDOW_DAYS = 144            # 窗口长度
SLIDE_STEP_DAYS = 8         # 滑动步长
X_SCALE = 1.5               # 时间轴缩放

# Train 模式固定时间与区域
START_TIME = datetime(2016, 1, 1, 0, 0)
END_TIME   = datetime(2018, 12, 31, 23, 59)
REGION_STR = "1_80"
TIME_STR   = "2016-2018"

# 路径设置
INPUT_FILE = f'./data/train/{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.pkl'
SAVE_DIR   = Path('./data/train')
OUTPUT_CSV = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.csv'
NORM_CSV   = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}_norm.csv'
STATS_CSV  = SAVE_DIR / f'{DATA_PREFIX}_delta_norm_stats.csv'
# =============================================

# ========== 第一步：加载数据 ==========
if not os.path.exists(INPUT_FILE):
    print(f"错误: 找不到文件 {INPUT_FILE}")
    exit()

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

raw_records = {}
print(f'正在处理 {DATA_PREFIX} Train 数据, 区域数量:', len(data))

for region in range(len(data)):
    region_row = []
    for row in data[region]:
        # row: [v0, v1, v2, v3, hour, min, week, day, mon, yr, interval]
        v_0 = row[SELECTED_VALUE_INDICES[0]]
        v_1 = row[SELECTED_VALUE_INDICES[1]]
        h, m, d, mon, yr = row[4], row[5], row[7], row[8], row[9]
        
        ts = datetime(int(yr), int(mon), int(d), int(h), int(m))
        region_row.append({'ts': ts, 'v0': v_0, 'v1': v_1})
    
    # 窗口内累加必须保证时间有序
    region_row.sort(key=lambda x: x['ts'])
    raw_records[region] = region_row

# ========== 第二步：生成滑动窗口序列 ==========
all_starts = []
curr_t = START_TIME
while curr_t <= (END_TIME - timedelta(days=WINDOW_DAYS)):
    all_starts.append(curr_t)
    curr_t += timedelta(days=SLIDE_STEP_DAYS)

print(f"生成的滑动窗口总数: {len(all_starts)}")

# ========== 第三步：滑动窗口处理与区间内累加 ==========
region_data = []
idx = 0
group_id = 0
window_duration = timedelta(days=WINDOW_DAYS)

for start_ts in all_starts:
    end_ts = start_ts + window_duration
    
    for region, all_rows in raw_records.items():
        # 筛选窗口内样本
        window_samples = [row for row in all_rows if start_ts <= row['ts'] < end_ts]
        
        if not window_samples:
            continue
            
        cum_v0 = 0
        cum_v1 = 0
        
        for row in window_samples:
            # 执行累加逻辑
            cum_v0 += row['v0']
            cum_v1 += row['v1']
            
            # 计算缩放后的时间点
            delta_days = (row['ts'] - start_ts).total_seconds() / 86400.0
            time_value = math.floor(delta_days * X_SCALE)

            region_data.append({
                'ID': idx,
                'group_ID': group_id,
                'Time': time_value,
                'Value_0': cum_v0,
                'Value_1': cum_v1,
                'Mask_0': 1.0,
                'Mask_1': 1.0,
            })
        idx += 1
    group_id += 1

df = pd.DataFrame(region_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"原始累加数据已保存至: {OUTPUT_CSV}")

# ========== 第四步：计算统计信息并标准化 ==========
# 计算 Train 集合的均值和标准差
inflow_mean, inflow_std = df["Value_0"].mean(), df["Value_0"].std()
outflow_mean, outflow_std = df["Value_1"].mean(), df["Value_1"].std()

norm_stats = pd.DataFrame({
    "var": ["Value_0", "Value_1"],
    "mean": [inflow_mean, outflow_mean],
    "std": [inflow_std, outflow_std],
})
norm_stats.to_csv(STATS_CSV, index=False)

# 执行归一化
df["Value_0"] = (df["Value_0"] - inflow_mean) / (inflow_std + 1e-9)
df["Value_1"] = (df["Value_1"] - outflow_mean) / (outflow_std + 1e-9)

df.to_csv(NORM_CSV, index=False)

print(f"标准化数据已保存至: {NORM_CSV}")
print(f"统计信息已保存至: {STATS_CSV}")
print('-'*20)
print(df.head())