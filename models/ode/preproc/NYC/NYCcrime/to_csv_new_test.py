import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
import math
from pathlib import Path

# ================== 核心参数区 ==================
MODE = "supervised"              # 选项: 'zero-shot' 或 'supervised'
DATA_PREFIX = "NYCcrime"
SELECTED_VALUE_INDICES = [0, 1] 

# 1. 窗口与缩放参数 (需与 train 保持一致)
WINDOW_DAYS = 144            
SLIDE_STEP_DAYS = 8         
X_SCALE = 1.5               

# 2. 模式动态设定
# 统一输出到 test 文件夹
SAVE_DIR = Path(f'./data/test/') 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if MODE == "zero-shot":
    REGION_STR = "81_160"
    TIME_STR = "2020"
    START_TIME = datetime(2020, 1, 1, 0, 0)
    END_TIME   = datetime(2020, 12, 31, 23, 59)
elif MODE == "supervised":
    REGION_STR = "1_80"
    TIME_STR = "2021"
    START_TIME = datetime(2021, 1, 1, 0, 0)
    END_TIME   = datetime(2021, 12, 31, 23, 59)

# 3. 路径构建 (去掉了 window_cum 后缀)
INPUT_FILE = Path(f'./data/test') / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.pkl'
RAW_CSV_OUT = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.csv'
NORM_CSV_OUT = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}_norm.csv'

# 强制读取 train 的归一化基准
STATS_PATH = Path('./data/train') / f'{DATA_PREFIX}_delta_norm_stats.csv'
# ===============================================

print(f"--- 模式: {MODE.upper()} ---")
print(f"归一化基准: {STATS_PATH}")

# ========== 第一步：加载数据 ==========
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

with open(INPUT_FILE, 'rb') as f:
    data = pickle.load(f)

raw_records = {}
print(f'加载区域数量: {len(data)}')

for region in range(len(data)):
    region_row = []
    for row in data[region]:
        # row: [v0, v1, v2, v3, h, m, week, d, mon, y, interval]
        v_0 = row[SELECTED_VALUE_INDICES[0]]
        v_1 = row[SELECTED_VALUE_INDICES[1]]
        h, m, d, mon, yr = row[4], row[5], row[7], row[8], row[9]
        
        ts = datetime(int(yr), int(mon), int(d), int(h), int(m))
        region_row.append({'ts': ts, 'v0': v_0, 'v1': v_1})
    
    region_row.sort(key=lambda x: x['ts'])
    raw_records[region] = region_row

# ========== 第二步：生成滑动窗口序列 ==========
all_starts = []
curr_t = START_TIME
while curr_t <= (END_TIME - timedelta(days=WINDOW_DAYS)):
    all_starts.append(curr_t)
    curr_t += timedelta(days=SLIDE_STEP_DAYS)

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
            
        cum_v0, cum_v1 = 0, 0
        for row in window_samples:
            # 累加逻辑
            cum_v0 += row['v0']
            cum_v1 += row['v1']
            
            # 时间缩放
            delta_days = (row['ts'] - start_ts).total_seconds() / 86400.0
            time_value = math.floor(delta_days * X_SCALE)

            region_data.append({
                'ID': idx, 'group_ID': group_id, 'Time': time_value,
                'Value_0': cum_v0, 'Value_1': cum_v1, 'Mask_0': 1.0, 'Mask_1': 1.0,
            })
        idx += 1
    group_id += 1

df = pd.DataFrame(region_data)
df.to_csv(RAW_CSV_OUT, index=False)

# ========== 第三步：强制使用 Train 的统计信息归一化 ==========
if not STATS_PATH.exists():
    raise FileNotFoundError(f"找不到统计文件 {STATS_PATH}。请确认 train 流程已完成。")

print(f"正在加载训练集归一化基准...")
stats_df = pd.read_csv(STATS_PATH).set_index("var")
in_m, in_s = stats_df.loc["Value_0", "mean"], stats_df.loc["Value_0", "std"]
out_m, out_s = stats_df.loc["Value_1", "mean"], stats_df.loc["Value_1", "std"]

df["Value_0"] = (df["Value_0"] - in_m) / (in_s + 1e-9)
df["Value_1"] = (df["Value_1"] - out_m) / (out_s + 1e-9)

df.to_csv(NORM_CSV_OUT, index=False)
print(f"处理完成！\n输出文件名: {NORM_CSV_OUT.name}")