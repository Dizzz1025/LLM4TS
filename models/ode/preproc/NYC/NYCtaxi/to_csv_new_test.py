import pickle
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import os
from pathlib import Path

# ================== 核心配置区 ==================
MODE = "zero-shot"              # 选项: 'supervised' 或 'zero-shot'
DATA_PREFIX = "NYCtaxi"          # 数据集前缀
# TIME_STR = "2021-12"             # 时间标识
# TIME_STR = "2020-01"
# 1. 设定输入输出目录 (保持一致)
# 假设数据存放在 ./data_accu/supervised 或 ./data_accu/zero-shot
SAVE_DIR = Path(f'./data_accu/test/') 

# 2. 根据模式动态设定读取的区域文件名
if MODE == "supervised":
    REGION_STR = "1_80"
    TIME_STR = "2021-12" 
elif MODE == "zero-shot":
    REGION_STR = "81_160"
    TIME_STR = "2020-01"
else:
    raise ValueError("MODE 必须是 'supervised' 或 'zero-shot'")

# 3. 强制指定【训练集】生成的归一化统计文件路径
# 无论当前跑哪个模式，都去读取 train 文件夹下的这个文件
STATS_PATH = Path('./data_accu/train') / f'{DATA_PREFIX}_delta_norm_stats.csv'

# 时间范围设定 (2021年12月)
if TIME_STR == "2021-12":
    START_TIME = datetime(2021, 12, 1, 0, 0)
    END_TIME   = datetime(2021, 12, 31, 23, 59)
elif TIME_STR == "2020-01":
    START_TIME = datetime(2020, 1, 1, 0, 0)
    END_TIME   = datetime(2020, 1, 15, 23, 59)

# 文件路径构建
INPUT_FILE = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.pkl'
RAW_CSV_OUT = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}.csv'
NORM_CSV_OUT = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_STR}_norm.csv'
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
        inflow, outflow = row[0], row[1]
        hour, minute, week, day, month, year = row[2:8]
        interval = int(row[8])
        if interval == 0: continue
        ts = datetime(int(year), int(month), int(day), int(hour), int(minute))
        region_row.append((ts, inflow, outflow, interval))
    raw_records[region] = region_row

# ========== 第二步：生成滑动窗口序列 ==========
x_scale = 1
windows_minutes = int(1.5 * 24 * 60) 
all_hours = []
t = START_TIME.replace(minute=0, second=0, microsecond=0)
while t <= END_TIME:
    all_hours.append(t)
    t += timedelta(hours=1)

region_data = []
idx = 0
group_id = 0
for i in range(0, len(all_hours) - int(windows_minutes / 60), 8):
    start_ts = all_hours[i]
    end_ts = start_ts + timedelta(minutes=windows_minutes)
    for region, all_rows in raw_records.items():
        for ts, inflow, outflow, interval in all_rows:
            if ts < start_ts: continue
            if ts >= end_ts: break
            delta_min = (ts - start_ts).total_seconds() / 60.0
            region_data.append({
                'ID': idx, 'group_ID': group_id, 'Time': delta_min / x_scale,
                'Value_0': inflow, 'Value_1': outflow, 'Mask_0': 1.0, 'Mask_1': 1.0,
            })
        idx += 1
    group_id += 1

df = pd.DataFrame(region_data)
df["Time"] = df["Time"] // 10
df.to_csv(RAW_CSV_OUT, index=False)

# ========== 第三步：强制使用 Train 的统计信息归一化 ==========
if not STATS_PATH.exists():
    raise FileNotFoundError(f"核心错误：找不到训练集归一化文件 {STATS_PATH}。请确保 train 流程已完成并生成了该文件。")

print(f"正在加载训练集归一化参数...")
stats_df = pd.read_csv(STATS_PATH)
key_col = "var" if "var" in stats_df.columns else "name"
stats = stats_df.set_index(key_col)[["mean", "std"]].to_dict("index")

inflow_mean, inflow_std = float(stats["Value_0"]["mean"]), float(stats["Value_0"]["std"])
outflow_mean, outflow_std = float(stats["Value_1"]["mean"]), float(stats["Value_1"]["std"])

df["Value_0"] = (df["Value_0"] - inflow_mean) / inflow_std
df["Value_1"] = (df["Value_1"] - outflow_mean) / outflow_std

df.to_csv(NORM_CSV_OUT, index=False)
print(f"处理完成！\n输入: {INPUT_FILE.name}\n输出: {NORM_CSV_OUT.name}")