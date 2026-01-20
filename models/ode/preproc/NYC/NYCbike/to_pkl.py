import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

# ================== 核心配置区 ==================
MODE = "zero-shot"              # 选项: 'supervised' 或 'zero-shot'
DATA_PREFIX = "NYCbike"         
SAVE_DIR = Path(f'./data_accu/test/') 

if MODE == "supervised":
    REGION_STR, TIME_STR = "1_80", "2021-12" 
elif MODE == "zero-shot":
    REGION_STR, TIME_STR = "81_160", "2020-01"

# 统计信息路径（保持读取 train）
STATS_PATH = Path('./data_accu/train') / f'{DATA_PREFIX}_delta_norm_stats.csv'

# Regular 数据处理参数
WINDOW_SIZE = 36  # 总窗口长度 (HIS 24 + PRE 12)
PRE_LEN = 12      # 我们只需要后 12 步作为 GT
STRIDE = 8        # 滑动步长

INPUT_FILE = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_regular_region_{REGION_STR}.pkl'
# 修改文件名要求
PKL_OUT    = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_gt.pkl'
# ===============================================

print(f"--- 生成 GT 模式: {MODE.upper()} ---")

# ========== 第一步：加载数据与统计信息 ==========
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

with open(INPUT_FILE, 'rb') as f:
    # 预期 Regular 数据形状: (Region, Total_T, Feature)
    data_raw = np.array(pickle.load(f))

num_regions, total_t, num_feats = data_raw.shape
print(f"原始数据形状: {data_raw.shape}")

# 加载归一化参数
if not STATS_PATH.exists():
    raise FileNotFoundError(f"找不到统计文件: {STATS_PATH}")

stats_df = pd.read_csv(STATS_PATH).set_index("var")
in_m, in_s = stats_df.loc["Value_0", "mean"], stats_df.loc["Value_0", "std"]
out_m, out_s = stats_df.loc["Value_1", "mean"], stats_df.loc["Value_1", "std"]

# ========== 第二步：标准化与截取后12步 ==========
gt_samples = []

# 1. 先进行全局归一化
data_norm = data_raw.copy().astype(np.float32)
data_norm[:, :, 0] = (data_norm[:, :, 0] - in_m) / (in_s + 1e-9)
data_norm[:, :, 1] = (data_norm[:, :, 1] - out_m) / (out_s + 1e-9)

print(f"正在滑动切片并提取后 {PRE_LEN} 步真值...")

# 2. 滑动窗口切片
# t 是窗口的起始点
for t in range(0, total_t - WINDOW_SIZE + 1, STRIDE):
    # 截取该时间窗内所有区域的后 12 步数据
    # 索引 [24:36] 对应后 12 步
    gt_window = data_norm[:, t + (WINDOW_SIZE - PRE_LEN) : t + WINDOW_SIZE, 0:2]
    
    for r in range(num_regions):
        # 提取单个区域的 12 步 GT (12, 2)
        sample_gt = gt_window[r].copy()
        gt_samples.append(sample_gt)

# ========== 第三步：保存结果 ==========
final_gt_array = np.array(gt_samples) # 形状: (Samples, 12, 2)

print(f"最终 GT 形状: {final_gt_array.shape}")
with open(PKL_OUT, 'wb') as f:
    pickle.dump(final_gt_array, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ 处理完成！GT 文件保存至: {PKL_OUT}")