import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ================== 核心参数区 ==================
MODE = "supervised"              # 选项: 'zero-shot' 或 'supervised'
DATA_PREFIX = "NYCcrime"
SELECTED_VALUE_INDICES = [0, 1]  # 数值位

# 窗口与步进
HIS, PRE = 24, 12                # HIS 24, PRE 12 (总窗口 36)
STRIDE = 2                       # 滑动窗步长

SAVE_DIR = Path(f'./data/test/') 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

if MODE == "zero-shot":
    REGION_STR, TIME_STR = "81_160", "2020"
elif MODE == "supervised":
    REGION_STR, TIME_STR = "1_80", "2021"

# 输入路径
INPUT_FILE = Path(f'./data/test') / f'{DATA_PREFIX}_{TIME_STR}_regular_region_{REGION_STR}.pkl'
# 输出路径
PKL_OUT = SAVE_DIR / f'{DATA_PREFIX}_{TIME_STR}_gt.pkl'
STATS_PATH = Path('./data/train') / f'{DATA_PREFIX}_delta_norm_stats.csv'
# ===============================================

print(f"--- 生成 GT 模式 (先滑动再归一化): {MODE.upper()} ---")

# 1. 加载数据
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"找不到输入文件: {INPUT_FILE}")

with open(INPUT_FILE, 'rb') as f:
    data_raw = np.array(pickle.load(f))
    
num_regions, total_T, num_feats = data_raw.shape
print(f"输入数据形状: {data_raw.shape}")

# 2. 加载训练集归一化基准 (用于最后一步)
if not STATS_PATH.exists():
    raise FileNotFoundError(f"找不到统计文件 {STATS_PATH}")
stats_df = pd.read_csv(STATS_PATH).set_index("var")
val_mu = np.array([stats_df.loc["Value_0", "mean"], stats_df.loc["Value_1", "mean"]])
val_std = np.array([stats_df.loc["Value_0", "std"], stats_df.loc["Value_1", "std"]])

# 3. 滑动窗口切片与累积计算 (使用原始数据)
gt_samples = []

print(f"开始滑动采样并计算累积值...")
# t 为 36 步窗口结束的位置
for t in range(HIS + PRE, total_T + 1, STRIDE):
    # 截取原始数据窗口 [Region, 36, Feat]
    window = data_raw[:, t-(HIS+PRE):t, :].copy().astype(np.float32)
    
    # 对每个区域的数值位进行累积 (基于原始数值累积)
    window[:, :, 0:2] = np.cumsum(window[:, :, 0:2], axis=1)
    
    # 仅取后 12 步 (从索引 24 到 35) 且仅保留数值位 (0:2)
    gt_window = window[:, HIS:, 0:2] 
    
    for r in range(num_regions):
        gt_samples.append(gt_window[r]) # (12, 2)

# 4. 封装为大张量
final_gt_array = np.array(gt_samples) # 形状: (Samples, 12, 2)

# 5. 归一化处理 (最后执行)
print(f"正在使用训练集基准进行归一化...")
# final_gt_array 形状是 (Samples, 12, 2)，val_mu 形状是 (2,)
# 利用广播机制进行计算
final_gt_array = (final_gt_array - val_mu) / (val_std + 1e-9)

# 6. 保存结果
print(f"最终 GT 输出形状: {final_gt_array.shape}")
with open(PKL_OUT, 'wb') as f:
    pickle.dump(final_gt_array, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"✅ 处理完成！文件保存至: {PKL_OUT}")