import pandas as pd
import numpy as np
from pathlib import Path

# ================== 配置区 ==================
MODE = "zero-shot"  # 选项: 'supervised' 或 'zero-shot'
INPUT_PATH = "./data_accu/adjacency_matrix.csv"

# 定义基础路径
BASE_DIR = Path("./data_accu")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# 确保文件夹存在
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
# ===========================================

# 读取原始矩阵
df = pd.read_csv(INPUT_PATH)
print('原始矩阵形状:', df.shape)

# 根据模式决定切片范围和目标文件夹
if MODE == "zero-shot":
    # 提取 81-160 区域 (索引 80 到 159)
    adj_slice = df.iloc[80:160, 80:160]
    save_targets = [TEST_DIR]  # 仅保存到 test
    suffix = "_zero-shot"
else:
    # 提取 1-80 区域 (索引 0 到 79)
    adj_slice = df.iloc[:80, :80]
    save_targets = [TRAIN_DIR, TEST_DIR]  # train 和 test 都存
    suffix = ""

# 执行保存操作
for target_path in save_targets:
    csv_name = target_path / f"adjacency_matrix_80x80{suffix}.csv"
    npy_name = target_path / f"adjacency_matrix_80x80{suffix}.npy"
    
    # 保存 CSV (无表头，无索引)
    adj_slice.to_csv(csv_name, index=False, header=False)
    
    # 保存 Numpy
    adj_np = adj_slice.to_numpy()
    np.save(npy_name, adj_np)
    
    print(f"已保存至: {target_path.name} 文件夹")

print(f"\n--- 处理完成 ---")
print(f"模式: {MODE}")
print(f"矩阵形状: {adj_slice.shape}")