import numpy as np
import pandas as pd
from pathlib import Path

# ================== 核心配置区 ==================
MODE = "zero-shot"  # 选项: 'supervised' 或 'zero-shot'
GRID_H = 47          # 网格列数 (46*47 划分)

NPZ_PATH = "./selected_regions.npz"
BASE_DIR = Path("./data_accu")
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR = BASE_DIR / "test"

# 确保文件夹存在
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
# ===============================================

# 1. 加载 162 个选定区域的原始网格 ID
data = np.load(NPZ_PATH)
all_selected_indices = data['arr_0'] 

# 2. 根据模式筛选目标 80 个区域
if MODE == "supervised":
    # 选前 80 个区域
    target_indices = all_selected_indices[:80]
    save_targets = [TRAIN_DIR, TEST_DIR]
    suffix = ""
    print(f"--- 模式: SUPERVISED (区域 1-80) ---")
elif MODE == "zero-shot":
    # 选第 81-160 个区域
    target_indices = all_selected_indices[80:160]
    save_targets = [TEST_DIR]
    suffix = "_zero-shot"
    print(f"--- 模式: ZERO-SHOT (区域 81-160) ---")

# 3. 构造 80x80 邻接矩阵 (排除自环)
num_nodes = len(target_indices)
adj_80 = np.zeros((num_nodes, num_nodes))

for i in range(num_nodes):
    id_a = target_indices[i]
    r_a, c_a = id_a // GRID_H, id_a % GRID_H
    
    for j in range(num_nodes):
        # 【修改点】自己和自己不算相邻
        if i == j:
            continue
            
        id_b = target_indices[j]
        r_b, c_b = id_b // GRID_H, id_b % GRID_H
        
        # 计算地理位置上的行列距离
        dist_r = abs(r_a - r_b)
        dist_c = abs(c_a - c_b)
        
        # 邻接定义：在原始网格中处于 8-邻域 (上下左右斜对角)
        if dist_r <= 1 and dist_c <= 1:
            adj_80[i, j] = 1

# 4. 保存文件
for target_path in save_targets:
    csv_out = target_path / f"adjacency_matrix_80x80{suffix}.csv"
    npy_out = target_path / f"adjacency_matrix_80x80{suffix}.npy"
    
    # 保存 CSV (无表头，无索引)
    pd.DataFrame(adj_80).to_csv(csv_out, index=False, header=False)
    # 保存 Numpy
    np.save(npy_out, adj_80)
    
    print(f"保存成功: {target_path.name}/{csv_out.name}")

# 验证对角线是否全为 0
diag_sum = np.trace(adj_80)
print(f"\n校验: 对角线之和 = {diag_sum} (应为 0)")
print(f"总边数 (不含自环): {int(adj_80.sum())}")