import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 加载合并后的训练集 CSV
train_csv = "/home/zhangdi24/LLM4TS/models/ode/preproc/NYC/NYCmerged/train/NYCmerged_irregular_region_1_80_norm.csv"
df_train = pd.read_csv(train_csv)

# 2. 根据统计信息定义 group_ID 范围
# 数据集 1: 0-265, 数据集 2: 266-534, 数据集 3: 535-653
dataset_stats = {
    "taxi":  {"start": 0,   "end": 265},
    "bike":  {"start": 266, "end": 534},
    "crime": {"start": 535, "end": 653}
}

# 提取各数据集唯一的 group_ID 列表
groups_taxi = df_train.loc[(df_train['group_ID'] >= dataset_stats['taxi']['start']) & 
                           (df_train['group_ID'] <= dataset_stats['taxi']['end']), 'group_ID'].unique()

groups_bike = df_train.loc[(df_train['group_ID'] >= dataset_stats['bike']['start']) & 
                           (df_train['group_ID'] <= dataset_stats['bike']['end']), 'group_ID'].unique()

groups_crime = df_train.loc[(df_train['group_ID'] >= dataset_stats['crime']['start']) & 
                            (df_train['group_ID'] <= dataset_stats['crime']['end']), 'group_ID'].unique()

print(f"Taxi groups: {len(groups_taxi)}, Bike groups: {len(groups_bike)}, Crime groups: {len(groups_crime)}")

Nfolds = 5
np.random.seed(432)

for fold in range(Nfolds):
    # --- 关键步骤：对三个部分分别进行 8:2 划分 ---
    train_g_tax, val_g_tax = train_test_split(groups_taxi, test_size=0.2)
    train_g_bik, val_g_bik = train_test_split(groups_bike, test_size=0.2)
    train_g_cri, val_g_cri = train_test_split(groups_crime, test_size=0.2)
    
    # 合并所有训练组 ID 和 验证组 ID
    train_groups = np.concatenate([train_g_tax, train_g_bik, train_g_cri])
    val_groups = np.concatenate([val_g_tax, val_g_bik, val_g_cri])
    
    # 根据 group_ID 提取对应的行 ID
    # 这里注意：df_train 中的 ID 应该是你合并时已经处理成全局唯一的 ID
    train_ids = df_train.loc[df_train["group_ID"].isin(train_groups), "ID"].unique()
    val_ids = df_train.loc[df_train["group_ID"].isin(val_groups), "ID"].unique()
    
    # 转换为 numpy 数组并排序，确保 ID 对应关系正确
    train_idx = np.array(sorted(train_ids), dtype=int)
    val_idx = np.array(sorted(val_ids), dtype=int)
    
    # 3. 保存索引文件
    fold_dir = f"/home/zhangdi24/LLM4TS/models/ode/gru_ode_bayes/datasets/NYC/folds_merged/fold_{fold}/"
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
        
    np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)
    
    print(f"Fold {fold}: 训练集 ID 总数={len(train_idx)}, 验证集 ID 总数={len(val_idx)}")
    print(f"        (采样明细 - Taxi: {len(val_g_tax)} groups, Bike: {len(val_g_bik)} groups, Crime: {len(val_g_cri)} groups)")

print("\nNYC 数据集分层 Fold 索引生成完毕。")