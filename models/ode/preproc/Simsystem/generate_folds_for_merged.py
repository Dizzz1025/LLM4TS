import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 1. 加载合并后的训练集 CSV
train_csv = "/home/zhangdi24/LLM4TS/models/ode/preproc/Simsystem/data_merged/train/Simsystem_binned_norm.csv"
df_train = pd.read_csv(train_csv)
last_group_id = df_train['group_ID'].iloc[-1]
print(f"df_train['group_ID'] 的最后一行值是: {last_group_id}")

# 2. 确定分界点（确保你能区分哪些 group_ID 属于 springs，哪些属于 charged）
# 假设你之前合并时 springs 的数量是 N_spr_groups
# 如果不确定，可以通过你原始 data_springs 的 group_ID 最大值来判断
# 这里的逻辑是：分别提取两类数据集的唯一 group_ID 列表
# 假设我们通过 group_ID 的范围来区分（因为 charged 加上了 offset）
# 如果你记得之前的 offset_group，请直接替换下面的 limit
limit = 20000  # 这是一个假设值，建议用具体的偏移界限
groups_springs = df_train.loc[df_train['group_ID'] < limit, 'group_ID'].unique()
groups_charged = df_train.loc[df_train['group_ID'] >= limit, 'group_ID'].unique()

print(f"Springs groups: {len(groups_springs)}, Charged groups: {len(groups_charged)}")

Nfolds = 5
np.random.seed(432)

for fold in range(Nfolds):
    # --- 关键步骤：分别对两部分进行 8:2 划分 ---
    train_g_spr, val_g_spr = train_test_split(groups_springs, test_size=0.2)
    train_g_cha, val_g_cha = train_test_split(groups_charged, test_size=0.2)
    
    # 合并两部分的组 ID，这样保证了验证集里两类数据比例是 1:1（假设原始数据量相等）
    train_groups = np.concatenate([train_g_spr, train_g_cha])
    val_groups = np.concatenate([val_g_spr, val_g_cha])
    
    # 根据 group_ID 提取对应的行 ID
    train_ids = df_train.loc[df_train["group_ID"].isin(train_groups), "ID"].unique()
    val_ids = df_train.loc[df_train["group_ID"].isin(val_groups), "ID"].unique()
    
    train_idx = np.array(sorted(train_ids), dtype=int)
    val_idx = np.array(sorted(val_ids), dtype=int)
    
    # 3. 保存索引文件
    fold_dir = f"/home/zhangdi24/LLM4TS/models/ode/gru_ode_bayes/datasets/Simsystem/folds_merged/fold_{fold}/"
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
        
    np.save(os.path.join(fold_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(fold_dir, "val_idx.npy"), val_idx)
    
    print(f"Fold {fold}: 训练集大小={len(train_idx)}, 验证集大小={len(val_idx)}")

print("\n所有 Fold 索引生成完毕（已跳过测试集）。")
