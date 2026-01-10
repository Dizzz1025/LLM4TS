import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os


train_csv = "/home/zhangdi24/LLM-for-Spatio-Temporal-Graph-of-Beihang-University/gru_ode_bayes/preproc/Simsystem/data_v4/train/Simsystem_binned_norm.csv"
df_train = pd.read_csv(train_csv)
N = df_train['group_ID'].nunique()
test_N = pd.read_csv('/home/zhangdi24/LLM-for-Spatio-Temporal-Graph-of-Beihang-University/gru_ode_bayes/preproc/Simsystem/data_v4/test/Simsystem_binned_norm.csv')["ID"].nunique()
# full_idx = np.arange(N)
test_idx = np.arange(test_N)
Nfolds = 5

np.random.seed(432)

for fold in range(Nfolds):
    train_groups, val_groups = train_test_split(np.arange(N),test_size=0.2)
    train_ids = df_train.loc[df_train["group_ID"].isin(train_groups), "ID"].unique()
    val_ids = df_train.loc[df_train["group_ID"].isin(val_groups), "ID"].unique()
    train_idx = np.array(sorted(train_ids), dtype=int)
    val_idx = np.array(sorted(val_ids), dtype=int)
    fold_dir = f"/home/zhangdi24/LLM-for-Spatio-Temporal-Graph-of-Beihang-University/gru_ode_bayes/gru_ode_bayes/datasets/Simsystem/folds_v4/small_chunk_fold_idx_{fold}/"
    if not os.path.exists(fold_dir):
        os.makedirs(fold_dir)
    np.save(fold_dir+f"train_idx.npy",train_idx)
    np.save(fold_dir+f"val_idx.npy",val_idx)
    np.save(fold_dir+f"test_idx.npy",test_idx)
