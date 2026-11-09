import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

test_N = pd.read_csv('./data_springs/test/Simsystem_norm.csv')["ID"].nunique()
test_idx = np.arange(test_N)
fold_dir = f"../../gru_ode_bayes/datasets/Simsystem/folds_springs/small_chunk_fold_idx_0/"
os.makedirs(fold_dir, exist_ok=True)
np.save(fold_dir+f"test_idx.npy",test_idx)