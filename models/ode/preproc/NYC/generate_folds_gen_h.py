import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

test_N = pd.read_csv('./NYCtaxi/data_accu/test/NYCtaxi_2020-01_irregular_region_81_160_norm.csv')["ID"].nunique()
# full_idx = np.arange(N)
test_idx = np.arange(test_N)
fold_dir = f"../../gru_ode_bayes/datasets/NYC/folds_gen_h_taxi_zeroshot/small_chunk_fold_idx_0/"
os.makedirs(fold_dir, exist_ok=True)
np.save(fold_dir+f"test_idx.npy",test_idx)