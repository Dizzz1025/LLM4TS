import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os

N = pd.read_csv("/home/zhangdi24/LLM-for-Spatio-Temporal-Graph-of-Beihang-University/gru_ode_bayes/preproc/NYCtaxi/data_delta_bin/NYCtaxi_2017_06_irregular_region_1_80_less8_bin_norm.csv")["ID"].nunique()
test_idx = np.arange(N)
fold_dir = f"/home/zhangdi24/LLM-for-Spatio-Temporal-Graph-of-Beihang-University/gru_ode_bayes/gru_ode_bayes/datasets/NYCtaxi/folds_7120/"
if not os.path.exists(fold_dir):
    os.makedirs(fold_dir)
np.save(fold_dir+f"test_idx.npy",test_idx)

