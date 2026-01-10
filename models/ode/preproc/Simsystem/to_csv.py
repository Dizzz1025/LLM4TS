import numpy as np
import pandas as pd

loc = np.load('./data/train/loc_train_springs5.npy', allow_pickle=True)
timestamp = np.load('./data/train/times_train_springs5.npy', allow_pickle=True)

assert loc.shape[0] == timestamp.shape[0], 'loc和time的num_sim不一致！'
num_sim = loc.shape[0]
n = len(loc[0])
data = []
idx = 0
for i in range(num_sim):
    for j in range(n):
        step = loc[i][j].shape[0]
        for t in range(step):
            data.append({
                'ID': idx,
                'Time': timestamp[i][j][t, 0], #[num_sim, n, (47, 1)]
                'Value_0': loc[i][j][t, 0],
                'Value_1': loc[i][j][t, 1],
                'Mask_0': 1.0,
                'Mask_1': 1.0,           
            })
        idx += 1
df = pd.DataFrame(data)
df.to_csv('./Simsystem.csv', index=False)
# ========= 2. 对 Value_0 / Value_1 做归一化（标准化） =========
# 计算每一列的均值和标准差
mean_0 = df['Value_0'].mean()
std_0  = df['Value_0'].std(ddof=0)  # ddof=0 对应总体标准差，如果想要样本标准差用 ddof=1

mean_1 = df['Value_1'].mean()
std_1  = df['Value_1'].std(ddof=0)

# 防止除 0（一般不会，但保险一点）
eps = 1e-8
std_0 = std_0 if std_0 > eps else eps
std_1 = std_1 if std_1 > eps else eps

# 替换为归一化后的数值
df['Value_0'] = (df['Value_0'] - mean_0) / std_0
df['Value_1'] = (df['Value_1'] - mean_1) / std_1

# ========= 3. 保存归一化后的数据 =========
df.to_csv('./Simsystem_norm.csv', index=False)

# ========= 4. 把归一化用到的均值和方差保存成 csv =========
stats_df = pd.DataFrame({
    'name': ['Value_0', 'Value_1'],
    'mean': [mean_0,     mean_1],
    'std':  [std_0,      std_1],
})
stats_df.to_csv('./Simsystem_norm_stats.csv', index=False)