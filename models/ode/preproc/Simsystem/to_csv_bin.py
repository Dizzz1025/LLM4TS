import numpy as np
import pandas as pd

# ====== 0. 配置：时间 bin 大小 ======
# 比如原来 Time 是 0,1,2,...，这里设 bin_size=10 就是把
#  [0..9] 合成一个时间点，[10..19] 合成一个时间点，以此类推。
bin_size = 100   # 想不做合并就设成 1

# ====== 1. 读取原始 npy 数据 ======
loc = np.load('./data_charged_100/train/loc_train_charged5.npy', allow_pickle=True)
vel = np.load('./data_charged_100/train/vel_train_charged5.npy', allow_pickle=True)
timestamp = np.load('./data_charged_100/train/times_train_charged5.npy', allow_pickle=True)

assert vel.shape[0] == timestamp.shape[0] == loc.shape[0], 'loc / vel / time 的 num_sim 不一致！'
num_sim = loc.shape[0]
n = len(loc[0])

data = []
idx = 0
for i in range(num_sim):
    for j in range(n):
        step = loc[i][j].shape[0]
        ts = timestamp[i][j]
        for t in range(step):
            data.append({
                'ID': idx,
                'group_ID': i,
                'Time': ts[t],
                'Value_0': loc[i][j][t, 0],
                'Value_1': loc[i][j][t, 1],
                # 'Value_2': vel[i][j][t, 0],
                # 'Value_3': vel[i][j][t, 1],
                'Mask_0': 1.0,
                'Mask_1': 1.0,
                # 'Mask_2': 1.0,
                # 'Mask_3': 1.0,
            })
        idx += 1
df = pd.DataFrame(data)

# ====== 3. 对 Value_0 / Value_1 做归一化（标准化），用“合并后的数据”算均值方差 ======
min_0 = df['Value_0'].min()
max_0 = df['Value_0'].max()

min_1 = df['Value_1'].min()
max_1 = df['Value_1'].max()

# min_2 = df['Value_2'].min()
# max_2 = df['Value_2'].max()

# min_3 = df['Value_3'].min()
# max_3 = df['Value_3'].max()

eps = 1e-8  # 防止除以0

df['Value_0'] = (df['Value_0'] - min_0) * 2 / max(eps, (max_0 - min_0)) - 1
df['Value_1'] = (df['Value_1'] - min_1) * 2 / max(eps, (max_1 - min_1)) - 1
# df['Value_2'] = (df['Value_2'] - min_2) * 2 / max(eps, (max_2 - min_2)) - 1
# df['Value_3'] = (df['Value_3'] - min_3) * 2 / max(eps, (max_3 - min_3)) - 1

# ====== 4. 保存归一化后的数据 ======
df.to_csv('./data_charged_100/train/Simsystem_binned_norm.csv', index=False)

# ====== 5. 保存归一化参数 ======
# stats_df = pd.DataFrame({
#     'name': ['Value_0', 'Value_1', 'Value_2', 'Value_3'],
#     'min':  [min_0,     min_1,     min_2,     min_3],
#     'max':  [max_0,     max_1,     max_2,     max_3],
# })
stats_df = pd.DataFrame({
    'name': ['Value_0', 'Value_1'],
    'min':  [min_0,     min_1],
    'max':  [max_0,     max_1],
})
stats_df.to_csv('./data_charged_100/train/Simsystem_binned_norm_stats.csv', index=False)

print("Done! 原始样本数:", len(df), "，合并后样本数:", len(df))
print("时间 bin_size =", bin_size)
