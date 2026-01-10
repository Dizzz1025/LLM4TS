import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

loc = np.load('./data_v3_02/train/loc_train_springs5.npy', allow_pickle=True)
loc_ori = np.load('./data_v3_02/train/loc_ori_train_springs5.npy', allow_pickle=True)
timestamp = np.load('./data_v3_02/train/times_train_springs5.npy', allow_pickle=True)
seq = 1
node = 0
loc_ij = loc[seq][node]
loc_ori_ij = loc_ori[seq, :, :, node]
print('loc_ori.shape:', loc_ori.shape)

plt.figure(figsize=(6, 5))

plt.plot(
    loc_ij[:, 0],
    loc_ij[:, 1],
    marker='o',
    label=f'Seq {seq}, Node {node}'
)
plt.plot(
    loc_ori_ij[:, 0],
    loc_ori_ij[:, 1],
    marker='o',
    label=f'Seq {seq}, Node {node}'
)

plt.scatter(
    loc_ij[0, 0],
    loc_ij[0, 1],
    s=100,
    color='red',
    label='Start',
    zorder=5
)

plt.title(f'Trajectory of Node {node} (Seq {seq})')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.axis('equal')
plt.tight_layout()

plt.savefig(f'loc_seq{seq}_node{node}.png', dpi=300)
plt.close()