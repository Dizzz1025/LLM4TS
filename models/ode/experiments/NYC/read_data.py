import numpy as np
data = np.load('/home/zhangdi24/LLM4TS/models/ode/experiments/NYC/h_mask_gt.npz', allow_pickle=True)
h = data['h']
print(h.shape)