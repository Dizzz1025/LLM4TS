import numpy as np
import matplotlib.pyplot as plt
data = np.load("preds_gts.npz")
preds = data['preds'] # [N*T, 2]
gts = data["gts"]
print('preds.shape:', preds.shape)
seq_id = 2
start = seq_id * 23
end = (seq_id + 1) * 23
plt.plot(gts[start:end, 0], gts[start:end, 1], label='Ground Truth', color='blue', marker='o', markersize=4)
plt.plot(preds[start:end, 0], preds[start:end, 1], label='Prediction', color='red', linestyle='--', marker='x', markersize=4)
plt.grid(True, alpha=0.3)
plt.show()
