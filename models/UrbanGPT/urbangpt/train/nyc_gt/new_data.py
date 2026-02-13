import pickle
import numpy as np
import os

# --- 配置路径 ---
GT_DIR = './'       
SAVE_DIR = './new_data/'

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

# --- 待删除索引 ---
# 2021 Supervised 索引
idx_2021 = sorted(list(set([207, 227, 447, 467, 638, 687, 707, 718, 927, 947, 958, 1407, 1647, 1887, 1907, 2127, 2147, 2318, 2367, 2387, 2398, 2607, 2638, 3327, 3567, 3807, 4047, 4287] + [47, 78, 92, 122, 129, 362, 442, 572, 602, 609, 798, 812, 842, 849, 878, 922, 1038, 1082, 1087, 1089, 1118, 1242, 1247, 1322, 1562, 1569, 1758, 1772, 1998, 2042, 2202, 2252, 2289, 2332, 2478, 2492, 2522, 2527, 2529, 2602, 2718, 2769, 2798, 2847, 3002, 3167, 3198, 3242, 3438, 3452, 3482, 3489, 3678, 3692, 3722, 3729, 3918, 3932, 3962, 3969, 4078, 4202, 4398, 4478, 4762, 4767, 4847, 5162, 5187, 5402, 5427, 5482, 5567, 5642, 5647, 5667, 5678, 5907, 6047, 6067, 6147, 6287, 6307, 6367, 6387, 6547, 6602, 6627, 6682, 6867, 7007, 7082, 7107])))

# 2020 Zeroshot 索引
idx_2020 = sorted(list(set([131, 211, 451, 461, 691, 700, 701, 851, 931, 1091, 1171, 1180, 1181, 1411, 1420, 1421, 1651, 1659, 1660, 1661, 1891, 1899, 1900, 1901, 2131, 2140, 2141, 2301, 2371, 2380, 2381, 2611, 2620, 2621, 2781, 2851, 2861, 3091, 3100, 3101] + [19, 33, 51, 57, 63, 67, 147, 291, 499, 531, 537, 617, 707, 739, 771, 947, 993, 1011, 1017, 1023, 1027, 1037, 1073, 1103, 1107, 1251, 1257, 1491, 1497, 1731, 1737, 1820, 1971, 2007, 2211, 2217, 2227, 2247, 2387, 2451, 2457, 2467, 2524, 2593, 2627, 2673, 2691, 2697, 2703, 2707, 2717, 2753, 2783, 2787, 2899, 2931, 2937, 3171, 3177, 3259])))

def process_pkl(file_path, indices):
    print(f"正在处理 Taxi GT 文件: {os.path.basename(file_path)}")
    with open(file_path, 'rb') as f:
        gt_data = pickle.load(f)
    
    # 过滤数据
    if isinstance(gt_data, np.ndarray):
        new_gt = np.delete(gt_data, indices, axis=0)
    elif isinstance(gt_data, list):
        new_gt = [item for i, item in enumerate(gt_data) if i not in indices]
    else:
        print(f"⚠️ 格式跳过: {type(gt_data)}")
        return

    # 保存
    save_path = os.path.join(SAVE_DIR, os.path.basename(file_path))
    with open(save_path, 'wb') as f:
        pickle.dump(new_gt, f)
    print(f"  - 样本数从 {len(gt_data)} 降至 {len(new_gt)}")

# --- 执行过滤 ---
for file_name in os.listdir(GT_DIR):
    # 核心修改：增加对 'taxi' 关键字的检查
    if file_name.endswith('.pkl') and ('taxi' in file_name.lower()):
        full_path = os.path.join(GT_DIR, file_name)
        
        if '2021' in file_name:
            process_pkl(full_path, idx_2021)
        elif '2020' in file_name:
            process_pkl(full_path, idx_2020)
        else:
            print(f"🔎 找到 taxi 文件但年份不匹配，跳过: {file_name}")

print("\n✨ Taxi Ground Truth 处理完毕！")