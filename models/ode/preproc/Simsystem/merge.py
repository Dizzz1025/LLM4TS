import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

def merge_datasets(mode="train"):
    # --- 1. 路径设置 ---
    base_path = "/home/zhangdi24/LLM4TS/models/ode/preproc/Simsystem"
    springs_dir = f"{base_path}/data_springs/{mode}"
    charged_dir = f"{base_path}/data_charged/{mode}"
    output_dir = f"{base_path}/data_merged/{mode}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 2. 合并 CSV 文件并处理 ID ---
    df_spr = pd.read_csv(f"{springs_dir}/Simsystem_binned_norm.csv")
    df_cha = pd.read_csv(f"{charged_dir}/Simsystem_binned_norm.csv")
    # 获取 springs 的最大值作为偏移量
    offset_group = df_spr['group_ID'].max() + 1
    offset_id = df_spr['ID'].max() + 1

    # 对 charged 数据集进行偏移
    df_cha['group_ID'] = df_cha['group_ID'] + offset_group
    df_cha['ID'] = df_cha['ID'] + offset_id

    # 合并并保存 CSV
    df_combined = pd.concat([df_spr, df_cha], axis=0, ignore_index=True)
    df_combined.to_csv(f"{output_dir}/Simsystem_binned_norm.csv", index=False)
    
    print(f"[{mode}] CSV 合并完成: {len(df_spr)} (spr) + {len(df_cha)} (cha) -> {len(df_combined)} 行")

    # --- 3. 合并 Numpy 文件 (Edges & Features) ---
    # 定义需要合并的文件前缀（根据你截图里的文件列表）
    files_to_merge = [
        "edges"
    ]

    for prefix in files_to_merge:
        # 匹配对应的文件名（注意 springs5 后缀）
        spr_file = f"{springs_dir}/{prefix}_{mode}_springs5.npy"
        cha_file = f"{charged_dir}/{prefix}_{mode}_charged5.npy" # 假设 charged 文件夹里叫这个名字
        
        # 如果 charged 文件命名也是 springs5，请改用这一行:
        # cha_file = f"{charged_dir}/{prefix}_{mode}_springs5.npy"

        if os.path.exists(spr_file) and os.path.exists(cha_file):
            data_spr = np.load(spr_file)
            data_cha = np.load(cha_file)
            print('data_str.shape:', data_spr.shape)
            print('data_cha.shape:', data_cha.shape)
            # 在第一个维度（样本数维度）进行拼接
            data_combined = np.concatenate([data_spr, data_cha], axis=0)
            
            # 保存为 concat 命名的文件
            save_path = f"{output_dir}/{prefix}_{mode}_merged.npy"
            np.save(save_path, data_combined)
            print(f"合并 Numpy: {prefix} -> {data_combined.shape}")

    return df_combined

# --- 执行合并 ---
# 处理训练集
df_train_final = merge_datasets("train")