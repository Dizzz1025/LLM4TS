import pandas as pd
import numpy as np

def merge_and_stat_datasets(file_list, output_csv):
    """
    file_list: 列表，顺序为 [Taxi, Bike, Crime]
    """
    combined_dfs = []
    stats = {}
    
    current_id_offset = 0
    current_group_offset = 0
    
    for i, file_path in enumerate(file_list):
        df = pd.read_csv(file_path)
        ds_id = i + 1  # 数据集标识符 1, 2, 3
        
        # 1. 记录当前数据集的原始 group 数量
        unique_groups = df['group_ID'].nunique()
        unique_ids = df['ID'].nunique()
        
        # 2. 这里的逻辑很关键：确保合并后的 ID 和 group_ID 是连续递增的
        # 先将当前 ID 归零化，再加全局偏移量
        df['ID'] = (df['ID'] - df['ID'].min()) + current_id_offset
        df['group_ID'] = (df['group_ID'] - df['group_ID'].min()) + current_group_offset
        
        # 3. 添加数据集标识
        df['dataset_id'] = ds_id
        
        # 记录统计信息
        stats[ds_id] = {
            "name": file_path.split('.')[0],
            "group_start": current_group_offset,
            "group_count": unique_groups,
            "id_count": unique_ids
        }
        
        # 4. 更新下一次循环的偏移量
        current_id_offset += unique_ids
        current_group_offset += unique_groups
        
        combined_dfs.append(df)
        
    # 保存结果
    full_df = pd.concat(combined_dfs, axis=0, ignore_index=True)
    full_df.to_csv(output_csv, index=False)
    
    print("--- 数据集统计信息 ---")
    for ds_id, info in stats.items():
        print(f"数据集 {ds_id} ({info['name']}): Groups {info['group_start']} 到 {info['group_start'] + info['group_count'] - 1} (共 {info['group_count']} 个)")
        
    return stats

# 执行合并
dataset_stats = merge_and_stat_datasets(['./NYCtaxi/data_accu/train/NYCtaxi_2017_01-03_irregular_region_1_80_norm.csv', 
                                         './NYCbike/data_accu/train/NYCbike_2017_04-06_irregular_region_1_80_norm.csv', 
                                         './NYCcrime/data/train/NYCcrime_2016-2018_irregular_region_1_80_norm.csv'], 
                                        './NYCmerged/train/NYCmerged_irregular_region_1_80_norm.csv')