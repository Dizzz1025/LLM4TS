import json
import numpy as np
import pandas as pd
import os
from collections import Counter

def generate_nyc_prompts_final(csv_path, taxi_poi_path, bc_poi_path, selected_regions_path=None):
    """
    根据 CSV 中去重后的 (dataset_id, ID) 生成指定 JSON 格式的数据
    格式：{"id": 0, "conversations": [{"from": "human", "value": "Task: ..."}]}
    value 内部不再包含编号
    """
    # 1. 环境准备：提取文件名判断数据组 (1_80 或 81_160)
    dataset_name = os.path.splitext(os.path.basename(csv_path))[0]
    
    # 2. 载入外部资源文件
    with open(taxi_poi_path, 'r', encoding='utf-8') as f:
        taxi_poi_data = json.load(f)
    with open(bc_poi_path, 'r', encoding='utf-8') as f:
        bc_poi_data = json.load(f)
        
    bc_real_ids = []
    if selected_regions_path:
        data = np.load(selected_regions_path)
        bc_real_ids = data['arr_0'] if 'arr_0' in data else data['selected_regions']

    # 3. 读取 CSV 并按唯一 ID 组合去重
    df = pd.read_csv(csv_path)
    if 'dataset_id' in df.columns:
        unique_df = df[['dataset_id', 'ID']].drop_duplicates().sort_values(by=['dataset_id', 'ID'])
    else:
        unique_df = df[['ID']].drop_duplicates().sort_values(by='ID')
    
    unique_items = unique_df.to_dict('records')
    formatted_data = []

    # 4. 遍历 ID 生成指定格式的对象
    for idx, item in enumerate(unique_items):
        raw_id = int(item['ID']) 
        d_id = item.get('dataset_id', None)

        # 4.1 判定数据集任务类型
        if d_id == 1 or "taxi" in dataset_name.lower():
            curr_type = 'taxi'
        elif d_id == 2 or "bike" in dataset_name.lower():
            curr_type = 'bike'
        elif d_id == 3 or "crime" in dataset_name.lower():
            curr_type = 'crime'
        else:
            curr_type = 'taxi'

        # 4.2 计算组偏移与索引
        offset = 80 if "81_160" in dataset_name else 0
        idx_within_group = raw_id % 80

        # 4.3 物理 ID 映射逻辑
        if curr_type == 'taxi':
            # Taxi 物理 ID 从 1 开始
            target_rid = str(offset + idx_within_group + 1)
            poi_source = taxi_poi_data
            task_str = "Task: Predict the future inflow and outflow at the query slots."
        else:
            # Bike/Crime 物理 ID 从 0 开始，通过 NPZ 映射
            idx_in_npz = offset + idx_within_group
            target_rid = str(bc_real_ids[idx_in_npz])
            poi_source = bc_poi_data
            
            if curr_type == 'bike':
                task_str = "Task: Predict the future inflow and outflow at the query slots."
            else: # crime
                task_str = "Task: Predict the future Robbery and Burglary at the query slots."

        # 4.4 提取 POI
        region_pois = poi_source.get(target_rid, [])
        if not region_pois:
            region_info = "Region: []"
        else:
            borough = region_pois[0].get('borough_name', 'NYC')
            categories = [p['category_name'] for p in region_pois]
            top3 = [cat for cat, _ in Counter(categories).most_common(3)]
            region_info = f"Region: {borough} area with dominant POIs: {', '.join(top3)}."

        # 4.5 组装 Prompt Value (去掉了 idx 编号)
        prompt_value = f"{task_str}\n{region_info}\nQuerySlots: "

        # 4.6 组织成要求的字典结构
        entry = {
            "id": idx,
            "conversations": [
                {
                    "from": "human",
                    "value": prompt_value
                }
            ]
        }
        formatted_data.append(entry)

    return formatted_data

# ================= 保存与运行 =================

csv_path = '/home/zhangdi24/LLM4TS/models/ode/preproc/NYC/NYCbike/data_accu/test/NYCbike_2021-12_irregular_region_1_80_norm.csv'
taxi_poi = 'zone_poi.json'
bc_poi = 'region_poi.json'
npz_file = 'selected_regions.npz'

# 执行生成
final_output = generate_nyc_prompts_final(csv_path, taxi_poi, bc_poi, npz_file)

# 自动生成输出文件名
output_json = f"{os.path.splitext(os.path.basename(csv_path))[0]}_formatted.json"

# 写入 JSON
with open(output_json, 'w', encoding='utf-8') as f:
    json.dump(final_output, f, ensure_ascii=False, indent=4)

print(f"--- 处理完成 ---")
print(f"生成的 Prompt 总数: {len(final_output)}")
print(f"文件保存路径: {output_json}")