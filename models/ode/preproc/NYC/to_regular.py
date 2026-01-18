import numpy as np
import pandas as pd
from datetime import datetime
import pickle
import re
import calendar
from tqdm import tqdm
import os

def load_with_pickle(file_path):
    with open(file_path, 'rb') as f:
        result = pickle.load(f)
    return result

def save_with_pickle(result, file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(result, f)

def get_config(dataset, mode):
    """
    配置中心：管理不同数据集在不同模式下的参数
    """
    configs = {
        'NYCtaxi': {
            'train':     {'year': 2017, 'months': (1, 3),  'm_str': "01-03", 'regions': "1_80"},
            'test':      {'year': 2021, 'months': (12, 12), 'm_str': "12",    'regions': "1_80"},
            'zero-shot': {'year': 2020, 'months': (1, 1),  'm_str': "01",    'regions': "81_160"},
            'default_interval': '60min',
            'base_dir': 'data_accu' # Taxi 使用 data_accu
        },
        'NYCbike': {
            'train':     {'year': 2017, 'months': (4, 6),  'm_str': "04-06", 'regions': "1_80"},
            'test':      {'year': 2021, 'months': (12, 12), 'm_str': "12",    'regions': "1_80"},
            'zero-shot': {'year': 2020, 'months': (1, 1),  'm_str': "01",    'regions': "81_160"},
            'default_interval': '60min',
            'base_dir': 'data_accu' # Bike 使用 data_accu
        },
        'NYCcrime': {
            'train':     {'year': 2016, 'months': (1, 12), 'm_str': "2016-2018", 'regions': "1_80", 'multi_year': [2016, 2017, 2018]},
            'test':      {'year': 2021, 'months': (1, 12), 'm_str': "2021",      'regions': "1_80"},
            'zero-shot': {'year': 2020, 'months': (1, 12), 'm_str': "2020",      'regions': "81_160"},
            'default_interval': '4d',
            'base_dir': 'data' # Crime 使用 data
        }
    }
    return configs[dataset][mode], configs[dataset]['default_interval'], configs[dataset]['base_dir']

def run_resample(dataset_name, mode, user_interval=None):
    # 获取配置、默认间隔和基础目录
    conf, def_interval, base_dir = get_config(dataset_name, mode)
    
    # 优先使用用户指定的间隔
    hoped_interval = user_interval if user_interval else def_interval
    
    # 提取配置参数
    year = conf['year']
    m_range = conf['months']
    m_str = conf['m_str']
    regions = conf['regions']
    
    # 构建输入路径
    file_name = f'{dataset_name}_{str(year)}-{m_str}_irregular_region_{regions}.pkl'
    
    # 特殊处理：如果 NYCcrime 的文件名已经是具体的 2016-2018
    if dataset_name == 'NYCcrime' and mode == 'train':
        file_name = f'NYCcrime_2016-2018_irregular_region_1_80.pkl'

    # 根据数据集自动选择 input_dir (base_dir)
    file_path = os.path.join(base_dir, file_name)
    save_path = file_path.replace('irregular', 'regular').replace('.pkl', f'_{hoped_interval}.pkl')

    print(f"任务确认: 数据集={dataset_name}, 模式={mode}, 采样间隔={hoped_interval}")
    print(f"读取文件: {file_path}")

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    # 计算时间边界
    if 'multi_year' in conf:
        start_time = datetime(conf['multi_year'][0], 1, 1, 0, 0, 0)
        end_year = conf['multi_year'][-1]
        _, last_day = calendar.monthrange(end_year, 12)
        end_time = datetime(end_year, 12, last_day, 23, 59, 59)
    else:
        _, last_day = calendar.monthrange(year, m_range[1])
        start_time = datetime(year, m_range[0], 1, 0, 0, 0)
        end_time = datetime(year, m_range[1], last_day, 23, 59, 59)

    # 加载数据
    data = load_with_pickle(file_path)
    num_of_regions = len(data)
    print(f"区域总数: {num_of_regions}")

    resampled = []
    target_times = pd.date_range(start=start_time, end=end_time, freq=hoped_interval)

    for region in tqdm(range(num_of_regions), desc=f"正在处理 {dataset_name}-{mode}"):
        region_ts = data[region]
        
        # 自动识别数据列和时间列 (最后6列为时间)
        data_part = region_ts[:, :-6]
        time_part = region_ts[:, -6:]

        # 1. 转换原始时间戳 [hour, minute, week, day, month, year]
        region_timestamps = [
            datetime(int(row[5]), int(row[4]), int(row[3]), int(row[0]), int(row[1]))
            for row in time_part
        ]

        # 2. 构造 DataFrame
        df = pd.DataFrame(data_part, index=region_timestamps).sort_index()
        df = df[~df.index.duplicated(keep='last')]

        # 3. 时间插值
        interp_data = (
            df.reindex(df.index.union(target_times))
            .interpolate(method="time")
            .loc[target_times]
            .fillna(method='ffill') 
            .fillna(0)             
            .values
        )

        # 4. 构造新的时间特征列
        timestamps_info = np.array([
            [ts.hour, ts.minute, ts.weekday(), ts.day, ts.month, ts.year]
            for ts in target_times
        ])

        # 提取时间间隔数值 (支持 '60min' -> 60, '4d' -> 4)
        interval_val_match = re.findall(r'\d+', hoped_interval)
        interval_val = int(interval_val_match[0]) if interval_val_match else 0
        interval_col = np.full((len(timestamps_info), 1), interval_val)

        # 5. 合并
        region_result = np.concatenate((interp_data, timestamps_info, interval_col), axis=1)
        resampled.append(region_result)

    resampled = np.array(resampled)
    save_with_pickle(resampled, save_path)
    print(f"处理完成！保存至: {save_path}\n")

# ==========================================
# 用户操作区
# ==========================================

if __name__ == "__main__":
    # 示例：一次性循环处理所有配置
    for ds in ['NYCtaxi', 'NYCbike', 'NYCcrime']:
        for md in ['train', 'test', 'zero-shot']:
            run_resample(dataset_name=ds, mode=md)