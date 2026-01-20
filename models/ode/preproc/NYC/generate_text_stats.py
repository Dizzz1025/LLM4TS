import pandas as pd
import numpy as np
import json

# ================== Configuration ==================
TIME_THRESHOLD = 144.0  # Base time step for historical window
SEGMENT_K = 4           # Number of segments
EXPECTED_STEPS = 24.0   # Baseline steps for obs_rate (24 steps expected in 144 range)

# File paths
DATA_PREFIX = "NYCbike"
MODE = "test"
TIME_STR = "2021-12"
REGION_RANGE = '1_80'
NORM_CSV = f'./{DATA_PREFIX}/data_accu/{MODE}/{DATA_PREFIX}_{TIME_STR}_irregular_region_{REGION_RANGE}_norm.csv'
OUTPUT_JSON = f'./{DATA_PREFIX}/data_accu/{MODE}/{DATA_PREFIX}_{TIME_STR}_text_stats.json'
# ===================================================

def generate_nyc_text_prior(csv_path, threshold, k):
    # Load normalized CSV
    df = pd.read_csv(csv_path)
    
    # Identify variable names based on dataset_id
    # 1: Taxi, 2: Bike -> inflow/outflow
    # 3: Crime -> Selected from [robbery, burglary, grand larceny, felony assault]
    ds_id = df['dataset_id'].iloc[0] if 'dataset_id' in df.columns else 1
    
    if ds_id in [1, 2]:
        v0_name, v1_name = "inflow", "outflow"
    elif ds_id == 3:
        # Selecting the first two specified crime types
        v0_name, v1_name = "robbery", "burglary"
    else:
        v0_name, v1_name = "value_0", "value_1"

    seg_len = threshold / k  # e.g., 36.0
    results = {}

    # Process by ID
    for obj_id, group in df.groupby('ID'):
        # Filter observations within the 144 time-step window
        obs_history = group[group['Time'] <= threshold].sort_values('Time')
        
        text_parts = [f"[SEG_STATS K={int(k)}]"]
        
        # Calculate statistics for each of the 4 segments
        for i in range(1, int(k) + 1):
            t_start = (i - 1) * seg_len
            t_end = i * seg_len
            
            # Segment data range: [t_start, t_end)
            seg_data = obs_history[(obs_history['Time'] >= t_start) & (obs_history['Time'] < t_end)]
            
            if not seg_data.empty:
                v0_mean, v1_mean = seg_data['Value_0'].mean(), seg_data['Value_1'].mean()
                v0_max, v0_min = seg_data['Value_0'].max(), seg_data['Value_0'].min()
                v1_max, v1_min = seg_data['Value_1'].max(), seg_data['Value_1'].min()
                
                # Calculate observation rate and cap it at 100%
                o_rate_raw = (len(seg_data) / (EXPECTED_STEPS / k)) * 100
                o_rate_pct = min(o_rate_raw, 100.0)
                
                # Highly structured text for better LLM semantic extraction
                line = (f"seg{i}: {v0_name}_mean={v0_mean:.2f}, {v1_name}_mean={v1_mean:.2f}, "
                        f"{v0_name}=[max: {v0_max:.2f}, min: {v0_min:.2f}], "
                        f"{v1_name}=[max: {v1_max:.2f}, min: {v1_min:.2f}], "
                        f"obs_rate={o_rate_pct:.2f}%")
            else:
                line = f"seg{i}: no_observation"
            
            text_parts.append(line)
        
        # max_gap: distance between 144 and the last observation time step
        if not obs_history.empty:
            last_obs_time = obs_history['Time'].max()
            max_gap = threshold - last_obs_time
        else:
            max_gap = threshold
            
        text_parts.append(f"max_gap={max_gap:.2f}")
        text_parts.append("[/SEG_STATS]")
        
        # Store description as string key for the JSON dictionary
        results[str(int(obj_id))] = " ".join(text_parts)
    
    return results

if __name__ == "__main__":
    print(f"Starting prioritized text generation for {DATA_PREFIX}...")
    stats_dict = generate_nyc_text_prior(NORM_CSV, TIME_THRESHOLD, SEGMENT_K)
    
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(stats_dict, f, ensure_ascii=False, indent=4)
        
    print(f"File saved: {OUTPUT_JSON}")
    if stats_dict:
        sample_id = list(stats_dict.keys())[0]
        print(f"\nExample format (ID {sample_id}):\n{stats_dict[sample_id]}")