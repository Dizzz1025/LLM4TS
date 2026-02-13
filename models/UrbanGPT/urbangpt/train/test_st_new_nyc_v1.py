import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pickle
import debugpy
debugpy.listen(("0.0.0.0", 5678))
print("Waiting for debugger attach...")
debugpy.wait_for_client()
# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件
from urbangpt.model.STLlama_new_nyc import STLlamaForCausalLM, STLlamaConfig

def test_inference(model_path, st_data_path, data_path, text_stats_path, gt_load_path, type, device="cuda"):
    """
    修改版：完全基于数据中的 mask_all 矩阵计算评价指标
    """
    model_path = os.path.abspath(model_path)
    st_data_path = os.path.abspath(st_data_path)

    print(f"Loading model from {model_path}...")
    config = STLlamaConfig.from_pretrained(model_path)
    model = STLlamaForCausalLM.from_pretrained(model_path, config=config)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # 1. 加载全量数据
    print("Loading test data and masks...")
    st_npz = np.load(st_data_path, allow_pickle=True)
    h_all = st_npz['h']
    mask_all = st_npz['mask']  # 关键：直接使用原始 mask
    gt_all = st_npz['gt']      # 关键：直接使用原始 gt

    with open(data_path, "r") as f:
        prompts_list = json.load(f)
    with open(text_stats_path, "r") as f:
        text_stats_dict = json.load(f)

    num_samples = len(prompts_list)
    all_preds, all_gts = [], []

    # 2. 循环推理
    print(f"Starting Inference based on provided masks for {num_samples} samples...")
    with torch.no_grad():
        for i in range(num_samples):
            sample_data = prompts_list[i]
            sample_key = str(sample_data.get('id', i))

            # 准备数值输入
            h = torch.from_numpy(h_all[i]).float().unsqueeze(0).to(device) # [1, 216, 50]
            # 这里 mask_obs 保持原始形状，内部模型会根据它识别哪些点是需要预测的
            mask_obs = torch.from_numpy(mask_all[i]).float().unsqueeze(0).to(device) # [1, 216]
            gt = torch.from_numpy(gt_all[i]).float().unsqueeze(0).to(device) # [1, 216, 2]
            # print('h.shape:', h.shape) # [1, 216, 50]
            # print('mask_obs.shape:', mask_obs.shape) # [1, 216]
            # print('gt.shape:', gt.shape) # [1, 216, 2]
            # 准备文本输入
            instruction = sample_data['conversations'][0]['value']
            inputs = tokenizer(instruction, return_tensors="pt").to(device)
            
            stat_text = text_stats_dict.get(sample_key, "[SEG_STATS] no_prior [/SEG_STATS]")
            stat_inputs = tokenizer(stat_text, return_tensors="pt", truncation=True).to(device)

            # 模型前向传播
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                stat_input_ids=stat_inputs.input_ids,
                stat_attention_mask=stat_inputs.attention_mask,
                h=h,
                mask_obs=mask_obs,
                return_dict=True,
            )
            
            hidden_states = outputs.hidden_states[-1] 
            
            # --- 核心修改部分：提取预测值和真值 ---
            # 1. 提取预测值：使用模型内部识别的 query 掩码
            final_query_mask = model.model._final_query_mask.bool() # [1, 58]
            preds = model.predict_head(hidden_states[final_query_mask]) # [1, 10]

            query_mask_2d = torch.zeros_like(mask_obs)
            query_mask_2d[:, 144:] = mask_obs[:, 144:]
            query_mask_2d = query_mask_2d.bool()
            # 2. 提取真值：直接使用从 npz 加载的 mask_all 对应位置
            # mask_all 中为 1 的位置即为我们要评估的 Ground Truth 位置 
            gt_aligned = gt[query_mask_2d] # [10, 2]

            # 处理对齐：防止模型输出长度与掩码长度因极小概率的不一致
            min_len = min(preds.shape[0], gt_aligned.shape[0])
            all_preds.append(preds[:min_len].cpu())
            all_gts.append(gt_aligned[:min_len].cpu())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples}...")

    # 3. 指标计算
    all_preds_cat = torch.cat(all_preds, dim=0) 
    all_gts_cat = torch.cat(all_gts, dim=0)
    
    # 总体指标
    mse = torch.nn.functional.mse_loss(all_preds_cat, all_gts_cat).item()
    mae = torch.nn.functional.l1_loss(all_preds_cat, all_gts_cat).item()

    # 分维度计算
    preds_x, gts_x = all_preds_cat[..., 0], all_gts_cat[..., 0]
    mse_x, mae_x = torch.nn.functional.mse_loss(preds_x, gts_x).item(), torch.nn.functional.l1_loss(preds_x, gts_x).item()

    preds_y, gts_y = all_preds_cat[..., 1], all_gts_cat[..., 1]
    mse_y, mae_y = torch.nn.functional.mse_loss(preds_y, gts_y).item(), torch.nn.functional.l1_loss(preds_y, gts_y).item()

    # 打印结果
    print("\n" + "="*40)
    print(f"Results for {type} (Using Original Masks):")
    print(f"Total Points Evaluated: {all_preds_cat.shape[0]}")
    print("-" * 40)
    print(f"{'Metric':<12} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 40)
    print(f"{'Overall':<12} | {mae:<10.4f} | {mse:<10.4f}")
    print(f"{'Dim X':<12} | {mae_x:<10.4f} | {mse_x:<10.4f}")
    print(f"{'Dim Y':<12} | {mae_y:<10.4f} | {mse_y:<10.4f}")
    print("="*40)

    # 保存
    np.savez(f"preds_gts_{type}.npz", preds=all_preds_cat.numpy(), gts=all_gts_cat.numpy())
    
    # 5. 可视化最后一个样本
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(all_gts[-1][:draw_len, 0], all_gts[-1][:draw_len, 1], label='GT', color='blue', marker='o')
    plt.plot(all_preds[-1][:draw_len, 0], all_preds[-1][:draw_len, 1], label='Pred', color='red', linestyle='--', marker='x')
    plt.legend()
    plt.title(f"Dual-Path Check (MSE: {mse:.6e})")
    plt.savefig("test_dual_visual.png")

if __name__ == "__main__":
    # TYPE = "taxi_supervised"    
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v4_2e_4/checkpoint-183120"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2021-12_gt.pkl"
    
    TYPE = "taxi_zeroshot"
    MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v6/checkpoint-34335"
    ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_zeroshot.npz"
    DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2020-01_text_stats.json"
    GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2020-01_gt.pkl"
    
    # TYPE = "bike_supervised"
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/bike/h_mask_gt_bike_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/bike/NYCbike_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/bike/NYCbike_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCbike_2021-12_gt.pkl"
    
    # TYPE = "bike_zeroshot"
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v3_2e_4/checkpoint-129165"
    # ST_DATA_PATH = f"{project_root}/data/NYC/bike/h_mask_gt_bike_zeroshot.npz"
    # DATA_PATH = f"{project_root}/data/NYC/bike/NYCbike_2020-01_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/bike/NYCbike_2020-01_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCbike_2020-01_gt.pkl"
    
    # TYPE = "crime_supervised"
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/crime/h_mask_gt_crime_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2021_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2021_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCcrime_2021_gt.pkl"
    
    # TYPE = "crime_zeroshot"
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/crime/h_mask_gt_crime_zeroshot.npz"
    # DATA_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2020_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2020_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCcrime_2020_gt.pkl"
    
    test_inference(MODEL_PATH, ST_DATA_PATH, DATA_PATH, TEXT_STATS_PATH, GT_LOAD_PATH, TYPE)