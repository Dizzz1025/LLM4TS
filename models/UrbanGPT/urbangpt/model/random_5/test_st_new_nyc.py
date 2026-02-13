import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pickle

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件
from STLlama_random5 import STLlamaConfig, STLlamaForCausalLM

def test_inference(model_path, st_data_path, data_path, text_stats_path, gt_load_path, type, device="cuda"):
    """
    双路推理测试脚本：支持动态文本 Prompt 和统计先验输入。
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

    # 2. 加载全量数据
    print("Loading test data and prompts...")
    st_npz = np.load(st_data_path, allow_pickle=True)
    h_all, mask_all, gt_all = st_npz['h'], st_npz['mask'], st_npz['gt']
    # print('h_all:', h_all.shape) # (3280, 216, 50) -> zeroshot: 41*80个样本
    # print('mask_all:', mask_all.shape) # (3280, 216)
    # print('gt_all:', gt_all.shape) # (3280, 216, 2)
    with open(gt_load_path, "rb") as f:
        gt_load = pickle.load(f) # 形状: (Samples, 12, 2)
    print('gt_load.shape:', gt_load.shape)
    mask_all[:, 144:] = 0
    mask_all[:, 144::6] = 1
    gt_all[:, 144:, :] = 0
    gt_all[:, 144::6, :] = gt_load
    
    with open(data_path, "r") as f:
        prompts_list = json.load(f)
    with open(text_stats_path, "r") as f:
        text_stats_dict = json.load(f)

    num_samples = len(prompts_list)
    all_preds, all_gts = [], []

    # 3. 循环推理
    print(f"Starting Dual-Path Inference on {num_samples} samples...")
    with torch.no_grad():
        for i in range(num_samples):
            # 获取 ID 匹配索引
            sample_data = prompts_list[i]
            sample_id = int(sample_data.get('id', i))
            sample_key = str(sample_id)

            # 准备数值输入
            h = torch.from_numpy(h_all[i]).float().unsqueeze(0).to(device)
            mask_obs = torch.from_numpy(mask_all[i]).float().unsqueeze(0).to(device)
            gt = torch.from_numpy(gt_all[i]).float().unsqueeze(0).to(device)

            # 准备双路文本输入
            # Path 1: 主 Instruction
            instruction = sample_data['conversations'][0]['value']
            inputs = tokenizer(instruction, return_tensors="pt").to(device)
            
            # Path 2: 统计先验 (Stat Input)
            stat_text = text_stats_dict.get(sample_key, "[SEG_STATS] no_prior [/SEG_STATS]")
            stat_inputs = tokenizer(stat_text, return_tensors="pt", truncation=True).to(device)

            # 调用模型
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                stat_input_ids=stat_inputs.input_ids,
                stat_attention_mask=stat_inputs.attention_mask,
                h=h,
                mask_obs=mask_obs,
                return_dict=True,
            )
            
            # 提取隐状态 (最后一层)
            hidden_states = outputs.hidden_states[-1] 
            
            # 使用模型内部生成的变长掩码提取预测点
            final_query_mask = model.model._final_query_mask.bool()
            preds = model.predict_head(hidden_states[final_query_mask]) 

            # 对齐真值：对应 61 步之后的观测点
            query_mask_2d = torch.zeros_like(mask_obs)
            query_mask_2d[:, 144:] = mask_obs[:, 144:]
            query_mask_2d = query_mask_2d.bool()
            gt_aligned = gt[query_mask_2d]

            # 处理变长：记录预测点和真值点
            min_len = min(preds.shape[0], gt_aligned.shape[0])
            all_preds.append(preds[:min_len].cpu())
            all_gts.append(gt_aligned[:min_len].cpu())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples}...")

    # 4. 指标计算 (MSE & MAE)
    # 将所有样本的预测点拼接成大张量进行全局计算
    all_preds_cat = torch.cat(all_preds, dim=0) 
    all_gts_cat = torch.cat(all_gts, dim=0)
    
    # 1. 计算总体指标
    mse = torch.nn.functional.mse_loss(all_preds_cat, all_gts_cat).item()
    mae = torch.nn.functional.l1_loss(all_preds_cat, all_gts_cat).item()

    # 2. 分维度计算 (X 为索引 0, Y 为索引 1)
    # 取出所有样本的第 0 维 (X)
    preds_x = all_preds_cat[..., 0]
    gts_x = all_gts_cat[..., 0]
    mse_x = torch.nn.functional.mse_loss(preds_x, gts_x).item()
    mae_x = torch.nn.functional.l1_loss(preds_x, gts_x).item()

    # 取出所有样本的第 1 维 (Y)
    preds_y = all_preds_cat[..., 1]
    gts_y = all_gts_cat[..., 1]
    mse_y = torch.nn.functional.mse_loss(preds_y, gts_y).item()
    mae_y = torch.nn.functional.l1_loss(preds_y, gts_y).item()

    # 3. 打印结果
    print("\n" + "="*40)
    print(f"Dual-Path Test Results (Detailed):")
    print(f"Total Prediction Points: {all_preds_cat.numel() // 2}") # 总点数
    print("-" * 40)
    print(f"{'Metric':<12} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 40)
    print(f"{'Overall':<12} | {mae:<10.4f} | {mse:<10.4f}")
    print(f"{'Dimension X':<12} | {mae_x:<10.4f} | {mse_x:<10.4f}")
    print(f"{'Dimension Y':<12} | {mae_y:<10.4f} | {mse_y:<10.4f}")
    print("="*40)

    # 保存结果
    np.savez(f"preds_gts_dual_{type}.npz", preds=all_preds_cat.numpy(), gts=all_gts_cat.numpy())
    
    # 5. 可视化最后一个样本
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(all_gts[-1][:draw_len, 0], all_gts[-1][:draw_len, 1], label='GT', color='blue', marker='o')
    plt.plot(all_preds[-1][:draw_len, 0], all_preds[-1][:draw_len, 1], label='Pred', color='red', linestyle='--', marker='x')
    plt.legend()
    plt.title(f"Dual-Path Check (MSE: {mse:.6e})")
    plt.savefig("test_dual_visual.png")

if __name__ == "__main__":
    project_root = '/home/zhangdi24/LLM4TS/models/UrbanGPT'
    TYPE = "taxi_supervised"    
    MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/urbangpt/model/random_5/checkpoints/NYC_random5/checkpoint-81750"
    ST_DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/h_mask_gt_taxi_supervised.npz"
    DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2021-12_text_stats.json"
    GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/new_data/NYCtaxi_2021-12_gt.pkl"
    # TYPE = "taxi_supervised"    
    # MODEL_PATH = f"{project_root}/checkpoints/NYC_v4_2e_4/checkpoint-183120"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2021-12_gt.pkl"
    
    TYPE = "taxi_zeroshot"
    MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/urbangpt/model/random_5/checkpoints/NYC_random5/checkpoint-81750"
    ST_DATA_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/data/NYC/taxi/h_mask_gt_taxi_zeroshot.npz"
    DATA_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/data/NYC/taxi/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    TEXT_STATS_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/data/NYC/taxi/NYCtaxi_2020-01_text_stats.json"
    GT_LOAD_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/urbangpt/train/nyc_gt/NYCtaxi_2020-01_gt.pkl"
    
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