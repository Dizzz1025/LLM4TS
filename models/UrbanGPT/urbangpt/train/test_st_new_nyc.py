import os
import sys
import json
import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件（用于非-PEFT checkpoint 兜底）
from urbangpt.model.STQwen2_new import STQwen2ForCausalLM, STQwen2Config


def unwrap_st_model(model):
    """
    将 PEFT 包装的模型解包到你真正的 STQwen2ForCausalLM。
    目标返回对象应当拥有：
      - .model （STQwen2Model）
      - .predict_head
    """
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model
    return model


def load_inference_model(model_path: str, device: str = "cuda"):
    model_path = os.path.abspath(model_path)
    compute_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float16

    quantization_config = None
    if torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    print(f"Loading model from {model_path}...")

    # 优先按 PEFT checkpoint 加载（你现在训练保存出来的大概率就是这种）
    try:
        model = AutoPeftModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        print("Loaded PEFT checkpoint.")
    except Exception as e:
        print(f"PEFT load failed: {e}")
        print("Falling back to plain STQwen2 checkpoint loading...")
        config = STQwen2Config.from_pretrained(model_path)
        model = STQwen2ForCausalLM.from_pretrained(
            model_path,
            config=config,
            quantization_config=quantization_config,
            torch_dtype=compute_dtype,
            device_map="auto" if torch.cuda.is_available() else None,
        )

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def pad_varlen_tensors(tensor_list, pad_value=np.nan):
    """
    将 [Li, D] 的变长张量列表 pad 成 [N, Lmax, D]，便于保存。
    """
    if len(tensor_list) == 0:
        return np.array([]), np.array([])

    lengths = np.array([t.shape[0] for t in tensor_list], dtype=np.int64)
    feat_dim = tensor_list[0].shape[-1]
    max_len = int(lengths.max())

    out = np.full((len(tensor_list), max_len, feat_dim), pad_value, dtype=np.float32)
    for i, t in enumerate(tensor_list):
        arr = t.cpu().numpy().astype(np.float32)
        out[i, :arr.shape[0], :] = arr
    return out, lengths


def test_inference(model_path, st_data_path, data_path, text_stats_path, gt_load_path, type, device="cuda"):
    """
    双路推理测试脚本：支持动态文本 Prompt 和统计先验输入。
    适配：
      - STQwen2
      - PEFT/LoRA checkpoint
      - 4bit 推理
    """
    model_path = os.path.abspath(model_path)
    st_data_path = os.path.abspath(st_data_path)

    model, tokenizer = load_inference_model(model_path, device=device)
    st_model = unwrap_st_model(model)

    # 2. 加载全量数据
    print("Loading test data and prompts...")
    st_npz = np.load(st_data_path, allow_pickle=True)
    h_all, mask_all, gt_all = st_npz["h"], st_npz["mask"], st_npz["gt"]

    with open(gt_load_path, "rb") as f:
        gt_load = pickle.load(f)  # 形状: (Samples, 12, 2)
    print("gt_load.shape:", gt_load.shape)

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

    print(f"Starting Dual-Path Inference on {num_samples} samples...")
    with torch.no_grad():
        for i in range(num_samples):
            sample_data = prompts_list[i]
            sample_id = int(sample_data.get("id", i))
            sample_key = str(sample_id)

            # 数值输入：这里要和 train 对齐，用 sample_id 索引，不是 i
            h = torch.from_numpy(h_all[sample_id]).float().unsqueeze(0).to(device)
            mask_obs = torch.from_numpy(mask_all[sample_id]).float().unsqueeze(0).to(device)
            gt = torch.from_numpy(gt_all[sample_id]).float().unsqueeze(0).to(device)

            # 文本输入
            instruction = sample_data["conversations"][0]["value"]
            inputs = tokenizer(
                instruction,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)

            stat_text = text_stats_dict.get(sample_key, "[SEG_STATS] no_prior [/SEG_STATS]")
            stat_inputs = tokenizer(
                stat_text,
                return_tensors="pt",
                truncation=True,
                max_length=tokenizer.model_max_length,
            ).to(device)

            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                stat_input_ids=stat_inputs.input_ids,
                stat_attention_mask=stat_inputs.attention_mask,
                h=h,
                mask_obs=mask_obs,
                return_dict=True,
                output_hidden_states=True,
            )

            hidden_states = outputs.hidden_states[-1]

            # _final_query_mask 在 STQwen2Model 内部
            final_query_mask = st_model.model._final_query_mask.bool()
            preds = st_model.predict_head(hidden_states[final_query_mask])

            # 对齐真值：对应预测区间内的观测点
            query_mask_2d = torch.zeros_like(mask_obs)
            query_mask_2d[:, 144:] = mask_obs[:, 144:]
            query_mask_2d = query_mask_2d.bool()
            gt_aligned = gt[query_mask_2d]

            min_len = min(preds.shape[0], gt_aligned.shape[0])
            all_preds.append(preds[:min_len].cpu())
            all_gts.append(gt_aligned[:min_len].cpu())

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples}...")

    # 4. 指标计算
    all_preds_cat = torch.cat(all_preds, dim=0)
    all_gts_cat = torch.cat(all_gts, dim=0)

    mse = torch.nn.functional.mse_loss(all_preds_cat, all_gts_cat).item()
    mae = torch.nn.functional.l1_loss(all_preds_cat, all_gts_cat).item()

    preds_x = all_preds_cat[..., 0]
    gts_x = all_gts_cat[..., 0]
    mse_x = torch.nn.functional.mse_loss(preds_x, gts_x).item()
    mae_x = torch.nn.functional.l1_loss(preds_x, gts_x).item()

    preds_y = all_preds_cat[..., 1]
    gts_y = all_gts_cat[..., 1]
    mse_y = torch.nn.functional.mse_loss(preds_y, gts_y).item()
    mae_y = torch.nn.functional.l1_loss(preds_y, gts_y).item()

    print("\n" + "=" * 40)
    print("Dual-Path Test Results (Detailed):")
    print(f"Total Prediction Points: {all_preds_cat.numel() // 2}")
    print("-" * 40)
    print(f"{'Metric':<12} | {'MAE':<10} | {'MSE':<10}")
    print("-" * 40)
    print(f"{'Overall':<12} | {mae:<10.4f} | {mse:<10.4f}")
    print(f"{'Dimension X':<12} | {mae_x:<10.4f} | {mse_x:<10.4f}")
    print(f"{'Dimension Y':<12} | {mae_y:<10.4f} | {mse_y:<10.4f}")
    print("=" * 40)

    # 保存结果（变长样本做 pad，同时保存真实长度）
    all_preds_final, pred_lengths = pad_varlen_tensors(all_preds)
    all_gts_final, gt_lengths = pad_varlen_tensors(all_gts)

    np.savez(
        f"new_{type}.npz",
        preds=all_preds_final,
        gts=all_gts_final,
        pred_lengths=pred_lengths,
        gt_lengths=gt_lengths,
        preds_cat=all_preds_cat.cpu().numpy(),
        gts_cat=all_gts_cat.cpu().numpy(),
    )

    # 5. 可视化最后一个样本
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(all_gts[-1][:draw_len, 0], all_gts[-1][:draw_len, 1], label="GT", color="blue", marker="o")
    plt.plot(all_preds[-1][:draw_len, 0], all_preds[-1][:draw_len, 1], label="Pred", color="red", linestyle="--", marker="x")
    plt.legend()
    plt.title(f"Dual-Path Check (MSE: {mse:.6e})")
    plt.savefig("test_dual_visual.png")


if __name__ == "__main__":
    # TYPE = "taxi_supervised"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2021-12_gt.pkl"

    TYPE = "taxi_zeroshot"
    MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_zeroshot.npz"
    DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2020-01_text_stats.json"
    GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2020-01_gt.pkl"

    # TYPE = "taxi_supervised"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/new_data/NYCtaxi_2021-12_gt.pkl"

    # TYPE = "taxi_zeroshot"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/h_mask_gt_taxi_zeroshot.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/new_data/NYCtaxi_2020-01_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/new_data/NYCtaxi_2020-01_gt.pkl"

    # TYPE = "bike_supervised"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/bike/h_mask_gt_bike_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/bike/NYCbike_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/bike/NYCbike_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCbike_2021-12_gt.pkl"

    # TYPE = "bike_zeroshot"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/bike/h_mask_gt_bike_zeroshot.npz"
    # DATA_PATH = f"{project_root}/data/NYC/bike/NYCbike_2020-01_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/bike/NYCbike_2020-01_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCbike_2020-01_gt.pkl"

    # TYPE = "crime_supervised"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/crime/h_mask_gt_crime_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2021_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2021_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCcrime_2021_gt.pkl"

    # TYPE = "crime_zeroshot"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/crime/h_mask_gt_crime_zeroshot.npz"
    # DATA_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2020_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/crime/NYCcrime_2020_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCcrime_2020_gt.pkl"

    test_inference(MODEL_PATH, ST_DATA_PATH, DATA_PATH, TEXT_STATS_PATH, GT_LOAD_PATH, TYPE)
