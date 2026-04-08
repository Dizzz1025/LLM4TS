import os
import sys
import json
import time
import pickle
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
import transformers

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件
from urbangpt.model.STLlama_new_nyc import STLlamaForCausalLM, STLlamaConfig


def setup_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def build_text_inputs(tokenizer, instruction, stat_text, device):
    """
    按 train 脚本的方式构造文本输入：
    1) 主 instruction 直接 tokenizer
    2) 统计先验 stat_text 单独 tokenizer
    3) 显式构造 position_ids
    """
    input_ids = tokenizer(
        instruction,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.to(device)

    stat_input_ids = tokenizer(
        stat_text,
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids.to(device)

    attention_mask = input_ids.ne(tokenizer.pad_token_id)
    stat_attention_mask = stat_input_ids.ne(tokenizer.pad_token_id)

    batch_size, seq_len = input_ids.shape
    position_ids = (
        torch.arange(0, seq_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(batch_size, seq_len)
        .clone()
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "stat_input_ids": stat_input_ids,
        "stat_attention_mask": stat_attention_mask,
    }


def test_inference(
    model_path,
    st_data_path,
    data_path,
    text_stats_path,
    gt_load_path,
    run_type,
    device=None,
):
    setup_seed(42)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.abspath(model_path)
    st_data_path = os.path.abspath(st_data_path)

    print(f"Loading model from {model_path} ...")
    config = STLlamaConfig.from_pretrained(model_path)

    # 跟当前脚本保持一致：CUDA 下用 bf16
    model = STLlamaForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
    )
    model.to(device)
    model.eval()

    model_dtype = next(model.parameters()).dtype
    print("model_dtype:", model_dtype)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token or tokenizer.eos_token
    if tokenizer.pad_token is None:
        raise ValueError("Tokenizer is missing both unk_token and eos_token; cannot set pad_token.")

    print("Loading test data and prompts ...")
    st_npz = np.load(st_data_path, allow_pickle=True)
    h_all = st_npz["h"]
    mask_all = st_npz["mask"]
    gt_all = st_npz["gt"]

    with open(gt_load_path, "rb") as f:
        gt_load = pickle.load(f)
    print("gt_load.shape:", gt_load.shape)

    # 保持你原来的 future mask / gt 回填逻辑
    mask_all[:, 144:] = 0
    mask_all[:, 144::6] = 1
    gt_all[:, 144:, :] = 0
    gt_all[:, 144::6, :] = gt_load

    with open(data_path, "r", encoding="utf-8") as f:
        prompts_list = json.load(f)

    with open(text_stats_path, "r", encoding="utf-8") as f:
        text_stats_dict = json.load(f)

    num_samples = len(prompts_list)
    all_preds = []
    all_gts = []
    latency_list = []

    print(f"Starting inference on {num_samples} samples ...")

    with torch.no_grad():
        for i in range(num_samples):
            sample_data = prompts_list[i]
            sample_id = int(sample_data.get("id", i))
            sample_key = str(sample_id)

            if sample_id >= len(h_all):
                raise IndexError(
                    f"sample_id={sample_id} out of range for h_all with len={len(h_all)}"
                )

            # 关键修复：数值输入和模型 dtype 对齐
            h = torch.from_numpy(h_all[sample_id]).unsqueeze(0).to(device=device, dtype=model_dtype)
            mask_obs = torch.from_numpy(mask_all[sample_id]).unsqueeze(0).to(device=device, dtype=model_dtype)
            gt = torch.from_numpy(gt_all[sample_id]).unsqueeze(0).to(device=device, dtype=torch.float32)

            instruction = sample_data["conversations"][0]["value"]
            stat_text = text_stats_dict.get(sample_key, "[SEG_STATS] no_prior_available [/SEG_STATS]")

            text_inputs = build_text_inputs(tokenizer, instruction, stat_text, device)

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()

            # 用 autocast 稳一下混合精度
            if device.startswith("cuda"):
                with torch.autocast(device_type="cuda", dtype=model_dtype):
                    outputs = model(
                        input_ids=text_inputs["input_ids"],
                        position_ids=text_inputs["position_ids"],
                        attention_mask=text_inputs["attention_mask"],
                        stat_input_ids=text_inputs["stat_input_ids"],
                        stat_attention_mask=text_inputs["stat_attention_mask"],
                        h=h,
                        mask_obs=mask_obs,
                        return_dict=True,
                        output_hidden_states=True,
                    )
            else:
                outputs = model(
                    input_ids=text_inputs["input_ids"],
                    position_ids=text_inputs["position_ids"],
                    attention_mask=text_inputs["attention_mask"],
                    stat_input_ids=text_inputs["stat_input_ids"],
                    stat_attention_mask=text_inputs["stat_attention_mask"],
                    h=h,
                    mask_obs=mask_obs,
                    return_dict=True,
                    output_hidden_states=True,
                )

            if device.startswith("cuda"):
                torch.cuda.synchronize()
            latency_list.append(time.time() - t0)

            if outputs.hidden_states is None:
                raise RuntimeError(
                    "outputs.hidden_states is None. "
                    "Please check whether your model forward supports output_hidden_states=True."
                )

            hidden_states = outputs.hidden_states[-1]

            final_query_mask = model.model._final_query_mask.bool()
            preds = model.predict_head(hidden_states[final_query_mask])

            # 预测拿回 float32 再算指标，稳一点
            preds = preds.float()

            query_mask_2d = torch.zeros_like(mask_obs, dtype=torch.bool)
            query_mask_2d[:, 144:] = mask_obs[:, 144:].bool()
            gt_aligned = gt[query_mask_2d]

            min_len = min(preds.shape[0], gt_aligned.shape[0])
            preds = preds[:min_len].detach().cpu()
            gt_aligned = gt_aligned[:min_len].detach().cpu()

            all_preds.append(preds)
            all_gts.append(gt_aligned)

            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples} ...")

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

    print("\n" + "=" * 50)
    print("Dual-Path Test Results")
    print("-" * 50)
    print(f"Total Prediction Points: {all_preds_cat.shape[0]}")
    print(f"{'Metric':<12} | {'MAE':<12} | {'MSE':<12}")
    print("-" * 50)
    print(f"{'Overall':<12} | {mae:<12.6f} | {mse:<12.6f}")
    print(f"{'Dimension X':<12} | {mae_x:<12.6f} | {mse_x:<12.6f}")
    print(f"{'Dimension Y':<12} | {mae_y:<12.6f} | {mse_y:<12.6f}")
    print("-" * 50)
    print(f"Avg latency/sample: {np.mean(latency_list):.6f} s")
    print("=" * 50)

    np.savez(
        f"new_{run_type}.npz",
        preds=np.array([x.numpy() for x in all_preds], dtype=object),
        gts=np.array([x.numpy() for x in all_gts], dtype=object),
        preds_cat=all_preds_cat.numpy(),
        gts_cat=all_gts_cat.numpy(),
        latency=np.array(latency_list),
        mae=np.array([mae]),
        mse=np.array([mse]),
    )

    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(
        all_gts[-1][:draw_len, 0].numpy(),
        all_gts[-1][:draw_len, 1].numpy(),
        label="GT",
        marker="o",
    )
    plt.plot(
        all_preds[-1][:draw_len, 0].numpy(),
        all_preds[-1][:draw_len, 1].numpy(),
        label="Pred",
        linestyle="--",
        marker="x",
    )
    plt.legend()
    plt.title(f"Dual-Path Check (MSE: {mse:.6e})")
    plt.savefig(f"test_dual_visual_{run_type}.png", dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    # TYPE = "taxi_supervised"    
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_v3_daice/checkpoint-47415"
    # ST_DATA_PATH = f"{project_root}/data/NYC/taxi/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"{project_root}/data/NYC/taxi/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"{project_root}/urbangpt/train/nyc_gt/NYCtaxi_2021-12_gt.pkl"
    
    # TYPE = "taxi_zeroshot"
    # MODEL_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/checkpoints/NYC_llama/checkpoint-156960"
    # ST_DATA_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/taxi/new_data/h_mask_gt_taxi_zeroshot.npz"
    # DATA_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/taxi/new_data/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/taxi/new_data/NYCtaxi_2020-01_text_stats.json"
    # GT_LOAD_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/urbangpt/train/nyc_gt/NYCtaxi_2020-01_gt.pkl"
    
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
    
    TYPE = "bike_zeroshot"
    MODEL_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/checkpoints/NYC_llama/checkpoint-156960"
    ST_DATA_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/bike/h_mask_gt_bike_zeroshot.npz"
    DATA_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/bike/NYCbike_2020-01_irregular_region_81_160_norm_formatted.json"
    TEXT_STATS_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/data/NYC/bike/NYCbike_2020-01_text_stats.json"
    GT_LOAD_PATH = f"/root/autodl-tmp/LLM4TS/models/UrbanGPT/urbangpt/train/nyc_gt/NYCbike_2020-01_gt.pkl"
    
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