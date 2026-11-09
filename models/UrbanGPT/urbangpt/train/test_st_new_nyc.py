import os
import sys
import time
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
import pickle

# import debugpy
# debugpy.listen(("0.0.0.0", 5678))
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件
# from urbangpt.model.STLlama_new_nyc import STLlamaForCausalLM, STLlamaConfig
from urbangpt.model.STLlama_new import STLlamaForCausalLM, STLlamaConfig


def _to_float(value):
    """把 list / tuple / numpy / torch scalar 统一转成 float，方便统计效率指标。"""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return float(value.item())
        return float(value.flatten()[0].item())
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return float(value.item())
        return float(value.reshape(-1)[0].item())
    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return 0.0
        return _to_float(value[0])
    return float(value)


def test_inference(
    model_path,
    st_data_path,
    data_path,
    text_stats_path,
    gt_load_path,
    type,
    device="cuda",
    max_samples=None,
    warmup=1,
):
    """
    NYC 双路推理测试脚本：
    1. 保留 NYC 数据的 gt_load 重写逻辑；
    2. 参考 test_sim，增加 latency / throughput / GPU memory / token count / LLM seq len 统计；
    3. max_samples=None 时默认跑完整测试集；如只想快速统计效率，可设置 max_samples=200。
    """
    latencies = []
    total_time = 0.0
    num_effective_samples = 0
    peak_gpu_mem_mb = 0.0
    effective_token_counts = []
    llm_seq_lens = []

    if "cuda" in str(device) and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

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
    h_all, mask_all, gt_all = st_npz["h"], st_npz["mask"], st_npz["gt"]

    # NYC 的预测 GT 单独从 pkl 写回到 144 之后的 query 位置
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
    if max_samples is not None:
        num_samples = min(num_samples, int(max_samples))

    all_preds, all_gts = [], []

    # 3. 循环推理
    print(f"Starting Dual-Path Inference on {num_samples} samples...")
    with torch.no_grad():
        for i in range(num_samples):
            # 获取 ID 匹配索引
            sample_data = prompts_list[i]
            sample_id = int(sample_data.get("id", i))
            sample_key = str(sample_id)

            # 准备数值输入
            h = torch.from_numpy(h_all[i]).float().unsqueeze(0).to(device)
            mask_obs = torch.from_numpy(mask_all[i]).float().unsqueeze(0).to(device)
            gt = torch.from_numpy(gt_all[i]).float().unsqueeze(0).to(device)

            # Path 1: 主 Instruction
            instruction = sample_data["conversations"][0]["value"]
            inputs = tokenizer(instruction, return_tensors="pt").to(device)

            # Path 2: 统计先验
            stat_text = text_stats_dict.get(sample_key, "[SEG_STATS] no_prior [/SEG_STATS]")
            stat_inputs = tokenizer(stat_text, return_tensors="pt", truncation=True).to(device)

            if "cuda" in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
            start_t = time.perf_counter()

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

            if "cuda" in str(device) and torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start_t

            # 跳过前 warmup 个样本，避免首轮 CUDA / cache 初始化影响平均延迟
            if i >= warmup:
                latencies.append(elapsed * 1000.0)
                total_time += elapsed
                num_effective_samples += 1

            if "cuda" in str(device) and torch.cuda.is_available():
                cur_peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
                peak_gpu_mem_mb = max(peak_gpu_mem_mb, cur_peak)

            # 统计 token 开销：优先使用模型内部记录的 _efficiency_meta
            meta = getattr(model.model, "_efficiency_meta", None)
            if meta is not None:
                try:
                    text_prompt_tokens = _to_float(meta["text_prompt_token_count"])
                    stat_text_tokens = _to_float(meta["stat_text_token_count"])
                    query_slots = _to_float(meta["query_slot_count"])
                    feat_latents = _to_float(meta["feat_latent_count"])
                    text_latents = _to_float(meta["text_latent_count"])

                    llm_seq_len = text_prompt_tokens + query_slots
                    effective_token_count = (
                        llm_seq_len + stat_text_tokens + feat_latents + text_latents
                    )

                    llm_seq_lens.append(llm_seq_len)
                    effective_token_counts.append(effective_token_count)
                except KeyError:
                    # 如果当前模型没有完整记录这些字段，则跳过 token 统计，不影响指标计算
                    pass

            # 提取隐状态
            hidden_states = outputs.hidden_states[-1]

            # 使用模型内部生成的变长掩码提取预测点
            final_query_mask = model.model._final_query_mask.bool()
            preds = model.predict_head(hidden_states[final_query_mask])

            # NYC 对齐真值：144 之后才是 query 区域，并且 mask 中每 6 步一个预测点
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

    if len(all_preds) == 0:
        raise RuntimeError("No predictions were collected. Please check num_samples / max_samples.")

    # 4. 指标计算
    all_preds_cat = torch.cat(all_preds, dim=0)
    all_gts_cat = torch.cat(all_gts, dim=0)

    # 若每个样本预测点数量一致，则保存成 (samples, query_points, dim)；否则保存为 object 会不方便后续处理
    same_len = all(p.shape == all_preds[0].shape for p in all_preds)
    if same_len:
        all_preds_final = torch.stack(all_preds, dim=0)
        all_gts_final = torch.stack(all_gts, dim=0)
    else:
        all_preds_final = all_preds_cat
        all_gts_final = all_gts_cat
        print("Warning: variable query lengths detected; saved flattened preds/gts.")

    # 总体指标
    overall_mse = torch.nn.functional.mse_loss(all_preds_cat, all_gts_cat).item()
    overall_mae = torch.nn.functional.l1_loss(all_preds_cat, all_gts_cat).item()

    # 分维度指标
    mae_x = torch.nn.functional.l1_loss(all_preds_cat[:, 0], all_gts_cat[:, 0]).item()
    mse_x = torch.nn.functional.mse_loss(all_preds_cat[:, 0], all_gts_cat[:, 0]).item()
    mae_y = torch.nn.functional.l1_loss(all_preds_cat[:, 1], all_gts_cat[:, 1]).item()
    mse_y = torch.nn.functional.mse_loss(all_preds_cat[:, 1], all_gts_cat[:, 1]).item()

    avg_latency_ms = float(np.mean(latencies)) if len(latencies) > 0 else 0.0
    throughput = num_effective_samples / total_time if total_time > 0 else 0.0
    avg_token_count = float(np.mean(effective_token_counts)) if len(effective_token_counts) > 0 else 0.0
    avg_llm_seq_len = float(np.mean(llm_seq_lens)) if len(llm_seq_lens) > 0 else 0.0

    # 5. 打印结果
    print("\n" + "=" * 70)
    print(f"Dual-Path NYC Test Results ({type})")
    print("=" * 70)
    print(f"Total Samples:            {num_samples}")
    print(f"Total Prediction Points:  {all_preds_cat.shape[0]}")
    print("-" * 70)
    print(f"{'Metric Scope':<15} | {'MAE':<12} | {'MSE':<12}")
    print("-" * 70)
    print(f"{'Overall':<15} | {overall_mae:<12.4f} | {overall_mse:<12.4f}")
    print(f"{'Dimension X':<15} | {mae_x:<12.4f} | {mse_x:<12.4f}")
    print(f"{'Dimension Y':<15} | {mae_y:<12.4f} | {mse_y:<12.4f}")
    print("-" * 70)
    print(f"Latency (ms/sample):      {avg_latency_ms:.4f}")
    print(f"Peak GPU Mem (MB):        {peak_gpu_mem_mb:.2f}")
    print(f"Throughput (samples/s):   {throughput:.4f}")
    print(f"Token Count:              {avg_token_count:.2f}")
    print(f"LLM Seq Len:              {avg_llm_seq_len:.2f}")
    print("=" * 70 + "\n")

    # 6. 保存结果
    save_name = f"new_{type}.npz"
    np.savez(save_name, preds=all_preds_final.numpy(), gts=all_gts_final.numpy())
    print(f"Results saved to {save_name}")

    # 7. 可视化最后一个样本
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(all_gts[-1][:draw_len, 0], all_gts[-1][:draw_len, 1], label="GT", color="blue", marker="o")
    plt.plot(all_preds[-1][:draw_len, 0], all_preds[-1][:draw_len, 1], label="Pred", color="red", linestyle="--", marker="x")
    plt.legend()
    plt.title(f"Dual-Path NYC Check ({type}, MSE: {overall_mse:.6e})")
    fig_name = f"test_dual_visual_{type}.png"
    plt.savefig(fig_name)
    print(f"Figure saved to {fig_name}")


if __name__ == "__main__":
    # TYPE = "taxi_supervised"    
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_k1/checkpoint-88290"
    # ST_DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/h_mask_gt_taxi_supervised.npz"
    # DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/NYCtaxi_2021-12_irregular_region_1_80_norm_formatted.json"
    # TEXT_STATS_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/NYCtaxi_2021-12_text_stats.json"
    # GT_LOAD_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/llmode/train/nyc_gt/new_data/NYCtaxi_2021-12_gt.pkl"
    
    # TYPE = "taxi_zeroshot"
    # MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_k90/checkpoint-170040"
    # ST_DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/h_mask_gt_taxi_zeroshot.npz"
    # DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/NYCtaxi_2020-01_irregular_region_81_160_norm_formatted.json"
    # TEXT_STATS_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/taxi/new_data/NYCtaxi_2020-01_text_stats.json"
    # GT_LOAD_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/llmode/train/nyc_gt/new_data/NYCtaxi_2020-01_gt.pkl"
    
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
    MODEL_PATH = f"/home/zhangdi24/LLM4TS/models/UrbanGPT/checkpoints/NYC_k90/checkpoint-170040"
    ST_DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/bike/h_mask_gt_bike_zeroshot.npz"
    DATA_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/bike/NYCbike_2020-01_irregular_region_81_160_norm_formatted.json"
    TEXT_STATS_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/data/NYC/bike/NYCbike_2020-01_text_stats.json"
    GT_LOAD_PATH = f"/home/zhangdi24/LLM4TS/models/LLMODE/llmode/train/nyc_gt/NYCbike_2020-01_gt.pkl"
    
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
    
    # 如果想和 test_sim 一样只快速测试 200 个样本，可以加 max_samples=200
    test_inference(MODEL_PATH, ST_DATA_PATH, DATA_PATH, TEXT_STATS_PATH, GT_LOAD_PATH, TYPE)