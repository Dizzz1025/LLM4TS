import os
import sys
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# 导入自定义模型组件
from urbangpt.model.STLlama_new import STLlamaForCausalLM, STLlamaConfig

def test_inference(model_path, st_data_path, data_path, text_stats_path, device="cuda"):
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
            query_mask_2d[:, 61:] = mask_obs[:, 61:]
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
    
    mse = torch.nn.functional.mse_loss(all_preds_cat, all_gts_cat).item()
    mae = torch.nn.functional.l1_loss(all_preds_cat, all_gts_cat).item()

    print("\n" + "="*30)
    print(f"Dual-Path Test Results:")
    print(f"Total Prediction Points: {all_preds_cat.shape[0]}")
    print(f"Overall MSE: {mse:.6e}") # 使用科学计数法
    print(f"Overall MAE: {mae:.6e}")
    print("="*30)

    # 保存结果
    np.savez("preds_gts_dual.npz", preds=all_preds_cat.numpy(), gts=all_gts_cat.numpy())
    
    # 5. 可视化最后一个样本
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(all_preds[-1]))
    plt.plot(all_gts[-1][:draw_len, 0], all_gts[-1][:draw_len, 1], label='GT', color='blue', marker='o')
    plt.plot(all_preds[-1][:draw_len, 0], all_preds[-1][:draw_len, 1], label='Pred', color='red', linestyle='--', marker='x')
    plt.legend()
    plt.title(f"Dual-Path Check (MSE: {mse:.6e})")
    plt.savefig("test_dual_visual.png")

if __name__ == "__main__":
    MODEL_PATH = f"{project_root}/checkpoints/UrbanGPT_v4/checkpoint-500"
    ST_DATA_PATH = f"{project_root}/data/Sim/test/h_mask_gt_charged_test.npz"
    DATA_PATH = f"{project_root}/data/Sim/test/Springs_test_prompts.json"
    TEXT_STATS_PATH = f"{project_root}/data/Sim/test/Simsystem_text_stats_charged.json"
    
    test_inference(MODEL_PATH, ST_DATA_PATH, DATA_PATH, TEXT_STATS_PATH)