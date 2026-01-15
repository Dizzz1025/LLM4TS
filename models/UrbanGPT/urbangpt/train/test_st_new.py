import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from torch.nn import MSELoss

# 1. 环境路径设置
current_dir = os.path.dirname(os.path.abspath(__file__))
# 假设脚本位于 project_root/scripts/test 目录下，根据实际情况调整
project_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

print('Project Root:', project_root)

# 导入自定义模型组件
from urbangpt.model.STLlama_new import STLlamaForCausalLM, STLlamaConfig

def test_inference(model_path, st_data_path, device="cuda"):
    """
    全量推理测试脚本：利用模型内部的 2D 转 4D 掩码逻辑，简化外部调用。
    """
    model_path = os.path.abspath(model_path)
    st_data_path = os.path.abspath(st_data_path)

    print(f"Loading model from {model_path}...")
    config = STLlamaConfig.from_pretrained(model_path, local_files_only=True)
    model = STLlamaForCausalLM.from_pretrained(model_path, config=config, local_files_only=True)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # 2. 加载 NPZ 数据
    print(f"Loading test data from {st_data_path}...")
    st_npz = np.load(st_data_path, allow_pickle=True)
    
    h_all = st_npz['h']
    mask_all = st_npz['mask']
    gt_all = st_npz['gt']
    
    num_samples = h_all.shape[0]
    print(f"Detected {num_samples} samples.")

    # 准备文本 Prompt
    prompt = "Task: Predict the future state at the query slots. QuerySlots:"
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    # 只需要传递 2D 的文本 attention_mask，模型内部会自动扩展处理 Query Slots 长度
    text_attention_mask = inputs.attention_mask.to(device)
    
    all_preds = []
    all_gts = []
    
    # 3. 循环推理
    print("Starting Inference...")
    with torch.no_grad():
        for i in range(num_samples):
            # 准备当前样本的时空输入
            h = torch.from_numpy(h_all[i]).float().unsqueeze(0).to(device)
            mask_obs = torch.from_numpy(mask_all[i]).float().unsqueeze(0).to(device)
            gt = torch.from_numpy(gt_all[i]).float().unsqueeze(0).to(device)

            # 调用模型：直接利用 STLlamaForCausalLM 的 forward 逻辑
            # 它会自动处理 internal_attention_mask 和预测头提取
            outputs = model(
                input_ids=input_ids,
                attention_mask=text_attention_mask,
                h=h,
                mask_obs=mask_obs,
                return_dict=True,
            )
            
            hidden_states = outputs.hidden_states[0]
            # 模型内部会将 final_query_mask 存放在 past_key_values 中返回
            final_query_mask = model.model._final_query_mask.bool()
            final_query_mask[:, -24] = False
            # 使用掩码提取预测值
            preds = model.predict_head(hidden_states[final_query_mask]) 
            # 对齐真值：对应后 60 步的观测点
            query_mask_2d = torch.zeros_like(mask_obs)
            query_mask_2d[:, 61:] = mask_obs[:, 61:]
            query_mask_2d = query_mask_2d.bool()
            gt_aligned = gt[query_mask_2d]

            all_preds.append(preds.cpu())
            all_gts.append(gt_aligned.cpu())
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{num_samples}...")

    # 4. 指标计算
    all_preds = torch.cat(all_preds, dim=0) # [11500, 2]
    all_gts = torch.cat(all_gts, dim=0) # [11500, 2]
    print('all_preds.shape:', all_preds.shape)
    print('all_gts.shape:', all_gts.shape)
    
    mse = torch.nn.functional.mse_loss(all_preds, all_gts).item()
    mae = torch.nn.functional.l1_loss(all_preds, all_gts).item()
    sq = (all_preds - all_gts) ** 2
    num = sq.numel()          # 除以的元素个数 = N*D
    # mse_my = (sq.sum() / num).item()
    print("\n" + "="*30)
    print(f"Overall Test Results:")
    print(f"Total Prediction Points: {all_preds.shape[0]}")
    print("num elements:", num)
    print(f"Overall MSE: {mse:.4f}")
    print(f"Overall MAE: {mae:.4f}")
    print("="*30)
    preds_np = all_preds.detach().cpu().numpy()
    gts_np   = all_gts.detach().cpu().numpy()
    np.savez("preds_gts.npz", preds=preds_np, gts=gts_np)
    print("saved to preds_gts.npz")
    
    # 5. 可视化检查 (展示最后一个样本的部分序列)
    plt.figure(figsize=(10, 5))
    draw_len = min(48, len(preds)) # 绘制前 48 个预测点
    
    # 假设数据是多维的，我们只绘制第一个维度 [:, 0]
    plt.plot(all_gts[:draw_len, 0], all_gts[:draw_len, 1], label='Ground Truth', color='blue', marker='o', markersize=4)
    plt.plot(all_preds[:draw_len, 0], all_preds[:draw_len, 1], label='Prediction', color='red', linestyle='--', marker='x', markersize=4)
    
    plt.title(f"Inference Visual Check (MSE: {mse:.4f})")
    plt.xlabel("Time Step")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "test_mse_charged_visual1.png"
    plt.savefig(save_path)
    print(f"Visual check plot saved to: {save_path}")

if __name__ == "__main__":
    # 根据你的环境修改路径
    MODEL_PATH = f"{project_root}/checkpoints/UrbanGPT_confirm/checkpoint-v1-50epoch"
    DATA_PATH = f"{project_root}/data/Sim/charged/test/test_data.npz"
    
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        test_inference(MODEL_PATH, DATA_PATH)
    else:
        print(f"Error: Path not found. \nModel: {MODEL_PATH}\nData: {DATA_PATH}")