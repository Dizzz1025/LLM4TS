# 1. 基础路径配置
model_path=/home/zhangdi24/LLM4TS/models/LLMODE/checkpoints/vicuna-7b-v1.5-16k  # 确保此目录下有 config.json 和权重
# instruct_ds=./data/Sim/train/Sim_train_prompts.json      # 对应 data_path
# st_data_path=./data/Sim/train/h_mask_gt_merged.npz      # 对应 包含h,mask,gt
# text_stats_ds=./data/Sim/train/Simsystem_text_stats.json       # 对应 text_stats_path (新增)
# output_model=./checkpoints/Sim_v3
instruct_ds=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/train/Sim_train_prompts.json      # 对应 data_path
st_data_path=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/train/h_mask_gt_merged_100.npz      # 对应 包含h,mask,gt
text_stats_ds=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/train/Simsystem_text_stats.json       # 对应 text_stats_path (新增)

instruct_eval_ds=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/springs/Springs_test_prompts.json      # 对应 data_path
st_data_eval_path=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/springs/h_mask_gt_springs.npz      # 对应 包含h,mask,gt
text_stats_eval_ds=/home/zhangdi24/LLM4TS/models/LLMODE/data/Sim/springs/Simsystem_text_stats.json       # 对应 text_stats_path (新增)

output_model=./checkpoints/Sim_k8


# 离线运行 wandb
export WANDB_MODE=offline

# 2. 启动分布式训练
# 注意：如果 GPU 显存不足（如 3090/4090），建议将 nproc_per_node 设小或调低 batch_size
# python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 \
#     urbangpt/train/train_mem.py \
python urbangpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --st_data_path ${st_data_path} \
    --text_stats_path ${text_stats_ds} \
    --data_eval_path ${instruct_eval_ds} \
    --st_data_eval_path ${st_data_eval_path} \
    --text_stats_eval_path ${text_stats_eval_ds} \
    --freeze_backbone True \
    --tune_st_mlp_adapter True \
    --resampler_num_latents 8 \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 300 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to wandb \

