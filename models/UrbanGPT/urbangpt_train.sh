# 1. 基础路径配置
model_path=./checkpoints/vicuna-7b-v1.5-16k  # 确保此目录下有 config.json 和权重
instruct_ds=./data/Sim/train/Sim_prompts.json   # 你刚才生成的 JSON 文件
st_data_path=./data/Sim/train/train_data.npz # 包含 h, mask, gt 的 PT 文件
output_model=./checkpoints/UrbanGPT_v2

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
    --freeze_backbone True \
    --tune_st_mlp_adapter True \
    --resampler_num_latents 5 \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "constant" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --report_to wandb \

