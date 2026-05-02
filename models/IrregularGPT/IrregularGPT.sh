# 1. Base Path Configuration
model_path=./checkpoints/vicuna-7b-v1.5-16k  # Ensure this directory contains config.json and weights
instruct_ds=/data/NYC/train/NYC_train_prompts.json       # Corresponds to data_path
st_data_path=./data/NYC/train/h_mask_gt_merged.npz       # Corresponds to the file containing h, mask, gt
text_stats_ds=./data/NYC/train/NYCmerged_text_stats.json        # Corresponds to text_stats_path (new)
instruct_eval_ds=./data/NYC/taxi/NYCtaxi_eval_formatted.json       # Corresponds to data_path
st_data_eval_path=./data/NYC/taxi/h_mask_gt_taxi_eval.npz       # Corresponds to the file containing h, mask, gt
text_stats_eval_ds=./data/NYC/taxi/NYCtaxi_eval_text_stats.json        # Corresponds to text_stats_path (new)
output_model=./checkpoints/NYC # Model save path

# Run wandb in offline mode
export WANDB_MODE=offline

# python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --master_port=20001 \
#     urbangpt/train/train_mem.py \

python IrregularGPT/train/train_st.py \
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
    --resampler_num_latents 5 \
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 300 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 5 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing False \
    --report_to wandb \
