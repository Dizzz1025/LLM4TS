# Adopted from https://github.com/lm-sys/FastChat. 
# Modified for ST-Llama (ODE hidden states + Mask + Ground Truth)
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback, BitsAndBytesConfig

from urbangpt.model.STQwen2_new import STQwen2ForCausalLM, STQwen2Config

import numpy as np

import random
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
IGNORE_INDEX = -100

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False)
    tune_st_mlp_adapter: bool = field(default=False)
    resampler_num_latents: int = field(default=5)

    # quantization
    load_in_8bit: bool = field(default=False)
    load_in_4bit: bool = field(default=False)
    bnb_4bit_quant_type: str = field(default="nf4")

    # LoRA for quantized training
    use_lora: bool = field(default=False)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.05)

@dataclass
class DataArguments:
    data_path: str = field(default="./generated_file/Sim_prompts.json", metadata={"help": "Path to the training data (json)."})
    st_data_path: str = field(default="./Sim/train_data.npz", metadata={"help": "Path to the ODE hidden states (h) and masks."})
    text_stats_path: str = field(default=None, metadata={"help": "Path to the generated statistical text JSON (e.g., Sim_text_stats.json)."})
    data_eval_path: str = field(default="./generated_file/Sim_prompts.json", metadata={"help": "Path to the training data (json)."})
    st_data_eval_path: str = field(default="./Sim/train_data.npz", metadata={"help": "Path to the ODE hidden states (h) and masks."})
    text_stats_eval_path: str = field(default=None, metadata={"help": "Path to the generated statistical text JSON (e.g., Sim_text_stats.json)."})
    
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning with temporal features."""

    def __init__(self, data_path: str, st_data_path: str, text_stats_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = json.load(open(data_path, "r"))

        logging.warning("Formatting inputs...")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        
        # Load independent statistical text priors (the JSON we just generated)
        # Expected format: {"0": "[SEG_STATS ...] ...", "1": "..."}
        if os.path.exists(text_stats_path):
            self.text_prior_dict = json.load(open(text_stats_path, "r"))
        else:
            logging.error(f"Text stats file not found at {text_stats_path}")
            self.text_prior_dict = {}
            
        # 假设 st_data_path 是一个保存了 'h', 'mask', 'gt' 的 .pt 或 .pkl 文件
        # 也可以根据你的实际存储格式（如每个样本一个文件）在这里修改加载逻辑
        st_npz = np.load(st_data_path, allow_pickle=True)
        self.h_dict = st_npz['h']
        self.mask_dict = st_npz['mask']
        self.gt_dict = st_npz['gt'] 

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        sample_id = int(sources.get('id', i))
        sample_key = str(sample_id)
        
        # 1. 获取文本 Prompt 并进行 Tokenize
        # 你的 Prompt 结构示例: "Task: Predict future state... QuerySlots: <q1> <q2>..."
        instruction = sources['conversations'][0]['value']
        
        input_ids = self.tokenizer(
            instruction,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
        ).input_ids[0]

        # 2. Path Delta_text: Statistical Prior (Secondary Source)
        # Get the [SEG_STATS] string we generated for this ID
        stat_text = self.text_prior_dict.get(sample_key, "[SEG_STATS] no_prior_available [/SEG_STATS]")
        
        # Tokenize the statistical text independently
        stat_input_ids = self.tokenizer(
            stat_text,
            return_tensors="pt",
            padding="longest",
            max_length=self.tokenizer.model_max_length, # Statistical text can be long, adjust as needed
            truncation=True,
        ).input_ids[0]
        
        # 2. 获取对应的时间序列特征
        # h: [T, D], mask: [T], gt: [T]
        h_val = self.h_dict[sample_id]
        mask_val = self.mask_dict[sample_id]
        gt_val = self.gt_dict[sample_id]
        
        # 转换为 torch.Tensor
        h = torch.from_numpy(h_val).float()
        mask = torch.from_numpy(mask_val).float()
        gt = torch.from_numpy(gt_val).float()

        return dict(
            input_ids=input_ids,
            stat_input_ids=stat_input_ids,
            h=h,
            mask_obs=mask,
            gt=gt
        )

@dataclass
class DataCollatorForSTQwen2(object):
    """Collate examples for ST-Llama."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [instance['input_ids'] for instance in instances]
        stat_input_ids = [instance['stat_input_ids'] for instance in instances] # 新增
        h = [instance['h'] for instance in instances]
        mask_obs = [instance['mask_obs'] for instance in instances]
        gt = [instance['gt'] for instance in instances]

        # 对文本 ID 进行 Padding
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 3. 对统计先验 stat_input_ids 进行 Padding [B, L_stat]
        # 动态补齐到当前 Batch 的最大长度，节省显存
        stat_input_ids = torch.nn.utils.rnn.pad_sequence(
            stat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        # 显式生成 position_ids [batch, seq_len]
        # batch_size, seq_len = input_ids.shape
        # position_ids = torch.arange(0, seq_len, dtype=torch.long).unsqueeze(0).expand(batch_size, seq_len).clone()
        
        # 对时间序列特征 h 进行 Padding [B, T_max, D]
        h = torch.nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=0.0).contiguous()
        
        # 对 mask 和 gt 进行 Padding [B, T_max]
        mask_obs = torch.nn.utils.rnn.pad_sequence(mask_obs, batch_first=True, padding_value=0).contiguous()
        gt = torch.nn.utils.rnn.pad_sequence(gt, batch_first=True, padding_value=0.0).contiguous()

        batch = dict(
            input_ids=input_ids,
            # position_ids=position_ids,
            stat_input_ids=stat_input_ids,
            h=h,
            mask_obs=mask_obs,
            gt=gt,
            labels=gt,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            stat_attention_mask=stat_input_ids.ne(self.tokenizer.pad_token_id),
        )

        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Train/Test split can be added here."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        st_data_path=data_args.st_data_path,
        text_stats_path=data_args.text_stats_path
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_eval_path, # 举例：替换路径名
        st_data_path=data_args.st_data_eval_path,
        text_stats_path=data_args.text_stats_eval_path
    )
    
    # max_eval_samples = 40
    # if len(eval_dataset.list_data_dict) > max_eval_samples:
    #     logging.warning(f"Slicing eval_dataset from {len(eval_dataset.list_data_dict)} to {max_eval_samples} samples.")
    #     eval_dataset.list_data_dict = eval_dataset.list_data_dict[:max_eval_samples]
    max_train_samples = 64   # 冒烟测试先设小一点，比如 16 / 32 / 64
    max_eval_samples = 40

    if len(train_dataset.list_data_dict) > max_train_samples:
        logging.warning(
            f"Slicing train_dataset from {len(train_dataset.list_data_dict)} to {max_train_samples} samples."
        )
        train_dataset.list_data_dict = train_dataset.list_data_dict[:max_train_samples]

    if len(eval_dataset.list_data_dict) > max_eval_samples:
        logging.warning(
            f"Slicing eval_dataset from {len(eval_dataset.list_data_dict)} to {max_eval_samples} samples."
        )
        eval_dataset.list_data_dict = eval_dataset.list_data_dict[:max_eval_samples]
        
    data_collator = DataCollatorForSTQwen2(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

def _get_monitor_model(real_model):
    if hasattr(real_model, "model") and hasattr(real_model.model, "get_monitor_stats"):
        return real_model.model

    if hasattr(real_model, "base_model") and hasattr(real_model.base_model, "model"):
        base = real_model.base_model.model
        if hasattr(base, "model") and hasattr(base.model, "get_monitor_stats"):
            return base.model

    return None


class CustomParamMonitorCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        """
        每当达到 logging_steps 时，该函数会被 Trainer 调用
        """
        # 1. 只有主进程记录日志
        if not state.is_world_process_zero:
            return
            
        # 2. 检查 model 和 logs 是否存在
        # 在正常的 on_log 调用中，logs 是必须存在的字典
        if model is not None and logs is not None:
            # 处理 DDP 下的模型包裹
            real_model = model.module if hasattr(model, "module") else model
            # 获取你的 STLlamaModel
            if hasattr(real_model, "model"):
                st_model = real_model.model
                
                # 获取自定义指标：{"custom_stats/time_embed_scale": 0.1, ...}
                custom_metrics = st_model.get_monitor_stats()
                
                # 【核心修改点】：直接将指标更新到 logs 字典中
                # Trainer 随后会将这个更新后的 logs 字典发送给 WandB 等
                # logs.update(custom_metrics)
                wandb.log(custom_metrics, step=state.global_step)

class STTrainer(Trainer):
    def create_optimizer(self):
        """
        Setup the optimizer with special learning rates for gate parameters.
        """
        if self.optimizer is None:
            # 设定你的学习率倍数
            gate_lr = self.args.learning_rate  # 假设门控参数学习率放大10倍
            base_lr = self.args.learning_rate
            weight_decay = self.args.weight_decay

            # 定义需要特殊对待的关键字
            gate_keywords = ("attn_gate", "ff_gate", "fusion_lambda", "time_embed_scale")
            
            gate_params = []
            base_params_with_decay = []
            base_params_no_decay = []

            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if "original_module" in n:
                    continue
                
                # 检查是否属于门控参数
                if any(k in n for k in gate_keywords):
                    gate_params.append(p)
                    print(f"Optimizing {n} with gate_lr (no decay)")
                # # 优化：通常 bias 和 LayerNorm 也不应该有 weight_decay
                elif any(nd in n for nd in ["bias", "LayerNorm.weight"]):
                    base_params_no_decay.append(p)
                else:
                    base_params_with_decay.append(p)

            # 构造参数组
            optimizer_grouped_parameters = [
                {
                    "params": base_params_with_decay,
                    "weight_decay": weight_decay,
                    "lr": base_lr,
                },
                {
                    "params": base_params_no_decay,
                    "weight_decay": 0.0,
                    "lr": base_lr,
                },
                {
                    "params": gate_params,
                    "weight_decay": 0.0,  # 按照你的要求，关掉 weight_decay
                    "lr": gate_lr,        # 按照你的要求，使用更大的学习率
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
    
def train():
    setup_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 在 train() 函数中，解析完 training_args 后进行如下设置：
    training_args.evaluation_strategy = "epoch"  # 每轮训练结束后进行评估
    training_args.save_strategy = "epoch"        # 只有评估完才考虑保存，两者需一致
    training_args.logging_strategy = "epoch"     # 让日志也按 epoch 输出，方便对齐指标

    # 保存最优模型的核心参数
    training_args.load_best_model_at_end = True  # 训练结束自动加载指标最好的权重
    training_args.metric_for_best_model = "eval_loss" # 这里填 eval_loss 里的 "loss"
    training_args.greater_is_better = False      # Loss 越小越好
    training_args.save_total_limit = 2           # 最多保留 2 个 checkpoint，节省磁盘
    if model_args.load_in_8bit or model_args.load_in_4bit:
        if training_args.optim == "adamw_torch":
            training_args.optim = "paged_adamw_8bit"
        # PEFT 包装后，保留自定义输入字段，避免被 Trainer 丢弃
        training_args.remove_unused_columns = False
    local_rank = getattr(training_args, "local_rank", -1)
    # device_map = None
    # if local_rank != -1:
    #     device_map = {"": local_rank}

    # 初始化配置与模型
    config = STQwen2Config.from_pretrained(model_args.model_name_or_path)
    config.resampler_num_latents = model_args.resampler_num_latents
    
    quantization_config = None
    compute_dtype = torch.bfloat16 if training_args.bf16 else torch.float16

    if model_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
            llm_int8_skip_modules=[
                "st_projector",
                "feat_resampler",
                "text_resampler",
                "time_embedding",
                "gca_layers",
                "predict_head",
                "lm_head",
            ],
        )
    elif model_args.load_in_8bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=[
                "st_projector",
                "feat_resampler",
                "text_resampler",
                "time_embedding",
                "gca_layers",
                "predict_head",
                "lm_head",
            ],
        )

    model = STQwen2ForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        # device_map=device_map,
        quantization_config=quantization_config,
        torch_dtype=compute_dtype
    )
    
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None

    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_args.load_in_8bit or model_args.load_in_4bit:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=training_args.gradient_checkpointing,
        )

        # 给 Trainer 一个最小 PEFT 外壳，避免“purely quantized model”报错。
        # 这里 LoRA 只作为壳，主干仍然不训练；真正训练的是你插入的自定义模块。
        peft_config = LoraConfig(
            r=1,
            lora_alpha=1,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj"],
        )
        model = get_peft_model(model, peft_config)

        # 先全部冻结
        for _, p in model.named_parameters():
            p.requires_grad = False

        # 手动打开你自己插入的模块（全参训练）
        base_model = model.base_model.model if hasattr(model, "base_model") else model

        base_model.model.st_projector.requires_grad_(True)
        base_model.model.feat_resampler.requires_grad_(True)
        base_model.model.text_resampler.requires_grad_(True)
        base_model.model.time_embedding.requires_grad_(True)
        base_model.model.gca_layers.requires_grad_(True)
        base_model.predict_head.requires_grad_(True)
        base_model.model.time_embed_scale.requires_grad_(True)

        # 保留一个极小 LoRA 壳参与训练，确保量化 + Trainer 路径稳定
        for n, p in model.named_parameters():
            if "lora_" in n and p.dtype.is_floating_point:
                p.requires_grad = True

    # 冻结策略
    is_kbit = model_args.load_in_8bit or model_args.load_in_4bit
    if model_args.freeze_backbone and not is_kbit:
        model.model.requires_grad_(False)
    
    # 确保 Resampler, TimeEmbedding, GCA, PredictHead 是可训练的
    if not (model_args.load_in_8bit or model_args.load_in_4bit):
        model.model.st_projector.requires_grad_(True)
        model.model.feat_resampler.requires_grad_(True)
        model.model.text_resampler.requires_grad_(True)
        model.model.time_embedding.requires_grad_(True)
        model.model.gca_layers.requires_grad_(True)
        model.predict_head.requires_grad_(True)
        model.model.time_embed_scale.requires_grad_(True)
    
    trainable = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    if data_module['eval_dataset'] is None:
        logging.error("Evaluation dataset is missing! Evaluation strategy 'epoch' will fail.")
    # monitor_cb = CustomParamMonitorCallback()
    trainer = STTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        # callbacks=[monitor_cb]
    )

    # 7. 训练过程
    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    trainer.train()
        
    trainer.save_state()
    
    # 8. 安全保存：只在主进程执行
    if trainer.is_world_process_zero():
        # 保存 PEFT 外壳 / adapter
        trainer.model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

        # 额外保存你自己插入模块的权重
        base_model = trainer.model.base_model.model if hasattr(trainer.model, "base_model") else trainer.model
        torch.save(
            {
                "st_projector": base_model.model.st_projector.state_dict(),
                "feat_resampler": base_model.model.feat_resampler.state_dict(),
                "text_resampler": base_model.model.text_resampler.state_dict(),
                "time_embedding": base_model.model.time_embedding.state_dict(),
                "gca_layers": base_model.model.gca_layers.state_dict(),
                "predict_head": base_model.predict_head.state_dict(),
                "time_embed_scale": base_model.model.time_embed_scale.detach().cpu(),
            },
            os.path.join(training_args.output_dir, "custom_modules.pt"),
        )

if __name__ == "__main__":
    train()