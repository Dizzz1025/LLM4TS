# Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
# This file is adopted from the Stanford Alpaca repository and has been modified
# for the LLMODE project, including adaptations for spatio-temporal data handling
# influenced by the UrbanGPT project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script is the main entry point for training the STLlama model, which is the
LLM component of the LLMODE framework. It handles data loading, model configuration,
and the training process using the Hugging Face Transformers library.

The script is responsible for:
1.  Parsing model, data, and training arguments.
2.  Loading and pre-processing textual prompts and their corresponding spatio-temporal
    data (ODE latent states, observation masks, and ground truth values).
3.  Configuring the STLlama model with a frozen LLM backbone and trainable components
    for spatio-temporal adaptation.
4.  Setting up a custom optimizer and callbacks for monitoring.
5.  Executing the training loop and saving the final model.
"""

import os
import copy
import json
import logging
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer, TrainerCallback
import numpy as np
import random
import wandb

# Local imports for the custom model architecture.
from llmode.model import *

IGNORE_INDEX = -100

def setup_seed(seed):
    """Sets the random seed for reproducibility across all relevant libraries."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

# =============================================================================
# 1. Argument Classes
# =============================================================================

@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from."""
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v1")
    freeze_backbone: bool = field(default=False, metadata={"help": "Whether to freeze the pretrained LLM backbone."})
    resampler_num_latents: int = field(default=5, metadata={"help": "Number of latent tokens for the Perceiver Resampler (K in the paper)."})

@dataclass
class DataArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    data_path: str = field(default=None, metadata={"help": "Path to the training data (JSON file with text prompts)."})
    st_data_path: str = field(default=None, metadata={"help": "Path to the spatio-temporal features (e.g., a .npz file with h, mask, gt)."})
    text_stats_path: str = field(default=None, metadata={"help": "Path to the auxiliary statistical text descriptors."})
    data_eval_path: str = field(default=None, metadata={"help": "Path to the evaluation data (JSON file)."})
    st_data_eval_path: str = field(default=None, metadata={"help": "Path to the evaluation spatio-temporal features."})
    text_stats_eval_path: str = field(default=None, metadata={"help": "Path to the evaluation statistical text descriptors."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """Arguments for the training process."""
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})

# =============================================================================
# 2. Data Handling: Dataset and Collator
# =============================================================================

class LazySupervisedDataset(Dataset):
    """
    A dataset that lazily loads examples for supervised fine-tuning. It combines
    textual prompts with corresponding spatio-temporal (ST) data.
    """
    def __init__(self, data_path: str, st_data_path: str, text_stats_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(LazySupervisedDataset, self).__init__()
        logging.warning("Loading text prompt data...")
        self.list_data_dict = json.load(open(data_path, "r"))
        self.tokenizer = tokenizer

        # Load auxiliary statistical priors (the text descriptors). This corresponds
        # to the "Auxiliary Descriptor Tokenization" branch in Sec. 3.3.
        if text_stats_path and os.path.exists(text_stats_path):
            self.text_prior_dict = json.load(open(text_stats_path, "r"))
        else:
            logging.warning(f"Statistical text priors not found at {text_stats_path}. Proceeding without them.")
            self.text_prior_dict = {}

        # Load spatio-temporal features from a NumPy archive.
        # 'h': Latent states from the ODE encoder (Sec. 3.2).
        # 'mask': Observation mask.
        # 'gt': Ground truth values for loss calculation.
        logging.warning("Loading spatio-temporal data...")
        st_npz = np.load(st_data_path, allow_pickle=True)
        self.h_dict = st_npz['h']
        self.mask_dict = st_npz['mask']
        self.gt_dict = st_npz['gt']

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        """
        Retrieves a single training example, consisting of tokenized text and ST features.
        """
        sources = self.list_data_dict[i]
        sample_id = int(sources.get('id', i))
        sample_key = str(sample_id)
        
        # --- Tokenize Textual Inputs ---
        # Main prompt containing task description and query slot placeholders.
        instruction = sources['conversations'][0]['value']
        input_ids = self.tokenizer(
            instruction, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True
        ).input_ids[0]

        # Auxiliary statistical text prior.
        stat_text = self.text_prior_dict.get(sample_key, "")
        stat_input_ids = self.tokenizer(
            stat_text, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True
        ).input_ids[0]
        
        # --- Load Spatio-Temporal Tensors ---
        h = torch.from_numpy(self.h_dict[sample_id]).float()
        mask = torch.from_numpy(self.mask_dict[sample_id]).float()
        gt = torch.from_numpy(self.gt_dict[sample_id]).float()

        return dict(
            input_ids=input_ids,
            stat_input_ids=stat_input_ids,
            h=h,
            mask_obs=mask,
            gt=gt
        )

@dataclass
class DataCollatorForSTLlama:
    """
    Collates examples for ST-Llama, padding all elements (text IDs, ST features)
    to the maximum length in a batch.
    """
    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract sequences from instances.
        input_ids, stat_input_ids, h, mask_obs, gt = (
            [instance[key] for instance in instances] for key in ('input_ids', 'stat_input_ids', 'h', 'mask_obs', 'gt')
        )

        # Pad all sequences to the max length in the batch.
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        stat_input_ids = torch.nn.utils.rnn.pad_sequence(stat_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        h = torch.nn.utils.rnn.pad_sequence(h, batch_first=True, padding_value=0.0)
        mask_obs = torch.nn.utils.rnn.pad_sequence(mask_obs, batch_first=True, padding_value=0)
        gt = torch.nn.utils.rnn.pad_sequence(gt, batch_first=True, padding_value=0.0)

        # Create attention masks for the padded text inputs.
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        stat_attention_mask = stat_input_ids.ne(self.tokenizer.pad_token_id)
        
        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stat_input_ids=stat_input_ids,
            stat_attention_mask=stat_attention_mask,
            h=h,
            mask_obs=mask_obs,
            gt=gt,
            labels=gt, # `labels` is the conventional key for loss computation in Trainer.
        )

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args: DataArguments) -> Dict:
    """Creates the training and evaluation datasets and the data collator."""
    train_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_path,
        st_data_path=data_args.st_data_path,
        text_stats_path=data_args.text_stats_path
    )
    eval_dataset = LazySupervisedDataset(
        tokenizer=tokenizer,
        data_path=data_args.data_eval_path,
        st_data_path=data_args.st_data_eval_path,
        text_stats_path=data_args.text_stats_eval_path
    )
    data_collator = DataCollatorForSTLlama(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=data_collator)

# =============================================================================
# 3. Custom Trainer and Callbacks
# =============================================================================

class CustomParamMonitorCallback(TrainerCallback):
    """
    A custom callback to monitor the learnable gate parameters and other custom
    metrics during training, logging them via WandB. This allows for analyzing
    the behavior of the injection mechanism, as discussed in Sec. 4.5.
    """
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if not state.is_world_process_zero or model is None or logs is None:
            return
            
        real_model = model.module if hasattr(model, "module") else model
        if hasattr(real_model, "model") and hasattr(real_model.model, "get_monitor_stats"):
            custom_metrics = real_model.model.get_monitor_stats()
            wandb.log(custom_metrics, step=state.global_step)

class STTrainer(Trainer):
    """
    Custom Trainer that sets up an optimizer with different learning rates for
    the gate parameters versus the rest of the model.
    """
    def create_optimizer(self):
        if self.optimizer is None:
            gate_lr = self.args.learning_rate
            base_lr = self.args.learning_rate
            weight_decay = self.args.weight_decay

            # Identify parameters to be optimized with a special learning rate (and no weight decay).
            gate_keywords = ("attn_gate", "ff_gate", "fusion_lambda", "time_embed_scale")
            
            gate_params, base_params = [], []
            for n, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                if any(k in n for k in gate_keywords):
                    gate_params.append(p)
                else:
                    base_params.append(p)

            optimizer_grouped_parameters = [
                {"params": base_params, "weight_decay": weight_decay, "lr": base_lr},
                {"params": gate_params, "weight_decay": 0.0, "lr": gate_lr},
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer

# =============================================================================
# 4. Main Training Function
# =============================================================================
    
def train():
    """Main function to run the training process."""
    # --- 1. Initialization and Argument Parsing ---
    setup_seed(42)
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- 2. Model and Tokenizer Setup ---
    # Load model configuration and update with custom LLMODE parameters.
    config = STLlamaConfig.from_pretrained(model_args.model_name_or_path)
    config.resampler_num_latents = model_args.resampler_num_latents
    
    # Load the pretrained LLM and adapt it to the STLlamaForCausalLM architecture.
    model = STLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        torch_dtype=(torch.bfloat16 if training_args.bf16 else torch.float32)
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    # --- 3. Model Freezing Strategy ---
    # Freeze the LLM backbone and only train the newly added adapter layers.
    # This aligns with the "frozen LLM setting" described in the paper's abstract.
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)
    
    # Explicitly set the trainable components.
    model.model.st_projector.requires_grad_(True)
    model.model.feat_resampler.requires_grad_(True)
    model.model.text_resampler.requires_grad_(True)
    model.model.time_embedding.requires_grad_(True)
    model.model.gca_layers.requires_grad_(True)
    model.predict_head.requires_grad_(True)
    model.model.time_embed_scale.requires_grad_(True)
    
    # --- 4. Data and Trainer Setup ---
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    
    trainer = STTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
        callbacks=[CustomParamMonitorCallback()]
    )

    # --- 5. Start Training ---
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
        
    trainer.save_state()
    
    # --- 6. Save Final Model ---
    if trainer.is_world_process_zero():
        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)

if __name__ == "__main__":
    train()
