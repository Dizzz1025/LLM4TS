#    Copyright 2023 Haotian Liu
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


from typing import List, Optional, Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, \
    LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
import json
import os.path as osp

from PerceiverResampler_bias import PerceiverResampler, TimeEmbedding, GatedCrossAttentionBlock
# from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
# IGNORE_INDEX = -100
# DEFAULT_STHIS_TOKEN = "<ST_HIS>"
# DEFAULT_STPRE_TOKEN = "<ST_PRE>"
# DEFAULT_ST_PATCH_TOKEN = "<ST_patch>"
# DEFAULT_ST_START_TOKEN = "<ST_start>"
# DEFAULT_ST_END_TOKEN = "<ST_end>"

# def MAE_torch(pred, true, mask_value=None):
#     if mask_value != None:
#         mask = torch.gt(true, mask_value)
#         pred = torch.masked_select(pred, mask)
#         true = torch.masked_select(true, mask)
#     mae_loss = torch.abs(true - pred)
#     return torch.mean(mae_loss)

# def scaler_mae_loss(scaler=None, mask_value=None):
#     def loss(preds, labels, mask=None):
#         if scaler is not None:
#             preds = scaler.inverse_transform(preds)
#             labels = scaler.inverse_transform(labels)
#         if mask is not None:
#             preds = preds * mask
#             labels = labels * mask
#         mae = MAE_torch(pred=preds, true=labels, mask_value=mask_value)
#         return mae
#     return loss

class STLlamaConfig(LlamaConfig):
    model_type = "STLlama"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Resampler Config
        self.resampler_num_latents = kwargs.get("resampler_num_latents", 5)
        self.text_resampler_num_latents = kwargs.get("text_resampler_num_latents", 5)
        self.resampler_depth = kwargs.get("resampler_depth", 2)
        self.resampler_heads = kwargs.get("resampler_heads", 8)
        self.resampler_dim_head = kwargs.get("resampler_dim_head", 64)
        self.st_input_dim = kwargs.get("st_input_dim", 50) # 默认输入维度，根据你的 ODE 输出修改
        self.gca_layer_interval = kwargs.get("gca_layer_interval", 16)
        self.target_dim = kwargs.get("target_dim", 2) # 目标预测变量的数量（例：inflow和outflow，2)
        # 默认不使用 immediate media attention 限制
        self.only_attend_immediate_media = kwargs.get("only_attend_immediate_media", False)

# class STPretrainConfig:
#     def __init__(self, dictionary):
#         for key, value in dictionary.items():
#             setattr(self, key, value)


# def load_model_pretrained(model_name, pretrain_model_path):
#     # load conig json
#     print("************************", pretrain_model_path)
#     assert osp.exists(osp.join(pretrain_model_path, 'config.json')), 'config.json missing'
#     with open(osp.join(pretrain_model_path, 'config.json'), 'r') as f:
#         config_dict = json.load(f)
#     args = STPretrainConfig(config_dict)
#     model = model_name(args)
#     pkl_files = glob.glob(osp.join(pretrain_model_path, '*.pkl'))
#     state_dict = torch.load(pkl_files[0])
#     # print(state_dict.keys())
#     if 'logit_scale' in state_dict.keys():
#         state_dict.pop('logit_scale')
#     print('loading ST pre train model')
#     model.load_state_dict(state_dict)

#     return model, args


class STLlamaModel(LlamaModel):
    config_class = STLlamaConfig

    def __init__(self, config: LlamaConfig):
        super(STLlamaModel, self).__init__(config)
        self.st_projector = nn.Linear(config.st_input_dim, config.hidden_size)
        
        # 1. Perceiver Resamplers
        self.feat_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.resampler_num_latents
        )
        self.text_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.text_resampler_num_latents
        )
        
        self.time_embedding = TimeEmbedding(config.hidden_size)
        self.time_embed_scale = nn.Parameter(torch.tensor(0.1))
        
        # --- REMOVED: Gated Cross Attention Layers ---
        # self.gca_layers is gone. We use standard Llama blocks now.
        
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            stat_input_ids: torch.LongTensor = None,
            stat_attention_mask: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            h: Optional[torch.Tensor] = None,
            mask_obs: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        target_dtype = self.st_projector.weight.dtype
        batch_size = input_ids.shape[0] if input_ids is not None else inputs_embeds.shape[0]

        # ------------------------------------------------------------------
        # 1. Extract Latents from Resamplers
        # ------------------------------------------------------------------
        # Path A: Feature Latents (Delta_feat)
        T_max = mask_obs.shape[1]
        idx = torch.arange(0, T_max, device=h.device).float()
        t_input = (idx / (T_max - 1)).unsqueeze(0)
        
        h_projected = self.st_projector(h.to(target_dtype))
        feat_latents, center_times = self.feat_resampler(h_projected, t_input)
        
        if center_times is not None:
            time_emb_out = self.time_embedding(center_times)
            feat_latents = feat_latents + self.time_embed_scale * time_emb_out

        # Path B: Text Latents (Delta_text)
        text_latents = None
        if stat_input_ids is not None:
            stat_embeds = self.embed_tokens(stat_input_ids).to(target_dtype)
            text_latents, _ = self.text_resampler(stat_embeds)

        # ------------------------------------------------------------------
        # 2. Construct Integrated Prompt: [Text_Latents] + [Feat_Latents] + [Prompt] + [Query Slots]
        # ------------------------------------------------------------------
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        combined_embeds_list = []
        final_query_mask_list = []

        for b in range(batch_size):
            # Sequence components
            parts = []
            if text_latents is not None:
                parts.append(text_latents[b]) # [K_text, D]
            parts.append(feat_latents[b])      # [K_feat, D]
            parts.append(inputs_embeds[b])     # [Seq_Prompt, D]
            
            # Query Slots for prediction
            pred_mask = torch.zeros_like(mask_obs[b])
            pred_mask[60:] = mask_obs[b, 60:]
            obs_indices = torch.nonzero(pred_mask).squeeze(-1).float()
            norm_time = (obs_indices / (T_max - 1)).to(target_dtype)
            
            time_embeds = self.time_embedding(norm_time)
            if time_embeds.dim() == 3: time_embeds = time_embeds.squeeze(0)
            parts.append(time_embeds)          # [Num_Obs, D]

            # Combine
            combined = torch.cat(parts, dim=0)
            combined_embeds_list.append(combined)

            # Mask: Only the very last part (time_embeds) are Query Slots for the Loss
            q_mask = torch.zeros(combined.shape[0], device=h.device, dtype=target_dtype)
            q_mask[-time_embeds.shape[0]:] = 1
            final_query_mask_list.append(q_mask)

        # Pad the sequences
        max_len = max([x.shape[0] for x in combined_embeds_list])
        device = h.device
        
        padded_embeds = torch.zeros(batch_size, max_len, self.config.hidden_size, device=device, dtype=target_dtype)
        padded_attn_mask = torch.zeros(batch_size, max_len, device=device, dtype=target_dtype)
        final_query_mask = torch.zeros(batch_size, max_len, device=device, dtype=target_dtype)
        
        for b in range(batch_size):
            length = combined_embeds_list[b].shape[0]
            padded_embeds[b, :length] = combined_embeds_list[b]
            padded_attn_mask[b, :length] = 1
            final_query_mask[b, :length] = final_query_mask_list[b]

        # Prepare 4D Causal Mask
        causal_mask = self._prepare_decoder_attention_mask(
            padded_attn_mask, (batch_size, max_len), padded_embeds, 0
        )

        # ------------------------------------------------------------------
        # 3. Standard Llama Forward (No GCA)
        # ------------------------------------------------------------------
        hidden_states = padded_embeds
        position_ids = torch.arange(0, max_len, device=device).unsqueeze(0)

        for decoder_layer in self.layers:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
            )
            hidden_states = layer_outputs[0]

        hidden_states = self.norm(hidden_states)
        self._final_query_mask = final_query_mask # Store for Head
        
        return BaseModelOutputWithPast(last_hidden_state=hidden_states, past_key_values=None)

    def get_monitor_stats(self):
        # 只记录还存在的全局参数：比如时间缩放系数
        stats = {
            "custom_stats/time_embed_scale": self.time_embed_scale.item()
        }
        
        # --- 删除了对 self.gca_layers 的循环遍历 ---
        # 原本这里的 gca_gate, gca_grad 等监控项已经没有意义了，因为层被删除了

        return stats

class STLlamaForCausalLM(LlamaForCausalLM):
    config_class = STLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = STLlamaModel(config)
        
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Prediction Head: Project hidden states of query slots to prediction
        # 假设预测维度为 1，如果 gt 维度不同请在此处修改
        self.predict_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.target_dim) 
        )

        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            stat_input_ids: torch.LongTensor = None,    # 确保添加这一行
            stat_attention_mask: Optional[torch.Tensor] = None, # 确保添加这一行
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            h: Optional[torch.Tensor] = None,          # [B, T_raw, D]
            mask_obs: Optional[torch.Tensor] = None,   # [B, T_raw]
            gt: Optional[torch.Tensor] = None,         # [B, T_raw] or [B, T_raw, D_out]
            return_dict: Optional[bool] = None,
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # 1. Forward Pass through Model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            stat_input_ids=stat_input_ids,
            stat_attention_mask=stat_attention_mask,
            inputs_embeds=inputs_embeds,
            h=h,
            mask_obs=mask_obs,
            return_dict=True,
            position_ids=position_ids,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        # query_mask = self.model._last_query_mask # [B, Seq_Len]
        final_query_mask = self.model._final_query_mask.bool() 
        # prediction_mask = self.model._prediction_mask_obs # 仅含后60步观测点的掩码        
        
        loss = None
        if gt is not None:
            # Flatten hidden states using mask
            # 只提取是 Query Slot 的那些 hidden state
            pred_hidden = hidden_states[final_query_mask] 
            
            # Predict
            preds = self.predict_head(pred_hidden) # [Total_Query_Slots]
            
            # Flatten GT using the SAME mask logic
            # mask_obs 和 query_mask 在逻辑上是对齐的（按顺序生成）
            # 直接使用 mask_obs 提取对应的 GT 值
            obs_mask_60 = torch.zeros_like(mask_obs)
            obs_mask_60[:, 60:] = mask_obs[:, 60:]
            obs_mask_60 = obs_mask_60.bool() # gt.shape = [1, 120, 2]
            true_values = gt[obs_mask_60].to(preds.device).to(preds.dtype)
            # Calculate Loss
            loss_fct = MSELoss()
            loss = loss_fct(preds, true_values)

        return CausalLMOutputWithPast(
            loss=loss,
            # logits=None, 
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,),
            # attentions=outputs.attentions,
        )

# 注册配置和模型
AutoConfig.register("STLlama", STLlamaConfig)
AutoModelForCausalLM.register(STLlamaConfig, STLlamaForCausalLM)
