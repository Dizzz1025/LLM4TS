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

from models.resampler.PerceiverResampler import PerceiverResampler, TimeEmbedding, GatedCrossAttentionBlock
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
        self.resampler_num_latents = kwargs.get("resampler_num_latents", 64)
        self.resampler_depth = kwargs.get("resampler_depth", 2)
        self.resampler_heads = kwargs.get("resampler_heads", 8)
        self.resampler_dim_head = kwargs.get("resampler_dim_head", 64)
        self.st_input_dim = kwargs.get("st_input_dim", 50) # 默认输入维度，根据你的 ODE 输出修改
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
        # 1. Perceiver Resampler: 将 h (ODE Hidden States) 压缩为 Latents
        self.resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.resampler_num_latents
        )
        
        # 2. Time Embedding: 将时间步编码为 Query Slot 的 Embedding
        self.time_embedding = TimeEmbedding(config.hidden_size)
        
        # 3. Gated Cross Attention Layers (插入到每一层 Llama Block 之后)
        # 这里的 GatedCrossAttentionBlock 来自 PerceiverResampler.py
        self.gca_layers = nn.ModuleList([
            GatedCrossAttentionBlock(
                dim=config.hidden_size,
                heads=config.resampler_heads,
                dim_head=config.resampler_dim_head
            ) for _ in range(config.num_hidden_layers)
        ])
        
        self.gradient_checkpointing = False

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            h: Optional[torch.Tensor] = None,          # [B, T_raw, D]
            mask_obs: Optional[torch.Tensor] = None,   # [B, T_raw]
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        ) # 是否返回每一层的hidden_state，而不是最后一层
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # ------------------------------------------------------------------
        # 1. Temporal Encoding: h -> Resampler -> Latents
        # ------------------------------------------------------------------
        if h is not None:
            # h: [Batch, Seq, Hidden]
            # Output: [Batch, Num_Latents, Hidden]
            h_projected = self.st_projector(h)
            latents = self.resampler(h_projected)
        else:
            raise ValueError("Input 'h' (ODE hidden variables) is required.")

        # ------------------------------------------------------------------
        # 2. Construct Prompt: [Text Prompt] + [Query Slots]
        # ------------------------------------------------------------------
        if inputs_embeds is None:
            # 首先对文本部分 (Task description / QuerySlots 占位符) 进行 Embedding
            inputs_embeds = self.embed_tokens(input_ids)
            
        # 根据 mask_obs 生成 Query Slots 的 Time Embedding
        # mask_obs: [B, T_raw], 1 代表该位置有观测值，需要生成对应的 Query Slot
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        
        combined_embeds_list = []
        query_slots_mask_list = [] # 用于标记哪些位置是 Query Slots (用于最后的预测头提取)
        prediction_mask_list = []
        
        for b in range(batch_size):
            full_mask = mask_obs[b]  # [T_raw]
            pred_mask = torch.zeros_like(full_mask)
            pred_mask[60:] = full_mask[60:]
            prediction_mask_list.append(pred_mask)            
            # 获取观测值的时间索引
            # indices: [Num_Obs]
            obs_indices = torch.nonzero(pred_mask).squeeze(-1).float()
            
            # 归一化时间步 (0~1)
            # 假设 mask_obs 的长度代表最大时间步 T_max
            T_max = mask_obs.shape[1]
            norm_time = obs_indices / (T_max - 1 + 1e-6)
            
            # 生成 Query Slot Embeddings
            # PerceiverResampler.py 中的 TimeEmbedding 会对输入进行 unsqueeze(-1) 扩展维度
            # 输入 norm_time: [Num_Obs]
            # 内部计算: [Num_Obs, 1] * [1, 1, D/2] -> [1, Num_Obs, D/2] -> cat -> [1, Num_Obs, D]
            time_embeds = self.time_embedding(norm_time)
            
            # 如果输出是 3D [1, Num_Obs, D]，我们需要去除 batch 维度以匹配 inputs_embeds
            if time_embeds.dim() == 3: 
                time_embeds = time_embeds.squeeze(0) # -> [Num_Obs, D]
            
            # 拼接: Text Embeddings + Query Slot Embeddings
            text_emb = inputs_embeds[b] # [Seq_Text, Hidden]
            
            combined = torch.cat([text_emb, time_embeds], dim=0)
            combined_embeds_list.append(combined)
            
            # 记录 Mask: 0 for text, 1 for query slots
            q_mask = torch.zeros(combined.shape[0], device=device)
            q_mask[len(text_emb):] = 1
            query_slots_mask_list.append(q_mask)

        # Pad to max length in batch
        max_len = max([x.shape[0] for x in combined_embeds_list])
        padded_embeds = torch.zeros(batch_size, max_len, self.config.hidden_size, device=device)
        padded_attention_mask = torch.zeros(batch_size, max_len, device=device)
        final_query_mask = torch.zeros(batch_size, max_len, device=device) # [B, Max_Len]

        for b in range(batch_size):
            length = combined_embeds_list[b].shape[0]
            padded_embeds[b, :length] = combined_embeds_list[b]
            padded_attention_mask[b, :length] = 1 
            final_query_mask[b, :length] = query_slots_mask_list[b]

        hidden_states = padded_embeds
        # attention_mask 仅用于处理 padding
        attention_mask = padded_attention_mask

        # ------------------------------------------------------------------
        # 3. Llama Layers + Gated Cross Attention
        # ------------------------------------------------------------------
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for i, (decoder_layer, gca_layer) in enumerate(zip(self.layers, self.gca_layers)):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # Standard Llama Layer Forward
            # 我们不传递 attention_mask (除了 padding)，让 Llama 内部处理 causal mask
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=None, 
                position_ids=None,
                past_key_value=past_key_values[i] if past_key_values else None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            # Injected Gated Cross Attention: LLM States <-> Time Latents
            # gca_layer 来自 PerceiverResampler.py，输入为 (x, media)
            hidden_states = gca_layer(hidden_states, latents)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 保存 query mask 供 Head 使用
        self._last_query_mask = final_query_mask
        self._prediction_mask_obs = torch.stack(prediction_mask_list)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)
            
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


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
            nn.Linear(config.hidden_size, 1) 
        )

        self.post_init()

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
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
            inputs_embeds=inputs_embeds,
            h=h,
            mask_obs=mask_obs,
            return_dict=True,
            **kwargs
        )

        hidden_states = outputs.last_hidden_state
        query_mask = self.model._last_query_mask # [B, Seq_Len]
        prediction_mask = self.model._prediction_mask_obs # 仅含后60步观测点的掩码        
        
        loss = None
        if gt is not None:
            # Flatten hidden states using mask
            # 只提取是 Query Slot 的那些 hidden state
            pred_hidden = hidden_states[query_mask.bool()] 
            
            # Predict
            preds = self.predict_head(pred_hidden).squeeze(-1) # [Total_Query_Slots]
            
            # Flatten GT using the SAME mask logic
            # mask_obs 和 query_mask 在逻辑上是对齐的（按顺序生成）
            # 直接使用 mask_obs 提取对应的 GT 值
            true_values = gt[prediction_mask.bool()] # [Total_Query_Slots]
            
            # Calculate Loss
            loss_fct = MSELoss()
            loss = loss_fct(preds, true_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None, 
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

# 注册配置和模型
AutoConfig.register("STLlama", STLlamaConfig)
AutoModelForCausalLM.register(STLlamaConfig, STLlamaForCausalLM)
