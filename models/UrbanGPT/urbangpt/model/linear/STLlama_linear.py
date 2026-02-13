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

from PerceiverResampler_linear import PerceiverResampler, TimeEmbedding, GatedCrossAttentionBlock, Linear_PerceiverResampler
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
        # self.st_norm = nn.LayerNorm(config.st_input_dim)
        # 1. Perceiver Resampler: 将 h (ODE Hidden States) 压缩为 Latents
        self.feat_resampler = Linear_PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.resampler_num_latents,
            seq_len=216
        )
        # --- 新增：2. Text Resampler (Delta_text) ---
        
        # 用于将详细的统计文本描述压缩为固定的 K 个 Token
        self.text_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.text_resampler_num_latents
        )
        # 2. Time Embedding: 将时间步编码为 Query Slot 的 Embedding
        self.time_embedding = TimeEmbedding(config.hidden_size)
        
        # 3. Gated Cross Attention Layers (插入到每一层 Llama Block 之后)
        # 这里的 GatedCrossAttentionBlock 来自 PerceiverResampler.py
        self.gca_interval = config.gca_layer_interval
        num_gca_layers = config.num_hidden_layers // self.gca_interval
        self.gca_layers = nn.ModuleList([
            GatedCrossAttentionBlock(
                dim=config.hidden_size,
                heads=config.resampler_heads,
                dim_head=config.resampler_dim_head
            ) for _ in range(num_gca_layers)
        ])
        
        self.gradient_checkpointing = False
        
        # 添加时间嵌入强度系数，初始化为较小值（如 0.1 或 0.0）
        self.time_embed_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
            self,
            input_ids: torch.LongTensor = None,
            stat_input_ids: torch.LongTensor = None,    # --- 新增输入 ---
            stat_attention_mask: Optional[torch.Tensor] = None, # --- 新增输入 ---
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
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
        target_dtype = self.st_projector.weight.dtype
        # ------------------------------------------------------------------
        # 1. Temporal Encoding: h -> Resampler -> Latents
        # ------------------------------------------------------------------
        # Path A: Feature Prior (Delta_feat)
        T_max = mask_obs.shape[1] # 120
        idx = torch.arange(0, T_max, device=h.device).float()
        t_query = idx / (T_max - 1)   # 归一化
        t_input = t_query.unsqueeze(0)
        if h is not None:
            # h: [Batch, Seq, Hidden]
            # Output: [Batch, Num_Latents, Hidden]
            h = h.to(target_dtype)
            # h = self.st_norm(h.float()).to(target_dtype)
            h_projected = self.st_projector(h)
            feat_kv, center_times = self.feat_resampler(h_projected, t_input)
        else:
            raise ValueError("Input 'h' (ODE hidden variables) is required.")
        
        if center_times is not None:
            # 使用 model 内置的 time_embedding 模块进行编码
            # time_emb_out: [B, K, D]
            time_emb_out = self.time_embedding(center_times)
            
            # 按照公式: feat_kv = feat_kv + gamma * time_embedding(center_times)
            # self.time_embed_scale 即为你要求的可学习系数
            feat_kv = feat_kv + self.time_embed_scale * time_emb_out
            
        # Path B: Text Prior (Delta_text) - 新增逻辑
        if stat_input_ids is not None:
            # 使用 LLM 自带的 Embedding 层将统计文本 ID 转换为向量
            stat_embeds = self.embed_tokens(stat_input_ids).to(target_dtype)
            # 通过第二个 Resampler 压缩
            text_kv, _ = self.text_resampler(stat_embeds) # [B, Text_Latents, D]
            
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
            pred_mask[144:] = full_mask[144:]
            prediction_mask_list.append(pred_mask)            
            # 获取观测值的时间索引
            # indices: [Num_Obs]
            obs_indices = torch.nonzero(pred_mask).squeeze(-1).float()
            
            # 归一化时间步 (0~1)
            # 假设 mask_obs 的长度代表最大时间步 T_max
            # T_max = mask_obs.shape[1]
            norm_time = obs_indices / (T_max - 1 + 1e-6)
            
            # 生成 Query Slot Embeddings
            # PerceiverResampler.py 中的 TimeEmbedding 会对输入进行 unsqueeze(-1) 扩展维度
            # 输入 norm_time: [Num_Obs]
            # 内部计算: [Num_Obs, 1] * [1, 1, D/2] -> [1, Num_Obs, D/2] -> cat -> [1, Num_Obs, D]
            norm_time = norm_time.to(target_dtype)
            time_embeds = self.time_embedding(norm_time)
            time_embeds = time_embeds.to(target_dtype)
            # 如果输出是 3D [1, Num_Obs, D]，我们需要去除 batch 维度以匹配 inputs_embeds
            if time_embeds.dim() == 3: 
                time_embeds = time_embeds.squeeze(0) # -> [Num_Obs, D]
            
            # 拼接: Text Embeddings + Query Slot Embeddings
            text_emb = inputs_embeds[b] # [Seq_Text, Hidden]
            
            combined = torch.cat([text_emb, time_embeds], dim=0)
            combined_embeds_list.append(combined)
            
            # 记录 Mask: 0 for text, 1 for query slots
            q_mask = torch.zeros(combined.shape[0], device=device, dtype=target_dtype)
            q_mask[len(text_emb):] = 1
            query_slots_mask_list.append(q_mask)

        # Pad to max length in batch
        max_len = max([x.shape[0] for x in combined_embeds_list])
        padded_embeds = torch.zeros(batch_size, max_len, self.config.hidden_size, device=device, dtype=target_dtype)
        padded_attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=target_dtype)
        final_query_mask = torch.zeros(batch_size, max_len, device=device, dtype=target_dtype) # [B, Max_Len]
        new_position_ids = torch.arange(0, max_len, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, max_len)
        for b in range(batch_size):
            length = combined_embeds_list[b].shape[0]
            padded_embeds[b, :length] = combined_embeds_list[b]
            padded_attention_mask[b, :length] = 1 
            final_query_mask[b, :length] = query_slots_mask_list[b]

        # --- 掩码处理：2D -> 4D ---
        if attention_mask is None or attention_mask.dim() == 2:
            mask_to_use = padded_attention_mask
            causal_mask = self._prepare_decoder_attention_mask(
                mask_to_use,
                (batch_size, max_len),
                padded_embeds,
                0
            )
        #     causal_mask = _prepare_4d_causal_attention_mask(
        #     mask_to_use,
        #     (batch_size, max_len),
        #     padded_embeds,
        #     past_key_values_length=0
        # )
        else:
            causal_mask = attention_mask
        if causal_mask is not None:
            causal_mask = causal_mask.to(target_dtype)
        hidden_states = padded_embeds
        # attention_mask 仅用于处理 padding
        attention_mask = padded_attention_mask

        # ------------------------------------------------------------------
        # 3. Llama Layers + Gated Cross Attention
        # ------------------------------------------------------------------
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        gca_idx = 0
        for i, decoder_layer in enumerate(self.layers):
            target_gca = None
            if i % self.gca_interval == 0 and gca_idx < len(self.gca_layers):
                target_gca = self.gca_layers[gca_idx]
                gca_idx += 1

            # if self.gradient_checkpointing and self.training:
            #     # 显存优化关键：包裹 checkpoint
            #     # 2. 显式传递所有依赖项：张量 + 模块
            #     def checkpoint_fn(hidden, mask, pos):
            #         if mask is not None:
            #             mask = mask.to(hidden.dtype)

            #         layer_out = decoder_layer(
            #             hidden,
            #             attention_mask=mask,
            #             position_ids=pos,
            #             past_key_value=None
            #         )
            #         return layer_out[0]

            #     hidden_states = torch.utils.checkpoint.checkpoint(
            #         checkpoint_fn,
            #         hidden_states,
            #         causal_mask,
            #         new_position_ids,
            #         use_reentrant=False
            #     )
            # else:
            #     layer_outputs = decoder_layer(hidden_states, attention_mask=causal_mask, position_ids=new_position_ids)
            #     hidden_states = layer_outputs[0]
            layer_outputs = decoder_layer(
                hidden_states, 
                attention_mask=causal_mask, 
                position_ids=new_position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            if target_gca is not None:
                hidden_states = target_gca(hidden_states, feat_kv, text_kv)

        hidden_states = self.norm(hidden_states)
        self._final_query_mask = final_query_mask
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            # 将 query_mask 传递出去供 Head 使用
            past_key_values=None 
        )

    def get_monitor_stats(self):
        # 记录全局时间缩放系数（通常这个不需要激活函数，直接记录数值）
        stats = {
            "custom_stats/time_embed_scale": self.time_embed_scale.item()
        }
        
        # 遍历监控每一层 GCA 的门控值
        for i, gca in enumerate(self.gca_layers):
            # 1. 门控系数经过 tanh (范围在 -1 到 1)
            # 注意：即使参数是标量，torch.tanh 也能正常处理
            stats[f"gca_gate/layer_{i}_attn_tanh"] = torch.tanh(gca.attn_gate).item()
            stats[f"gca_gate/layer_{i}_ff_tanh"] = torch.tanh(gca.ff_gate).item()

            if hasattr(gca, 'attn_ratio'):
                stats[f"gca_monitor/layer_{i}_attn_ratio"] = gca.attn_ratio
                
            if hasattr(gca, 'ff_ratio'):
                stats[f"gca_monitor/layer_{i}_ff_ratio"] = gca.ff_ratio

            # 2. lambda 系数经过 sigmoid (范围在 0 到 1)
            if hasattr(gca, 'fusion_lambda'):
                stats[f"gca_gate/layer_{i}_lambda_sigmoid"] = torch.sigmoid(gca.fusion_lambda).item()
            
            if gca.attn_gate.grad is not None:
                stats[f"gca_grad/l{i}_attn_gate"] = gca.attn_gate.grad.item()
            if gca.ff_gate.grad is not None:
                stats[f"gca_grad/l{i}_ff_gate"] = gca.ff_gate.grad.item()
            if gca.fusion_lambda.grad is not None:
                stats[f"gca_grad/l{i}_fusion_lambda"] = gca.fusion_lambda.grad.item()
                
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
            obs_mask_60[:, 144:] = mask_obs[:, 144:]
            obs_mask_60 = obs_mask_60.bool() # gt.shape = [1, 216, 2]
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
