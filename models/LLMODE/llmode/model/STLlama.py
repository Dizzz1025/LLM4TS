#    Copyright 2023 Haotian Liu
#    This file has been modified from its original version in the UrbanGPT repository.
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

"""
This script implements the Large Language Model (LLM) portion of the LLMODE framework,
which is designed for irregular spatio-temporal forecasting. It adapts a standard
Llama model to accept and process continuous-time dynamics from an ODE encoder
and discrete statistical priors.

The core components are:
1.  **STLlamaConfig**: A custom configuration class for the model.
2.  **STLlamaModel**: The main model that injects spatio-temporal information into
    a frozen Llama backbone via token compression and gated cross-attention.
3.  **STLlamaForCausalLM**: A wrapper class that adds a final regression head for
    the forecasting task and computes the training loss.

This implementation corresponds to Sections 3.3, 3.4, and 3.5 of the paper
"LLMODE: Aligning ODEs with LLMs via Gated Token Injection for Irregular
Spatio-Temporal Forecasting".
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .PerceiverResampler import PerceiverResampler, TimeEmbedding, GatedCrossAttentionBlock


# =============================================================================
# 1. Model Configuration
# =============================================================================

class STLlamaConfig(LlamaConfig):
    """
    Custom configuration for the STLlama model. It extends the base LlamaConfig
    with parameters specific to the LLMODE framework.
    """
    model_type = "STLlama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # --- Spatio-Temporal Feature Configuration ---
        # Dimension of the input latent states 'h' from the ODE encoder.
        self.st_input_dim = kwargs.get("st_input_dim", 50)
        
        # --- Perceiver Resampler Configuration (for Token Compression, Sec. 3.4) ---
        # Number of latent tokens for dynamics evidence (K in the paper).
        self.resampler_num_latents = kwargs.get("resampler_num_latents", 5)
        # Number of latent tokens for auxiliary descriptor priors (K in the paper).
        self.text_resampler_num_latents = kwargs.get("text_resampler_num_latents", 5)
        self.resampler_depth = kwargs.get("resampler_depth", 2)
        self.resampler_heads = kwargs.get("resampler_heads", 8)
        self.resampler_dim_head = kwargs.get("resampler_dim_head", 64)
        
        # --- Gated Cross-Attention Injection Configuration (Sec. 3.5) ---
        # The interval of Transformer layers at which to inject external information.
        self.gca_layer_interval = kwargs.get("gca_layer_interval", 16)

        # --- Prediction Head Configuration (Sec. 3.5, Eq. 21) ---
        # The number of target variables to predict (e.g., 2 for inflow/outflow).
        self.target_dim = kwargs.get("target_dim", 2)


# =============================================================================
# 2. Core Spatio-Temporal Llama Model
# =============================================================================

class STLlamaModel(LlamaModel):
    """
    The core STLlama model. It modifies the standard Llama architecture to
    incorporate spatio-temporal information through a dual-source gated injection
    mechanism, as described in the LLMODE paper.
    """
    config_class = STLlamaConfig

    def __init__(self, config: STLlamaConfig):
        super(STLlamaModel, self).__init__(config)

        # --- Modules for Spatio-Temporal Feature Processing ---

        # Projects the latent states 'h' from the ODE's dimension to the LLM's hidden dimension.
        # See Section 3.4: "we first linearly project the latent states to the LLM hidden dimension D".
        self.st_projector = nn.Linear(config.st_input_dim, config.hidden_size)

        # Compresses the variable-length dynamics evidence (projected 'h') into a fixed set
        # of K memory tokens. This is the "Token Compression" mechanism (Sec. 3.4).
        self.feat_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.resampler_num_latents
        )

        # Compresses the tokenized auxiliary statistical descriptors into a fixed set of K prior tokens.
        # This is the second branch of the "Token Compression" (Sec. 3.3).
        self.text_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.text_resampler_num_latents
        )
        
        # Encodes time values into embeddings for labeling query slots and memory tokens.
        # Corresponds to the TimeEmbed function in Eq. 15.
        self.time_embedding = TimeEmbedding(config.hidden_size)
        
        # Learnable scaling factor for fusing time embeddings with memory tokens.
        # Corresponds to the projection matrix W_r in Eq. 16.
        self.time_embed_scale = nn.Parameter(torch.tensor(0.1))

        # --- Modules for Gated Cross-Attention Injection (Sec. 3.5) ---
        self.gca_interval = config.gca_layer_interval
        num_gca_layers = config.num_hidden_layers // self.gca_interval
        self.gca_layers = nn.ModuleList([
            GatedCrossAttentionBlock(
                dim=config.hidden_size,
                heads=config.resampler_heads,
                dim_head=config.resampler_dim_head
            ) for _ in range(num_gca_layers)
        ])

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            # --- Custom Inputs for LLMODE ---
            h: Optional[torch.Tensor] = None,                   # ODE latent states [B, T_raw, D_ode]
            mask_obs: Optional[torch.Tensor] = None,            # Observation mask [B, T_raw]
            stat_input_ids: Optional[torch.LongTensor] = None,  # Tokenized descriptors [B, L_stat]
            stat_attention_mask: Optional[torch.Tensor] = None, # Attention mask for descriptors
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        target_dtype = self.st_projector.weight.dtype

        # ==================================================================
        # Step 1: Encode External Evidence into Memory Tokens
        # This corresponds to "Token Compression" (Sec. 3.4) and "Auxiliary
        # Descriptor Tokenization" (Sec. 3.3).
        # ==================================================================

        # --- Path A: Dynamics Evidence (from ODE latent states 'h') ---
        if h is not None:
            # Project ODE states to LLM dimension and compress using the feature resampler.
            h_projected = self.st_projector(h.to(target_dtype))
            
            # The resampler returns compressed tokens (feat_kv) and their attention-weighted
            # center times (center_times), as per Eq. 13.
            T_max = mask_obs.shape[1]
            t_input = (torch.arange(0, T_max, device=h.device).float() / (T_max - 1)).unsqueeze(0)
            feat_kv, center_times = self.feat_resampler(h_projected, t_input)

            # Fuse with time embeddings to create time-labeled memory tokens, as per Eq. 16.
            if center_times is not None:
                time_emb = self.time_embedding(center_times)
                feat_kv = feat_kv + self.time_embed_scale * time_emb
        else:
            raise ValueError("Input 'h' (ODE hidden variables) is required.")
            
        # --- Path B: Auxiliary Prior (from statistical text descriptors) ---
        text_kv = None
        if stat_input_ids is not None:
            # Embed tokenized text and compress using the text resampler to get prior tokens 'P'.
            stat_embeds = self.embed_tokens(stat_input_ids).to(target_dtype)
            text_kv, _ = self.text_resampler(stat_embeds)

        # ==================================================================
        # Step 2: Construct Input Prompt with Query Slots
        # This corresponds to the "Prompt construction" part of Sec. 3.5.
        # The prompt is composed of textual descriptions and time-aware query slots.
        # ==================================================================
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        
        batch_size, device = inputs_embeds.shape[0], inputs_embeds.device
        combined_embeds_list = []
        query_slots_mask_list = []

        for b in range(batch_size):
            # Identify time indices that require a prediction (query slots).
            # Here, we only predict for the second half of the sequence (e.g., future).
            pred_mask = torch.zeros_like(mask_obs[b])
            pred_mask[60:] = mask_obs[b][60:]
            obs_indices = torch.nonzero(pred_mask).squeeze(-1).float()

            # Normalize time and generate time embeddings for the query slots.
            norm_time = obs_indices / (mask_obs.shape[1] - 1 + 1e-6)
            time_embeds = self.time_embedding(norm_time.to(target_dtype))
            if time_embeds.dim() == 3:
                time_embeds = time_embeds.squeeze(0)

            # Concatenate text embeddings with query slot embeddings.
            combined = torch.cat([inputs_embeds[b], time_embeds], dim=0)
            combined_embeds_list.append(combined)
            
            # Create a mask to identify query slot positions later.
            q_mask = torch.zeros(combined.shape[0], device=device, dtype=torch.bool)
            q_mask[len(inputs_embeds[b]):] = True
            query_slots_mask_list.append(q_mask)

        # Pad the combined embeddings to the maximum length in the batch.
        max_len = max([x.shape[0] for x in combined_embeds_list])
        padded_embeds = torch.zeros(batch_size, max_len, self.config.hidden_size, device=device, dtype=target_dtype)
        padded_attention_mask = torch.zeros(batch_size, max_len, device=device)
        self._final_query_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)

        for b in range(batch_size):
            length = combined_embeds_list[b].shape[0]
            padded_embeds[b, :length] = combined_embeds_list[b]
            padded_attention_mask[b, :length] = 1
            self._final_query_mask[b, :length] = query_slots_mask_list[b]

        # ==================================================================
        # Step 3: Forward through Llama Layers with Gated Injection
        # This is the main loop where external information is fused into the LLM.
        # ==================================================================
        hidden_states = padded_embeds
        attention_mask = self._prepare_decoder_attention_mask(
            padded_attention_mask, (batch_size, max_len), padded_embeds, 0
        )
        position_ids = torch.arange(0, max_len, dtype=torch.long, device=device).unsqueeze(0)

        gca_idx = 0
        for i, decoder_layer in enumerate(self.layers):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )
            hidden_states = layer_outputs[0]
            
            # Perform gated cross-attention injection at specified intervals.
            # This implements Eq. 17 and Eq. 18.
            if (i + 1) % self.gca_interval == 0 and gca_idx < len(self.gca_layers):
                target_gca = self.gca_layers[gca_idx]
                hidden_states = target_gca(hidden_states, feat_kv, text_kv)
                gca_idx += 1

        hidden_states = self.norm(hidden_states)
        
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None # Caching is not used here
        )
        
    def get_monitor_stats(self):
        """
        Retrieves gate values and other statistics for monitoring during training.
        This helps analyze the behavior of the injection mechanism, as discussed
        in Sec. 4.5 and visualized in Fig. 5 of the paper.
        """
        stats = {"custom_stats/time_embed_scale": self.time_embed_scale.item()}
        for i, gca in enumerate(self.gca_layers):
            stats[f"gca_gate/layer_{i}_attn_tanh"] = torch.tanh(gca.attn_gate).item()
            stats[f"gca_gate/layer_{i}_ff_tanh"] = torch.tanh(gca.ff_gate).item()
            if hasattr(gca, 'fusion_lambda'):
                stats[f"gca_gate/layer_{i}_lambda_sigmoid"] = torch.sigmoid(gca.fusion_lambda).item()
        return stats


# =============================================================================
# 3. Causal LM for Forecasting
# =============================================================================

class STLlamaForCausalLM(LlamaForCausalLM):
    """
    A wrapper around STLlamaModel that adds a regression head for forecasting
    and computes the final loss.
    """
    config_class = STLlamaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = STLlamaModel(config)
        
        # The standard language model head (not used for regression loss).
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # A small MLP to project the hidden states of query slots to the final prediction.
        # This is the regression head `f_reg` from Eq. 21.
        self.predict_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.target_dim) 
        )
        self.post_init()

    def forward(
            self,
            gt: Optional[torch.Tensor] = None,         # Ground truth values [B, T_raw, D_out]
            labels: Optional[torch.LongTensor] = None, # Alias for gt
            **kwargs
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        # Forward pass through the main STLlamaModel.
        outputs = self.model(return_dict=True, **kwargs)
        hidden_states = outputs.last_hidden_state
        
        # The query mask identifies the positions corresponding to prediction requests.
        final_query_mask = self.model._final_query_mask
        
        loss = None
        if gt is None and labels is not None:
            gt = labels
            
        if gt is not None:
            # ==============================================================
            # Step 4: Prediction and Loss Calculation
            # This corresponds to the final regression and training objective (Eq. 22).
            # ==============================================================
            
            # Extract the hidden states of the query slots using the final mask.
            pred_hidden = hidden_states[final_query_mask]
            
            # Pass through the regression head to get the final predictions.
            preds = self.predict_head(pred_hidden)
            
            # Extract the corresponding ground truth values using the same logic.
            mask_obs = kwargs.get('mask_obs')
            obs_mask_60 = torch.zeros_like(mask_obs, dtype=torch.bool)
            obs_mask_60[:, 60:] = mask_obs[:, 60:].bool()
            true_values = gt[obs_mask_60].to(preds.device, preds.dtype)

            # Calculate the Mean Squared Error loss.
            loss_fct = MSELoss()
            loss = loss_fct(preds, true_values)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=None, 
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,),
        )

# Register the custom model and configuration with Hugging Face's Auto classes.
AutoConfig.register("STLlama", STLlamaConfig)
AutoModelForCausalLM.register(STLlamaConfig, STLlamaForCausalLM)
