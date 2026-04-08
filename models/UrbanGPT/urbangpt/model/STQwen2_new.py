from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import MSELoss

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2Config, Qwen2ForCausalLM, Qwen2Model
from transformers.cache_utils import Cache, DynamicCache, SlidingWindowCache, StaticCache
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .PerceiverResampler_bias import PerceiverResampler, TimeEmbedding, GatedCrossAttentionBlock


class STQwen2Config(Qwen2Config):
    model_type = "STQwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.resampler_num_latents = kwargs.get("resampler_num_latents", 5)
        self.text_resampler_num_latents = kwargs.get("text_resampler_num_latents", 5)
        self.resampler_depth = kwargs.get("resampler_depth", 2)
        self.resampler_heads = kwargs.get("resampler_heads", 8)
        self.resampler_dim_head = kwargs.get("resampler_dim_head", 64)
        self.st_input_dim = kwargs.get("st_input_dim", 50)
        self.gca_layer_interval = kwargs.get("gca_layer_interval", 16)
        self.target_dim = kwargs.get("target_dim", 2)
        self.only_attend_immediate_media = kwargs.get("only_attend_immediate_media", False)


class STQwen2Model(Qwen2Model):
    config_class = STQwen2Config

    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.st_projector = nn.Linear(config.st_input_dim, config.hidden_size)
        self.feat_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.resampler_num_latents,
        )
        self.text_resampler = PerceiverResampler(
            dim=config.hidden_size,
            depth=config.resampler_depth,
            heads=config.resampler_heads,
            dim_head=config.resampler_dim_head,
            num_latents=config.text_resampler_num_latents,
        )
        self.time_embedding = TimeEmbedding(config.hidden_size)
        self.gca_interval = config.gca_layer_interval
        num_gca_layers = max(1, config.num_hidden_layers // self.gca_interval)
        self.gca_layers = nn.ModuleList(
            [
                GatedCrossAttentionBlock(
                    dim=config.hidden_size,
                    heads=config.resampler_heads,
                    dim_head=config.resampler_dim_head,
                )
                for _ in range(num_gca_layers)
            ]
        )
        self.gradient_checkpointing = False
        self.time_embed_scale = nn.Parameter(torch.tensor(0.1))

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        stat_input_ids: torch.LongTensor = None,
        stat_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        h: Optional[torch.Tensor] = None,
        mask_obs: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        del stat_attention_mask, output_attentions, output_hidden_states

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if h is None:
            raise ValueError("Input 'h' (ODE hidden variables) is required.")
        if mask_obs is None:
            raise ValueError("Input 'mask_obs' is required.")

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("You must provide either input_ids or inputs_embeds.")
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        target_dtype = self.st_projector.weight.dtype
        inputs_embeds = inputs_embeds.to(target_dtype)
        h = h.to(target_dtype)
        mask_obs = mask_obs.to(inputs_embeds.device)

        # ------------------------------------------------------------------
        # 1) Build latent memories from ODE hidden states and statistical text.
        # ------------------------------------------------------------------
        t_max = mask_obs.shape[1]
        idx = torch.arange(0, t_max, device=h.device).float()
        t_query = idx / (t_max - 1 + 1e-6)
        t_input = t_query.unsqueeze(0).expand(h.shape[0], -1)

        h_projected = self.st_projector(h)
        feat_kv, center_times = self.feat_resampler(h_projected, t_input=t_input)
        if center_times is not None:
            feat_kv = feat_kv + self.time_embed_scale * self.time_embedding(center_times)

        if stat_input_ids is None:
            raise ValueError("Input 'stat_input_ids' is required for the text-prior branch.")
        stat_embeds = self.embed_tokens(stat_input_ids).to(target_dtype)
        text_kv, _ = self.text_resampler(stat_embeds)

        # ------------------------------------------------------------------
        # 2) Build sequence = [text prompt] + [query slots].
        # ------------------------------------------------------------------
        batch_size = inputs_embeds.shape[0]
        device = inputs_embeds.device
        combined_embeds_list = []
        query_slots_mask_list = []

        for b in range(batch_size):
            full_mask = mask_obs[b]
            pred_mask = torch.zeros_like(full_mask)
            pred_mask[60:] = full_mask[60:]

            obs_indices = torch.nonzero(pred_mask, as_tuple=False).squeeze(-1).float()
            norm_time = (obs_indices / (t_max - 1 + 1e-6)).to(target_dtype)
            time_embeds = self.time_embedding(norm_time)
            if time_embeds.dim() == 3:
                time_embeds = time_embeds.squeeze(0)
            time_embeds = time_embeds.to(target_dtype)

            text_emb = inputs_embeds[b]
            combined = torch.cat([text_emb, time_embeds], dim=0)
            combined_embeds_list.append(combined)

            q_mask = torch.zeros(combined.shape[0], device=device, dtype=torch.bool)
            q_mask[len(text_emb):] = True
            query_slots_mask_list.append(q_mask)

        max_len = max(x.shape[0] for x in combined_embeds_list)
        padded_embeds = torch.zeros(
            batch_size,
            max_len,
            self.config.hidden_size,
            device=device,
            dtype=target_dtype,
        )
        padded_attention_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.long)
        final_query_mask = torch.zeros(batch_size, max_len, device=device, dtype=torch.bool)

        for b in range(batch_size):
            length = combined_embeds_list[b].shape[0]
            padded_embeds[b, :length] = combined_embeds_list[b]
            padded_attention_mask[b, :length] = 1
            final_query_mask[b, :length] = query_slots_mask_list[b]

        if past_key_values is not None:
            past_seen_tokens = past_key_values.get_seq_length()
        else:
            past_seen_tokens = 0

        cache_position = torch.arange(
            past_seen_tokens,
            past_seen_tokens + max_len,
            device=device,
            dtype=torch.long,
        )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        causal_mask = self._update_causal_mask(
            padded_attention_mask,
            padded_embeds,
            cache_position,
            past_key_values,
            output_attentions=False,
        )

        hidden_states = padded_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # ------------------------------------------------------------------
        # 3) Qwen2 decoder layers + injected gated cross attention.
        # ------------------------------------------------------------------
        gca_idx = 0
        for i, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )
            hidden_states = layer_outputs[0] if isinstance(layer_outputs, tuple) else layer_outputs

            if i % self.gca_interval == 0 and gca_idx < len(self.gca_layers):
                hidden_states = self.gca_layers[gca_idx](hidden_states, feat_kv, text_kv)
                gca_idx += 1

        hidden_states = self.norm(hidden_states)
        self._final_query_mask = final_query_mask

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


    def _update_causal_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Optional[Cache],
        output_attentions: bool,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and past_key_values is not None:
                is_padding_right = attention_mask[:, -1].sum().item() != input_tensor.size(0)
                if is_padding_right:
                    raise ValueError(
                        "You are attempting to perform batched generation with padding_side='right'. "
                        "For Flash Attention 2, please use tokenizer.padding_side='left'."
                    )
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)
        using_sliding_window_cache = isinstance(past_key_values, SlidingWindowCache)

        if (
            self.config._attn_implementation == "sdpa"
            and not (using_static_cache or using_sliding_window_cache)
            and not output_attentions
        ):
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                sliding_window=self.config.sliding_window,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]

        if using_sliding_window_cache or using_static_cache:
            target_length = past_key_values.get_max_cache_shape()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=input_tensor.shape[0],
            config=self.config,
            past_key_values=past_key_values,
        )

        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type in ["cuda", "xpu"]
            and not output_attentions
        ):
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask

    @staticmethod
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: Optional[torch.Tensor],
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config: Qwen2Config,
        past_key_values: Optional[Cache],
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length),
                fill_value=min_dtype,
                dtype=dtype,
                device=device,
            )
            diagonal_attend_mask = torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)

            if config.sliding_window is not None:
                if not isinstance(past_key_values, SlidingWindowCache) or sequence_length > target_length:
                    sliding_attend_mask = torch.arange(target_length, device=device) <= (
                        cache_position.reshape(-1, 1) - config.sliding_window
                    )
                    diagonal_attend_mask.bitwise_or_(sliding_attend_mask)

            causal_mask *= diagonal_attend_mask
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)

            if attention_mask is not None:
                causal_mask = causal_mask.clone()
                if attention_mask.shape[-1] > target_length:
                    attention_mask = attention_mask[:, :target_length]
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask

    def get_monitor_stats(self):
        stats = {"custom_stats/time_embed_scale": self.time_embed_scale.item()}
        for i, gca in enumerate(self.gca_layers):
            stats[f"gca_gate/layer_{i}_attn_tanh"] = torch.tanh(gca.attn_gate).item()
            stats[f"gca_gate/layer_{i}_ff_tanh"] = torch.tanh(gca.ff_gate).item()
            if hasattr(gca, "attn_ratio"):
                stats[f"gca_monitor/layer_{i}_attn_ratio"] = gca.attn_ratio
            if hasattr(gca, "ff_ratio"):
                stats[f"gca_monitor/layer_{i}_ff_ratio"] = gca.ff_ratio
            if hasattr(gca, "fusion_lambda"):
                stats[f"gca_gate/layer_{i}_lambda_sigmoid"] = torch.sigmoid(gca.fusion_lambda).item()
            if gca.attn_gate.grad is not None:
                stats[f"gca_grad/l{i}_attn_gate"] = gca.attn_gate.grad.item()
            if gca.ff_gate.grad is not None:
                stats[f"gca_grad/l{i}_ff_gate"] = gca.ff_gate.grad.item()
            if gca.fusion_lambda.grad is not None:
                stats[f"gca_grad/l{i}_fusion_lambda"] = gca.fusion_lambda.grad.item()
        return stats


class STQwen2ForCausalLM(Qwen2ForCausalLM):
    config_class = STQwen2Config

    def __init__(self, config):
        super().__init__(config)
        self.model = STQwen2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.predict_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.target_dim),
        )
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        stat_input_ids: torch.LongTensor = None,
        stat_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        h: Optional[torch.Tensor] = None,
        mask_obs: Optional[torch.Tensor] = None,
        gt: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
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
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        final_query_mask = self.model._final_query_mask.bool()

        loss = None
        if gt is None and labels is not None:
            gt = labels
        if gt is not None:
            pred_hidden = hidden_states[final_query_mask]
            preds = self.predict_head(pred_hidden)
            obs_mask_60 = torch.zeros_like(mask_obs)
            obs_mask_60[:, 60:] = mask_obs[:, 60:]
            obs_mask_60 = obs_mask_60.bool()
            true_values = gt[obs_mask_60].to(preds.device).to(preds.dtype)
            loss = MSELoss()(preds, true_values)

        return CausalLMOutputWithPast(
            loss=loss,
            past_key_values=outputs.past_key_values,
            hidden_states=(hidden_states,),
        )


AutoConfig.register("STQwen2", STQwen2Config)
AutoModelForCausalLM.register(STQwen2Config, STQwen2ForCausalLM)
