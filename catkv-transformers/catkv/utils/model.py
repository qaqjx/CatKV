import time
from typing import Optional, Tuple
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast

import torch
import torch.distributed as dist

from catkv.kv_manager.context_manager import ContextManager
from catkv.kv_manager.utils import get_kvcache_filename
from catkv.strategy.abstract_blend import ProcessType

def model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask = None,
    position_ids = None,
    past_key_values = None,
    inputs_embeds = None,
    use_cache = None,
    output_attentions = None,
    output_hidden_states = None,
    return_dict = None,
    *args,
    **kwargs
):    
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)
        if hasattr(self, "config") and hasattr(self.config, "scale_emb"):
            inputs_embeds = inputs_embeds * self.config.scale_emb

    if past_key_values is None:
        past_key_values = ContextManager(
            position_embedding=self.position_bias,
            layer_num=len(self.layers),
            num_heads_kv=self.config.num_key_value_heads,
            head_dim = 128,
            dtype=inputs_embeds.dtype,
            device=self.device,
        )
        if hasattr(self, "blend_meta") and "compress_type" in self.blend_meta:
            compress_type = self.blend_meta["compress_type"]
            compress_config = self.blend_meta.get("compress_config", {})
            past_key_values.kv_manager.set_compress_type(compress_type, compress_config)

        # check chunks exist
        if self.blend_meta["state"] != "store":
            chunk_exist = past_key_values.kv_manager.check_keys_exist(
                [get_kvcache_filename(
                    text_hash, device=self.device ,layer_idx=0
                ) for text_hash in self.blend_meta["hash_text"]]
            )
            # print(f"Chunk exist status for current input: {chunk_exist}")
            chunk_exist = [True for x in chunk_exist]
            
            self.blend_meta["chunk_exist"] = chunk_exist
            
            # if the chunk no exist , we will store it
            # remove the texts whose chunks not exist
            store_text_hashs = []
            store_indices = []
            retrieved_text_hashs = []
            retrieved_indices = []

            for idx, exist in enumerate(chunk_exist):
                if not exist:
                    store_text_hashs.append(self.blend_meta["hash_text"][idx])
                    store_indices.append(self.blend_meta["indices"][idx])
                else:
                    retrieved_text_hashs.append(self.blend_meta["hash_text"][idx])
                    retrieved_indices.append(self.blend_meta["indices"][idx])
            
            # remove the texts meta from hash_text and indices
            self.blend_meta["hash_text"] = retrieved_text_hashs
            self.blend_meta["indices"] = retrieved_indices

            self.blend_meta["store_text"] = store_text_hashs
            self.blend_meta["store_indices"] = store_indices

            # print(f"store Chunks size: {len(store_indices)}")
        
    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    recompute_idx = None if self.blend_meta["state"] == "normal" else torch.arange(seq_length).to(hidden_states.device)

    offset = 1
    for i, decoder_layer in enumerate(self.layers):
        # torch.cuda.synchronize()
        # t0 = time.perf_counter_ns()

        if self.blend_meta["state"] != "store" and self.blend_meta["select_strategy"] != ProcessType.DEFAULT and self.blend_meta["phase"] == "prefill":
            past_key_values.prefetch_chunk_kv(
                self.blend_meta["hash_text"],
                self.blend_meta["indices"],
                self.blend_meta["input_len"],
                layer_idx = i + 1
            )

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        decoder_layer.layer_idx = i
        decoder_layer.self_attn.blend_meta = self.blend_meta 
        decoder_layer.self_attn.layer_idx = i
        decoder_layer.self_attn.recompute_idx = recompute_idx

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=self.position_bias,
            past_key_value=past_key_values,
            output_attentions=output_attentions,
            use_cache=use_cache,   
        )

        recompute_idx = decoder_layer.self_attn.recompute_idx
        hidden_states = layer_outputs[0]
        
        # torch.cuda.synchronize()
        # t1 = time.perf_counter_ns()
        # if self.blend_meta["phase"] == "prefill":
        #     print(f"Layer {i} forward time: {(t1 - t0) / 1e6:.2f} ms")

    
    hidden_states = hidden_states[:, -1:, :]
    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, None] if v is not None)
    
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
    )

def decoder_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, present_key_value, idx = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache
    )

    # inject reuse 
    if self.layer_idx == 1 and idx is not None:
        residual = residual[: , idx, :]
    
    if residual.size(-2) != 0:
        hidden_states = residual + hidden_states

    residual = hidden_states.clone()

    for start_idx in range(0 , hidden_states.size(-2), 8192):
        end_idx = min(start_idx + 8192, hidden_states.size(-2))
        hidden_states[:, start_idx:end_idx, :] = self.post_attention_layernorm(hidden_states[:, start_idx:end_idx, :])
        hidden_states[:, start_idx:end_idx, :] = self.mlp(hidden_states[:, start_idx:end_idx, :])
    
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

def attention_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask = None,
    position_ids = None,
    past_key_value = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    batch_size, sequence_len , hidden_size = hidden_states.size()
    
    # Check if tensor parallel is enabled
    is_tp_enabled = dist.is_initialized() and dist.get_world_size() > 1
    tp_size = dist.get_world_size() if is_tp_enabled else 1

    if self.blend_meta["phase"] == "decode" and past_key_value.phase != "decode":
        past_key_value.phase = "decode"
        # past_key_value.save()

    heads_per_rank = self.config.num_attention_heads // tp_size
    kv_heads_per_rank = self.config.num_key_value_heads // tp_size

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    
    # adapt the Qwen3 model
    if hasattr(self, "q_norm"):
        input_shape = hidden_states.shape[:-1]
        hidden_shape =  (*input_shape, -1, self.head_dim)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(key_states.view(hidden_shape))

    key_states = key_states.view(batch_size, sequence_len, kv_heads_per_rank, self.head_dim)
    value_states = value_states.view(batch_size, sequence_len, kv_heads_per_rank, self.head_dim)
    query_states = query_states.view(batch_size, sequence_len, heads_per_rank, self.head_dim)

    if self.blend_meta["phase"] == "prefill":
        if self.blend_meta["state"] != "store" and self.blend_meta["select_strategy"] != ProcessType.DEFAULT and self.layer_idx:
            if self.layer_idx > 1:
                o = past_key_value.prefill_blend(query_states, key_states, value_states, self.layer_idx , self.recompute_idx, self.blend_meta)
            elif self.layer_idx == 1:
                o, self.recompute_idx = past_key_value.prefill_select_token(
                    query_states, key_states, value_states, self.layer_idx , self.blend_meta
                )
        else:
            o = past_key_value.prefill(query_states, key_states, value_states, self.layer_idx ,self.blend_meta)
    else:
        # decode phase
        o = past_key_value.decode(query_states, key_states, value_states, self.layer_idx)
    

    o = o.view(batch_size, -1, self.head_dim * heads_per_rank)
    
    o = self.o_proj(o)

    return o, past_key_value, self.recompute_idx