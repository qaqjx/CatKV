# SPDX-License-Identifier: Apache-2.0
# Third Party
from asyncio.log import logger
from gguf import Optional
import os
from torch import nn
import time
import torch

# First Party
# from lmcache.v1.compute.attention.utils import infer_attn_backend_from_vllm
from lmcache.v1.compute.positional_encoding import get_fused_rope

ENABLE_PROFILING = os.environ.get("LMCACHE_ENABLE_PROFILING", "0") == "1"


def profile_log(msg: str) -> None:
    if ENABLE_PROFILING:
        print(f"[PROFILE] {msg}", flush=True)

# TODO(Jiayi): A few things need to be tested/supported:
# TP, PP, Multimodal


class CatKVLMCLlamaModel(nn.Module):
    def __init__(
        self,
        vllm_model,
        blender,
        enable_sparse: bool = False,
    ):
        super().__init__()
        self.vllm_model = vllm_model

        self.num_layers = len(vllm_model.model.layers)

        self.vllm_attn_layers = []
        self.lmc_attn_layers = []
        for i in range(self.num_layers):
            vllm_attn = vllm_model.model.layers[i].self_attn.attn
            self.vllm_attn_layers.append(vllm_attn)

            # self.lmc_attn_layers.append(
            #     infer_attn_backend_from_vllm(vllm_attn, enable_sparse)
            # )

        # NOTE(Jiayi): better not to pass the blender in init
        # if we want to make this LMCModel more general.
        self.blender = blender

        # remove hard code
        rotary_emb = vllm_model.model.layers[0].self_attn.rotary_emb
        head_dim = rotary_emb.head_size
        max_position_embeddings = rotary_emb.max_position_embeddings
        rope_scaling = None
        base = rotary_emb.base
        is_neox_style = rotary_emb.is_neox_style
        dtype = rotary_emb.dtype
        self.fused_rotary_emb = get_fused_rope(
            head_dim,
            rotary_dim=head_dim,
            max_position=max_position_embeddings,
            base=base,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    # @torch.compile
    def compute_layer(
        self,
        input_ids: torch.Tensor,
        blend_meta: Optional[dict] = None,
    ):
        """
        Compute layers with CatKV-style KV cache management.

        Args:
            input_ids: Input token IDs
            blend_meta: Metadata containing:
                - context_manager: CatKV context manager
                - gpu_connector: GPU connector for KV cache
                - hash_text: Hash of input text
                - hash_indices: Indices of matched chunks
                - state: "store" or "retrieve" mode
        """
        context_manager = blend_meta.get("context_manager", None) if blend_meta else None
        gpu_connector = blend_meta.get("gpu_connector", None) if blend_meta else None
        hash_text = blend_meta.get("hash_text", "") if blend_meta else ""
        blend_flag = blend_meta.get("flag", 0) if blend_meta else 0
        state = blend_meta.get("state", "retrieve") if blend_meta else "retrieve"
        request_id = blend_meta.get("request_id", "") if blend_meta else ""
        req_prefix = f"req={request_id} " if request_id else ""

        input_ids = input_ids.cuda()
        input_len = len(input_ids)
        hidden_states = self.vllm_model.get_input_embeddings(input_ids)
        residual = None

        attn_output = None

        # TODO(Jiayi): Need to build `attn_metadata` more elegantly.
        # attn_metadata = self.lmc_attn_layers[0].init_attn_metadata(
        #     input_ids=input_ids,
        # )

        # Reset context manager for every new request so per-layer positions
        # and reuse buffers never leak across requests. Store requests also
        # build rope positions from scratch and must not inherit lengths from
        # previous retrieve/store executions.
        if context_manager is not None:
            context_manager.reset()

        recomputed_idx = None
        for idx, layer in enumerate(
            self.vllm_model.model.layers[
                self.vllm_model.model.start_layer : self.vllm_model.model.end_layer
            ]
        ):
            layer_start = time.perf_counter()
            logger.info(f"Computing layer {idx} with CatKV blending")
            # CatKV: Prefetch KV cache for current layer
            prefetch_ms = 0.0
            if context_manager is not None and state != "store":
                prefetch_start = time.perf_counter()
                context_manager.prefetch_chunk_kv(
                    hash_text,
                    blend_meta.get("indices", []),
                    input_len,
                    layer_idx=idx + 1,
                )
                blend_meta["hash_text"] = context_manager.get_effective_hash_text()
                blend_meta["indices"] = context_manager.get_effective_indices()
                prefetch_ms = (time.perf_counter() - prefetch_start) * 1000
                profile_log(
                    f"compute_layer: {req_prefix}layer={idx} prefetch_chunk_kv took {prefetch_ms:.2f}ms"
                )

            use_runtime_reuse = (
                blend_flag != 0
                and idx
                and context_manager is not None
                and context_manager.has_valid_reuse()
            )

            # Layer norm
            if residual is None:
                residual = hidden_states
                hidden_states = layer.input_layernorm(hidden_states)
            else:
                hidden_states, residual = layer.input_layernorm(hidden_states, residual)
            sequence_len , hidden_size = hidden_states.size()

            # Compute Q, K, V
            qkv, _ = layer.self_attn.qkv_proj(hidden_states)
            q, k, v = qkv.split(
                [
                    layer.self_attn.q_size,
                    layer.self_attn.kv_size,
                    layer.self_attn.kv_size,
                ],
                dim=-1,
            )
            q = q.view(1, sequence_len, -1, 128)
            k = k.view(1, sequence_len, -1, 128)
            v = v.view(1, sequence_len, -1, 128)

            attn_start = time.perf_counter()
            attn_mode = "prefill"
            if use_runtime_reuse:
                if idx == 1:
                    attn_mode = "prefill_select_token"
                    attn_output, recomputed_idx = context_manager.prefill_select_token(
                        q, k, v, idx, blend_meta
                    )
                    residual = residual[recomputed_idx]

                else:
                    attn_mode = "prefill_blend"
                    attn_output = context_manager.prefill_blend(
                        q, k, v, idx, recomputed_idx, blend_meta
                    )
            else:
                attn_output = context_manager.prefill(q , k , v , idx, blend_meta)
            attn_ms = (time.perf_counter() - attn_start) * 1000
            profile_log(
                f"compute_layer: {req_prefix}layer={idx} attn_path={attn_mode} took {attn_ms:.2f}ms"
            )

            num_heads = self.vllm_attn_layers[idx].num_heads
            num_kv_heads = self.vllm_attn_layers[idx].num_kv_heads
            head_size = self.vllm_attn_layers[idx].head_size

            # Reshape back
            attn_output = attn_output.view(-1, num_heads * head_size)

            # Output projection
            post_attn_start = time.perf_counter()
            hidden_states, _ = layer.self_attn.o_proj(attn_output)

            # Fully Connected (MLP)
            hidden_states, residual = layer.post_attention_layernorm(
                hidden_states, residual
            )
            hidden_states = layer.mlp(hidden_states)
            post_attn_ms = (time.perf_counter() - post_attn_start) * 1000
            profile_log(
                f"compute_layer: {req_prefix}layer={idx} post_attn_mlp took {post_attn_ms:.2f}ms"
            )
            profile_log(
                f"compute_layer: {req_prefix}"
                f"layer={idx} total took {(time.perf_counter() - layer_start) * 1000:.2f}ms"
            )
