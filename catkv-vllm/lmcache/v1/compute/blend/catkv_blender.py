# SPDX-License-Identifier: Apache-2.0
# Standard
import hashlib
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.context_manager import ContextManager
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.blend.rope import RotaryEmbeddingESM
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)

def serialize_and_hash(input_list):
    serialized_data = str(input_list.tolist()).encode('utf-8')
    hash_object = hashlib.md5(serialized_data)
    return hash_object.hexdigest()


def build_hash_text(
    token_chunk,
    *,
    world_size: int,
    run_namespace: str = "",
):
    actual_hash = serialize_and_hash(token_chunk)
    prefix = f"{run_namespace}-" if run_namespace else ""
    return f"{prefix}wordl_size{world_size}{actual_hash}"


def _get_catkv_ops_config(extra_config):
    return extra_config.get("catkv_ops") or {}


class CatKVCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        extra_config = config.extra_config or {}
        enable_sparse = extra_config.get("enable_sparse", False)
        catkv_ops_config = _get_catkv_ops_config(extra_config)
        self.run_namespace = str(
            catkv_ops_config.get("run_namespace", "")
        ).strip()

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)
        self.num_heads_kv = vllm_model.model.layers[0].self_attn.attn.num_kv_heads

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

        ## hack context manager

        # get the current 

        # by the model name to get the embedding
        self.rope = RotaryEmbeddingESM(
                vllm_model.model.layers[0].self_attn.rotary_emb,
                128,
                gpu_connector.device
            )
        self.context_manager = ContextManager(
            self.rope,
            self.num_layers,
            num_heads_kv=self.num_heads_kv,
            device=gpu_connector.device,
            gpu_connector=self.gpu_connector,
            catkv_ops_config=catkv_ops_config,
        )

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        blend_meta: Optional[dict] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        self.layerwise_model.compute_layer(tokens, blend_meta)

        self.metadata.clean()

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()

        # get the hash and perform lookup
        kvcaches = kwargs.get("kvcaches", None)
        slot_mapping = kwargs.get("slot_mapping", None)

        # is blend
        flag = 0 if  len(kwargs.get("starts", [])) <= 1 else 1

        index = [(start , end) for start , end in zip(kwargs.get("starts", []) , kwargs.get("ends", [])) ]

        # Debug: print blend info
        print(f"[blend] tokens_len={len(tokens)}, flag={flag}, starts={kwargs.get('starts', [])}, ends={kwargs.get('ends', [])}", flush=True)

        hash_text = []
        for idx in index:
            token_chunk = tokens[idx[0]:idx[1]]
            hash_text.append(
                build_hash_text(
                    token_chunk,
                    world_size=kwargs.get("hash_val")[0].world_size,
                    run_namespace=self.run_namespace,
                )
            )

        blend_meta = {
            "context_manager": self.context_manager,
            "gpu_connector": self.gpu_connector,
            "kvcaches": kvcaches,
            "slot_mapping": slot_mapping,
            "flag": flag,
            "hash_text": hash_text,
            "indices": index,
            "input_len": len(tokens),
            "request_id": kwargs.get("request_id"),
        }

        blend_meta["state"] = "store" if flag == 0 else "retrieve"
        blend_meta["select_config"] = {
            "mask_type": "True"
        }

        if blend_meta["state"] == "store":
            blend_meta["hash_text"] = blend_meta["hash_text"][0]
            # print(f"Storing KV cache into CatKV backend {blend_meta['hash_text']}")
        # else:
            # print(f"index mapping {blend_meta['indices']}")
            # print(f"retrieving KV cache from CatKV backend {blend_meta['hash_text']}")
        self.blend_layer(tokens, mask, blend_meta, **kwargs)
