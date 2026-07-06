import os
from pathlib import Path
import threading
import time
from sympy import sequence
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional
from catkv.kv_manager.compress.abstract_compress import CompressType
from catkv.kv_manager.kv_manager import KVCacheManager
from catkv.kv_manager.utils import (
    _catkv_nvtx_annotate,
    get_kvcache_filename,
    get_shared_key_sv_filename,
    uuid_to_tensor,
    uuid_tensor_to_path_id,
)
from catkv.strategy.Cacheblend import CacheBlend
from catkv.strategy.Kvshare import KVShare
from catkv.strategy.utils import BlenderFactory

from ..utils.attention import AdaptiveKVCacheAttention

context_idx = 1
dir = os.environ.get(
    "CATKV_DEVIATION_COMPRESS_DIR",
    str(Path(__file__).resolve().parents[3] / "deviation" / "compress"),
)
OURS_COMPRESSED_KEYS = {
    "u_quantized",
    "u_meta",
    "key_sv_quantized",
    "key_sv_meta",
    "key_residual_sv",
    "value_sv_quantized",
    "value_sv_meta",
    "value_residual_sv",
}

class ContextManager:
    def __init__(
        self,
        position_embedding,
        layer_num: int,
        num_heads_kv: int = 8,
        head_dim: int = 128,
        batch_size: int = 1,
        dtype = torch.bfloat16,
        device: str = "cuda",
    ):
        self.dtype = dtype
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.num_heads_kv = num_heads_kv
        self.position_embedding = position_embedding
        self.attention = AdaptiveKVCacheAttention(phase="prefill")
        self.initialized = False
        self.device = device
        self.layer_num = layer_num
        self.phase = "prefill"
        self.shape = (self.batch_size , 1 , self.num_heads_kv , self.head_dim) 
        self.kv_manager = KVCacheManager.get_instance(shape = self.shape ,layer_num = self.layer_num , device=self.device)

        self.lengths = [0 for _ in range(layer_num)]
        self.kv = [None for _ in range(layer_num)]

        self.reuse_kv_finished = [False for _ in range(layer_num)]
        self.tasks = [None for _ in range(layer_num)] # prefetch tasks

    def init(self, query , key):
        assert query.dim() == 4
        batch_size, _, num_heads, head_dim = query.shape
        self.num_heads = num_heads
        self.dtype = query.dtype
        self.initialized = True

    def update_kv(self, key, value, layer_idx: int):
        """
        Update key and value tensors to the context manager.
        key: (batch_size, num_heads_kv, len_k, head_dim)
        value: (batch_size, num_heads_kv, len_k, head_dim)
        """
        self.kv[layer_idx] = (key, value)

    def save_all_kv_tensors(self, text_hash,  indices):
        """
        Save the key and value tensors to the disk.
        hash_str: the hash string of the text
        """
        if isinstance(text_hash, list):
            if len(text_hash) == 1:
                text_hash = text_hash[0] 

        print(f"save chunk num {len(text_hash)} for indices {indices}")

        key_hash_list = text_hash if isinstance(text_hash, list) else [text_hash]
        group_uuid = uuid_to_tensor(",".join(str(key_hash) for key_hash in key_hash_list))
        for idx , kv in enumerate(self.kv):
            self.kv_manager.store_multi_data(
                text_hash,
                indices=indices,
                layer_idx=idx,
                device=self.device,
                kv=kv,
                group_uuid=group_uuid,
            )

    def store_chunks_kv(self, kv ,store_text_hashs: List , store_indices: List, layer_idx: int):
        key , value = kv
        for text_hash , indices in zip(store_text_hashs , store_indices):
            chunk_key = key[:, indices[0]:indices[1], :, :]
            chunk_value = value[:, indices[0]:indices[1], :, :]

            # slice the chunk kv cache
            self.kv_manager.offload_compress_data(
                get_kvcache_filename(text_hash , layer_idx=layer_idx , device=self.device),
                (chunk_key, chunk_value)
            )

    def prefetch_chunk_kv(self, text_hash: list , indices: list , kv_len: int , layer_idx: int):
        """
        Prefetch the kv cache from the SSD to the CPU.
        text_hash: list of hash strings
        """

        if layer_idx == 1:
            self.all_reuse_cache = [
                {
                    "key": torch.zeros((self.batch_size, kv_len, self.num_heads_kv * self.head_dim), dtype=self.dtype, device=self.device),
                    "value": torch.zeros((self.batch_size, kv_len, self.num_heads_kv * self.head_dim), dtype=self.dtype, device=self.device)
                }
                for _ in range(self.layer_num)
            ]
            self.compress_data = [None for _ in range(self.layer_num) ]
            self.indices = indices
        
        if layer_idx >= self.layer_num or text_hash == []:
            return

        if self.kv_manager.compress_type != CompressType.OURS:
            self.compress_data[layer_idx] = self.kv_manager.retrieve_keys(
                [
                    get_kvcache_filename(text , layer_idx=layer_idx , device=self.device)
                        for text in text_hash
                ]
            )
        else:
            base_paths = [
                get_kvcache_filename(text , layer_idx=layer_idx , device=self.device)
                    for text in text_hash
            ]
            retrieve_key_sv_path = [f"{path}_key_sv" for path in base_paths]
            retrieve_other_path =     [
                    f"{path}_other"
                        for path in base_paths
                ]

            if layer_idx > 2:
                if hasattr(self, "key_sv_uuid_path_ids"):
                    retrieve_key_sv_path = [
                        get_shared_key_sv_filename(
                            self.key_sv_uuid_path_ids[idx],
                            layer_idx=layer_idx,
                            base_key=base_paths[self.key_sv_filter_idx[idx]],
                        )
                        for idx in range(len(self.key_sv_filter_idx))
                        if idx < len(self.key_sv_uuid_path_ids)
                    ]
                else:
                    retrieve_key_sv_path = [path for idx, path in enumerate(retrieve_key_sv_path) if idx in self.key_sv_filter_idx]
            self.compress_data[layer_idx] = self.kv_manager.retrieve_keys(retrieve_key_sv_path) , self.kv_manager.retrieve_keys(retrieve_other_path)

        if self.kv_manager.compress_type != CompressType.OURS:
            self.get_reuse_kv(layer_idx)

    def get_reuse_kv(self, layer_idx: int):
        if self.reuse_kv_finished[layer_idx]:
            return

        if self.compress_data[layer_idx] is None:
            self.all_reuse_cache[layer_idx]["key"] = self.all_reuse_cache[layer_idx]["key"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
            self.all_reuse_cache[layer_idx]["value"] = self.all_reuse_cache[layer_idx]["value"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
            return

        # get the compressed data from the kv manager(u , sv)
        if self.kv_manager.compress_type == CompressType.OURS:
            key_sv_data = self.kv_manager.retrieve_by_task_id(task_id=self.compress_data[layer_idx][0])
            other_data = self.kv_manager.retrieve_by_task_id(task_id=self.compress_data[layer_idx][1])
            
            if layer_idx == 1:
                self.key_sv_idx = list(range(len(other_data))) 
                self.key_sv_filter_idx = []
                self.key_sv_uuid_path_ids = []
                uuid_entries = []
                for idx in range(len(other_data)):
                    if idx >= len(key_sv_data):
                        self.key_sv_idx[idx] = -1
                        continue

                    uuid = key_sv_data[idx].get("uuid")
                    if uuid is None:
                        self.key_sv_idx[idx] = len(self.key_sv_filter_idx)
                        self.key_sv_filter_idx.append(idx)
                        self.key_sv_uuid_path_ids.append(str(idx))
                        continue

                    flag = -1
                    for key_sv_idx, existing_uuid in uuid_entries:
                        if uuid.equal(existing_uuid):
                            flag = key_sv_idx
                            break
                    if flag == -1:
                        key_sv_idx = len(self.key_sv_filter_idx)
                        self.key_sv_filter_idx.append(idx)
                        self.key_sv_uuid_path_ids.append(uuid_tensor_to_path_id(uuid))
                        uuid_entries.append((key_sv_idx, uuid))
                        self.key_sv_idx[idx] = key_sv_idx
                    else:
                        self.key_sv_idx[idx] = flag
                                 
            if layer_idx > 2:
                merged_data = []
                for idx, other_payload in enumerate(other_data):
                    key_sv_index = self.key_sv_idx[idx] if idx < len(self.key_sv_idx) else -1
                    if (
                        key_sv_index < 0
                        or key_sv_index >= len(key_sv_data)
                        or not isinstance(other_payload, dict)
                        or not isinstance(key_sv_data[key_sv_index], dict)
                    ):
                        merged_data.append({})
                        continue
                    merged_data.append(other_payload | key_sv_data[key_sv_index])
                self.compress_data[layer_idx] = merged_data
            else:
                merged_data = []
                for idx, other_payload in enumerate(other_data):
                    if (
                        idx >= len(key_sv_data)
                        or not isinstance(other_payload, dict)
                        or not isinstance(key_sv_data[idx], dict)
                    ):
                        merged_data.append({})
                        continue
                    merged_data.append(other_payload | key_sv_data[idx])
                self.compress_data[layer_idx] = merged_data
        else:
            self.compress_data[layer_idx] = self.kv_manager.retrieve_by_task_id(task_id=self.compress_data[layer_idx])
            

        # torch.cuda.synchronize()
        # t0 = time.perf_counter_ns()
        with _catkv_nvtx_annotate(f"decompress_reuse_kv_layer{layer_idx}"):
            for idx , compressed_data in enumerate(self.compress_data[layer_idx]):
                if (
                    self.kv_manager.compress_type == CompressType.OURS
                    and (
                        not isinstance(compressed_data, dict)
                        or not OURS_COMPRESSED_KEYS.issubset(compressed_data)
                    )
                ):
                    continue

                key , value = self.kv_manager.compressor.decompress(compressed_data, kv_len=self.indices[idx][1] - self.indices[idx][0])
              
                if self.indices[idx][1] - self.indices[idx][0] != key.numel() // (self.num_heads_kv * self.head_dim):
                    self.indices[idx][1] = self.indices[idx][0] + key.numel() // (self.num_heads_kv * self.head_dim)        
            
                self.all_reuse_cache[layer_idx]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(key.reshape(self.all_reuse_cache[layer_idx]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)
                self.all_reuse_cache[layer_idx]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(value.reshape(self.all_reuse_cache[layer_idx]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)

        self.all_reuse_cache[layer_idx]["key"] = self.all_reuse_cache[layer_idx]["key"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
        self.all_reuse_cache[layer_idx]["value"] = self.all_reuse_cache[layer_idx]["value"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
        
        # torch.cuda.synchronize()
        # t1 = time.perf_counter_ns()
        # print(f"Layer {layer_idx} decompress time: {(t1 - t0) / 1e6:.2f} ms")
        self.compress_data[layer_idx] = None

        self.reuse_kv_finished[layer_idx] = True

    @_catkv_nvtx_annotate
    def prefill(
        self,
        pre_rope_query: torch.Tensor,
        pre_rope_key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        blend_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        pre_rope_query: (batch_size,len_q ,num_heads, head_dim)
        pre_rope_key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)
        """

        len_k = pre_rope_key.size(1)
        if not self.initialized:
            self.init(pre_rope_query, pre_rope_key)

        positions = torch.arange(0, len_k, device=self.device) + self.lengths[layer_idx]

        # if "store_text" in blend_meta and len(blend_meta["store_text"]) > 0:
        #     print(f"Storing chunks at layer {layer_idx} for {len(blend_meta['store_text'])} texts.")
        #     self.store_chunks_kv(
        #         (pre_rope_key, value),
        #         blend_meta["store_text"],
        #         blend_meta["store_indices"],
        #         layer_idx = layer_idx
        #     )

        post_rope_query, post_rope_key = self.position_embedding(
            pre_rope_query.contiguous(), 
            pre_rope_key.contiguous(), 
            positions.unsqueeze(0).expand(self.batch_size, -1)
        )

        # step 1 : compute the attention
        o = self.attention.prefill(post_rope_query, post_rope_key, value).unsqueeze(0)
        
        # step 2 : offload the kv cache to manager
        if blend_meta is None or blend_meta["state"] != "store":
            self.update_kv(post_rope_key, value, layer_idx)
        else:
            self.update_kv(pre_rope_key, value, layer_idx)


        self.lengths[layer_idx] += len_k

        assert o.size(1) == post_rope_query.size(1)

        return o
    
    @_catkv_nvtx_annotate
    def prefill_select_token(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        blend_meta: Dict[str, Any],
    ):
        """
        query: (batch_size,len_q ,num_heads, head_dim)
        key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)

        result: (o , recomputed_token_idx)
        o: (batch_size, num_heads, len_q, head_dim)
        recomputed_token_idx: (len_q * recompute_ratio)

        1. get the previous kv from the CPU cache
        2. select the recomputed token index
        3. compute the attention
        """
        assert query.size(1) == blend_meta["input_len"]

        self.get_reuse_kv(layer_idx)
        blender = BlenderFactory().get_blender(
            layer_idx , blend_meta,
        )
        
        if isinstance(blender, KVShare) or isinstance(blender, CacheBlend):
            blender.set_rope(self.position_embedding)

        positions = blender.blend_forward(
            query, key, value, (
                self.all_reuse_cache[layer_idx]["key"], 
                self.all_reuse_cache[layer_idx]["value"]
            )
        )
        o = self.prefill(query, key, value, layer_idx, blend_meta)
        o = o[:, positions, :, :]
        return o, positions


    @_catkv_nvtx_annotate
    def prefill_blend(
        self,
        pre_rope_query: torch.Tensor,
        pre_rope_key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
        positions: torch.Tensor,
        blend_meta: Dict[str, Any],
    ):
        """
        query: (batch_size,len_q ,num_heads, head_dim)
        key: (batch_size, len_k, num_heads_kv, head_dim)
        value: (batch_size, len_k, num_heads_kv, head_dim)
        positions: (len_q)

        result: (out , recomputed_token_idx)
        out: (batch_size, num_heads, len_q, head_dim)
        recomputed_token_idx: (len_q * recompute_ratio)
        """

        # step1: concatenate the key and value
        kv_len = blend_meta["input_len"]

        self.get_reuse_kv(layer_idx)
        # Shard along the head dimension
        retrieve_layer_key = self.all_reuse_cache[layer_idx]["key"]
        retrieve_layer_value = self.all_reuse_cache[layer_idx]["value"]

        assert pre_rope_key.size(2) == self.num_heads_kv

        # torch.cuda.synchronize()
        # t0 = time.perf_counter_ns()
        retrieve_layer_key[:, positions, ...] = pre_rope_key
        retrieve_layer_value[:, positions, ...] = value
        
        # rotary the query and key
        with _catkv_nvtx_annotate("prefill_blend_position_embedding"):
            # if "store_text" in blend_meta and len(blend_meta["store_text"]) > 0:
            #     print(f"Storing chunks at layer {layer_idx} for {len(blend_meta['store_text'])} texts.")
            #     self.store_chunks_kv(
            #         (pre_rope_key, retrieve_layer_value),
            #         blend_meta["store_text"],
            #         blend_meta["store_indices"],
            #         layer_idx = layer_idx
            #     )

            post_rope_query, _ = self.position_embedding(
                pre_rope_query.contiguous(),
                torch.zeros_like(pre_rope_query),
                positions.unsqueeze(0).expand(self.batch_size, -1)
            )
            _, post_rope_key = self.position_embedding(
                torch.zeros_like(retrieve_layer_key),
                retrieve_layer_key.contiguous(),
                torch.arange(0, kv_len, device=self.device)
                .unsqueeze(0)
                .expand(self.batch_size, -1),
            )
           
        recomputed_len = positions.size(0)
        # generate the mask
        if "mask_type" not in blend_meta["select_config"] or blend_meta["select_config"]["mask_type"] == "True":
            mask = positions.unsqueeze(1) >= torch.arange(kv_len, device=self.device)
            # True Mask: 
            #   [[1, 0, 0, 0, 0],
            #    [1, 1, 1, 0, 0],
            #    [1, 1, 1, 1, 0]]
        elif blend_meta["select_config"]["mask_type"] == "Top-Right":
            mask = torch.tril(torch.ones(positions.size(0), kv_len, device=self.device))
            # Top-Right Mask:
            #   [[1, 0, 0, 0, 0],
            #    [1, 1, 0, 0, 0],
            #    [1, 1, 1, 0, 0]]
        elif blend_meta["select_config"]["mask_type"] == "Bottom-Right":
            mask = (
                    torch.arange(kv_len, device=self.device).unsqueeze(0) 
                    <= 
                    torch.arange(kv_len - recomputed_len, kv_len, device=self.device).unsqueeze(1)
                ).float()            # Bottom-Right mask
            #   [[1, 1, 1, 0, 0],
            #    [1, 1, 1, 1, 0],
            #    [1, 1, 1, 1, 1]]

        # torch.cuda.synchronize()
        # t1 = time.perf_counter_ns()
        # print(f"Layer {layer_idx} position embedding and mask generation time: {(t1 - t0) / 1e6:.2f} ms")

        with _catkv_nvtx_annotate("attention"):
            out = self.attention.prefill(post_rope_query, post_rope_key, retrieve_layer_value , mask).unsqueeze(0)

        # For tensor parallel, we need to update KV with the full tensors for proper storage
        self.update_kv(post_rope_key, retrieve_layer_value, layer_idx)
        self.lengths[layer_idx] += kv_len

        self.all_reuse_cache[layer_idx] = None
        assert out.size(1) == positions.size(0)

        return out

    def decode(
        self,
        pre_rope_query: torch.Tensor,
        pre_rope_key: torch.Tensor,
        value: torch.Tensor,
        layer_idx: int,
    ):
        """
        query: (batch_size, num_heads, 1, head_dim)
        key: (batch_size, num_heads_kv, len_k, head_dim)
        value: (batch_size, num_heads_kv, len_k, head_dim)

        result: (out)
        out: (batch_size, num_heads, len_q, head_dim)
        """
        self.all_reuse_cache = None  # Clear the cache before decoding
        assert pre_rope_query.size(1) == 1

        # step 1: use the rotary embedding to encode the query and key
        post_rope_query, post_rope_key = self.position_embedding(
            pre_rope_query.contiguous(),
            pre_rope_key.contiguous(),
            (torch.arange(0, 1, device=self.device) + self.lengths[layer_idx])
            .unsqueeze(0)
            .expand(self.batch_size, -1),
        )
       
        cache_key, cache_value = self.kv[layer_idx]

        key = torch.cat([cache_key, post_rope_key], dim=1)
        value = torch.cat([cache_value, value], dim=1)

        o = self.attention.decode(post_rope_query, key, value).unsqueeze(0)

        self.update_kv(key, value, layer_idx)
        self.lengths[layer_idx] += 1

        assert o.size(1) == post_rope_query.size(1)
        return o
