from asyncio.log import logger
import os
import threading
import time
from sympy import sequence
import torch
import torch.distributed as dist
from typing import Any, Dict, List, Optional
import flashinfer
import lmcache.c_ops as lmc_ops

from lmcache.v1.compute.blend.Cacheblend import CacheBlend
from lmcache.v1.compute.blend.compress.abstract import CompressType
from lmcache.v1.compute.blend.kvmanager import KVCacheManager, profile_log
from lmcache.v1.compute.blend.shared_key_sv import (
    get_shared_key_sv_filename,
    uuid_tensor_to_path_id,
)

store_kvcache_dir = "vllm/kvcache/"


def _get_env_compress_type_name() -> str:
    """
    Read compression type from env and normalize to uppercase for consistent paths.
    """
    return os.environ.get("LMCACHE_COMPRESS_TYPE", "OURS").upper()


def _resolve_catkv_ops_config(
    catkv_ops_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return catkv_ops_config or {}


def _get_config_compress_type_name(
    catkv_ops_config: Optional[Dict[str, Any]] = None,
) -> str:
    config = _resolve_catkv_ops_config(catkv_ops_config)
    if config.get("compress_type"):
        return str(config["compress_type"]).upper()
    return _get_env_compress_type_name()


def get_kvcache_filename(
    hash_str: str,
    layer_idx: int = -1,
    device: str = "cuda:0",
    compress_type: Optional[str] = None,
):
    compress_type = _get_env_compress_type_name() if compress_type is None else str(compress_type).upper()
    filename = os.path.join(
        store_kvcache_dir,
        compress_type,
        f"{hash_str}-layer_{layer_idx}-device_{device}.bin",
    )
    return filename


class AdaptiveKVCacheAttention:
    def __init__(self, phase):
        self.phase = phase

    def prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor = None,
    ):

        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)

        if mask is None:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, causal=True)
        else:
            o = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)

        return o

    def decode(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor
    ):
        # q, k, v: (batch_size, num_heads, len_q, dim_head)
        q = query.contiguous().squeeze(0)
        k = key.contiguous().squeeze(0)
        v = value.contiguous().squeeze(0)
       
        q = q.squeeze(0)
        o = flashinfer.single_decode_with_kv_cache(
            q, k, v, use_tensor_cores=True
        )

        return o.unsqueeze(0)

class ContextManager:
    def __init__(
        self,
        position_embedding,
        layer_num: int,
        num_heads_kv: int = 8,
        head_dim: int = 128,
        batch_size: int = 1,
        dtype = torch.bfloat16,
        gpu_connector = None,
        device: str = "cuda",
        catkv_ops_config: Optional[Dict[str, Any]] = None,
    ):
        self.gpu_connector = gpu_connector
        self.dtype = dtype
        self.batch_size = batch_size
        self.head_dim = head_dim
        self.num_heads_kv = num_heads_kv
        self.position_embedding = position_embedding
        self.attention = AdaptiveKVCacheAttention(phase="prefill")
        self.initialized = False
        self.device = device
        self.layer_num = layer_num
        self.catkv_ops_config = _resolve_catkv_ops_config(catkv_ops_config)
        self.compress_type = _get_config_compress_type_name(
            self.catkv_ops_config
        )
        self.phase = "prefill"
        self.shape = (self.batch_size , 1 , self.num_heads_kv , self.head_dim) 
        self.kv_manager = KVCacheManager.get_instance(
            shape=self.shape,
            layer_num=self.layer_num,
            device=self.device,
            compress_type=self.compress_type,
            catkv_ops_config=self.catkv_ops_config,
        )

        self.lengths = [0 for _ in range(layer_num)]
        self.kv = [None for _ in range(layer_num)]

        self.reuse_kv_finished = [False for _ in range(layer_num)]
        self.tasks = [None for _ in range(layer_num)] # prefetch tasks
        self.ping_pong_cache = None
        self.compress_data = [None for _ in range(layer_num)]
        self.active_hash_text: List[str] = []
        self.active_indices: List[List[int]] = []
        self.indices: List[List[int]] = []
        self.requested_hash_text: List[str] = []
        self.requested_indices: List[List[int]] = []
        self.missing_hash_text: List[str] = []
        self.missing_indices: List[List[int]] = []
        self.active_request_id: Optional[str] = None

    def reset(self):
        self.lengths = [0 for _ in range(self.layer_num)]
        self.kv = [None for _ in range(self.layer_num)]
        self.reuse_kv_finished = [False for _ in range(self.layer_num)]
        self.tasks = [None for _ in range(self.layer_num)] # prefetch tasks
        self.ping_pong_cache = None
        self.compress_data = [None for _ in range(self.layer_num)]
        self.active_hash_text = []
        self.active_indices = []
        self.indices = []
        self.requested_hash_text = []
        self.requested_indices = []
        self.missing_hash_text = []
        self.missing_indices = []
        self.active_request_id = None

    def _req_prefix(self) -> str:
        active_request_id = getattr(self, "active_request_id", None)
        return (
            f"req={active_request_id} "
            if active_request_id
            else ""
        )

    def has_valid_reuse(self) -> bool:
        return len(self.active_indices) > 0

    def get_effective_hash_text(self) -> List[str]:
        return list(self.active_hash_text)

    def get_effective_indices(self) -> List[List[int]]:
        return [list(index) for index in self.active_indices]

    def _initialize_reuse_request(
        self, text_hash: List[str], indices: List[List[int]], kv_len: int
    ) -> None:
        self.active_hash_text = list(text_hash or [])
        self.active_indices = [list(index) for index in (indices or [])]
        self.requested_hash_text = list(text_hash or [])
        self.requested_indices = [list(index) for index in (indices or [])]
        self.missing_hash_text = []
        self.missing_indices = []
        self.indices = self.get_effective_indices()
        self.compress_data = [None for _ in range(self.layer_num)]

        alloc_start = time.perf_counter()
        self.ping_pong_cache = [
            {
                "key": torch.zeros(
                    (self.batch_size, kv_len, self.num_heads_kv * self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
                "value": torch.zeros(
                    (self.batch_size, kv_len, self.num_heads_kv * self.head_dim),
                    dtype=self.dtype,
                    device=self.device,
                ),
            }
            for _ in range(2)
        ]

        logger.info(
            "ContextManager: Initializing all_reuse_cache for "
            f"{self.ping_pong_cache[0]['key'].shape} layers."
        )
        alloc_end = time.perf_counter()
        profile_log(
            "prefetch_chunk_kv: allocated reuse cache for "
            f"{self.layer_num} layers took {(alloc_end - alloc_start) * 1000:.2f}ms"
        )

    def _is_empty_reuse_payload(self, payload: Any) -> bool:
        if payload is None:
            return True
        if isinstance(payload, list):
            return len(payload) == 0
        if isinstance(payload, tuple):
            return len(payload) == 0
        if isinstance(payload, dict):
            return len(payload) == 0
        return False

    def _is_ours_compress(self) -> bool:
        return (
            getattr(self.kv_manager, "compress_type", None) == CompressType.OURS
            or str(self.compress_type).upper() == "OURS"
        )

    def _uses_catkv_ops_prefetch(self) -> bool:
        supports_prefetch = getattr(
            self.kv_manager, "_supports_catkv_ops_prefetch", None
        )
        if supports_prefetch is not None:
            return bool(supports_prefetch())
        return False

    def _merge_split_ours_payload(self, key_sv_payload: Any, other_payload: Any) -> Any:
        if self._is_empty_reuse_payload(key_sv_payload) or self._is_empty_reuse_payload(
            other_payload
        ):
            return {}
        merged = dict(other_payload)
        merged.update(key_sv_payload)
        return merged

    def _has_complete_ours_payload(self, payload: Any) -> bool:
        if self._is_empty_reuse_payload(payload) or not isinstance(payload, dict):
            return False
        if {"key", "value"}.issubset(payload):
            return True
        return {
            "key_sv_quantized",
            "key_sv_meta",
            "key_residual_sv",
            "u_quantized",
            "u_meta",
            "value_sv_quantized",
            "value_sv_meta",
            "value_residual_sv",
        }.issubset(payload)

    def _all_ours_payloads_complete(self, payloads: Any) -> bool:
        if not isinstance(payloads, list):
            return False
        return all(self._has_complete_ours_payload(payload) for payload in payloads)

    def _base_reuse_paths(self, layer_idx: int) -> List[str]:
        return [
            get_kvcache_filename(
                text,
                layer_idx=layer_idx,
                device=self.device,
                compress_type=self.compress_type,
            )
            for text in self.active_hash_text
        ]

    def _build_key_sv_dedup_groups(self, payloads: List[Any]) -> None:
        self.key_sv_idx = list(range(len(payloads)))
        self.key_sv_filter_idx = []
        self.key_sv_uuid_path_ids = []
        uuid_list: List[torch.Tensor] = []
        for idx, payload in enumerate(payloads):
            if self._is_empty_reuse_payload(payload):
                self.key_sv_idx[idx] = -1
                continue

            uuid = payload.get("uuid") if isinstance(payload, dict) else None
            if uuid is None:
                self.key_sv_filter_idx.append(idx)
                self.key_sv_idx[idx] = len(uuid_list)
                uuid_list.append(torch.tensor([], device=self.device))
                self.key_sv_uuid_path_ids.append(str(idx))
                continue

            matched_index = -1
            for existing_idx, existing_uuid in enumerate(uuid_list):
                if uuid.equal(existing_uuid):
                    matched_index = existing_idx
                    break
            if matched_index == -1:
                self.key_sv_filter_idx.append(idx)
                self.key_sv_idx[idx] = len(uuid_list)
                uuid_list.append(uuid)
                self.key_sv_uuid_path_ids.append(uuid_tensor_to_path_id(uuid))
            else:
                self.key_sv_idx[idx] = matched_index

    def _get_catkv_ops_key_sv_dedup_groups(self, layer_idx: int):
        if layer_idx <= 2 or not hasattr(self, "key_sv_idx"):
            return None
        base_paths = self._base_reuse_paths(layer_idx)
        shared_key_sv_paths = []
        if hasattr(self, "key_sv_uuid_path_ids"):
            for group_idx, representative_idx in enumerate(self.key_sv_filter_idx):
                if group_idx >= len(self.key_sv_uuid_path_ids):
                    continue
                if representative_idx >= len(base_paths):
                    continue
                shared_key_sv_paths.append(
                    get_shared_key_sv_filename(
                        self.key_sv_uuid_path_ids[group_idx],
                        layer_idx=layer_idx,
                        base_key=base_paths[representative_idx],
                    )
                )
        return {
            "representative_indices": list(self.key_sv_filter_idx),
            "chunk_to_group": list(self.key_sv_idx),
            "shared_key_sv_paths": shared_key_sv_paths,
        }

    def _submit_ours_retrieve(self, layer_idx: int):
        base_paths = self._base_reuse_paths(layer_idx)
        if self._uses_catkv_ops_prefetch():
            dedup_groups = self._get_catkv_ops_key_sv_dedup_groups(layer_idx)
            if dedup_groups is None:
                return self.kv_manager.retrieve_keys(base_paths)
            return self.kv_manager.retrieve_keys(
                base_paths,
                dedup_key_sv_groups=dedup_groups,
            )

        key_sv_paths = [f"{path}_key_sv" for path in base_paths]
        if layer_idx > 2 and hasattr(self, "key_sv_filter_idx"):
            if hasattr(self, "key_sv_uuid_path_ids"):
                key_sv_paths = [
                    get_shared_key_sv_filename(
                        self.key_sv_uuid_path_ids[group_idx],
                        layer_idx=layer_idx,
                        base_key=base_paths[representative_idx],
                    )
                    for group_idx, representative_idx in enumerate(
                        self.key_sv_filter_idx
                    )
                    if group_idx < len(self.key_sv_uuid_path_ids)
                    and representative_idx < len(base_paths)
                ]
            else:
                key_sv_paths = [
                    path
                    for idx, path in enumerate(key_sv_paths)
                    if idx in self.key_sv_filter_idx
                ]
        other_paths = [f"{path}_other" for path in base_paths]
        return (
            self.kv_manager.retrieve_keys(key_sv_paths),
            self.kv_manager.retrieve_keys(other_paths),
        )

    def _materialize_single_retrieve_result(self, layer_idx: int, task_id):
        if self._is_ours_compress() and isinstance(task_id, tuple):
            original_task = self.compress_data[layer_idx]
            self.compress_data[layer_idx] = task_id
            try:
                return self._materialize_split_ours_payloads(layer_idx)
            finally:
                self.compress_data[layer_idx] = original_task
        return self.kv_manager.retrieve_by_task_id(task_id=task_id)

    def _materialize_ours_with_retry(self, layer_idx: int) -> List[Any]:
        timeout_s = float(
            os.environ.get("LMCACHE_CATKV_OPS_RETRIEVE_RETRY_TIMEOUT_S", "0")
        )
        interval_s = float(
            os.environ.get("LMCACHE_CATKV_OPS_RETRIEVE_RETRY_INTERVAL_S", "0.05")
        )
        deadline = None if timeout_s <= 0 else time.perf_counter() + timeout_s
        task_id = self.compress_data[layer_idx]

        while True:
            payloads = self._materialize_single_retrieve_result(layer_idx, task_id)
            if self._all_ours_payloads_complete(payloads):
                return payloads
            if deadline is not None and time.perf_counter() >= deadline:
                profile_log(
                    "materialize_ours_with_retry: timed out waiting for complete "
                    f"split payloads at layer {layer_idx}"
                )
                raise RuntimeError(
                    f"Timed out waiting for complete OURS split payloads at layer {layer_idx}"
                )
            if interval_s > 0:
                time.sleep(interval_s)
            task_id = self._submit_ours_retrieve(layer_idx)

    def _materialize_split_ours_payloads(self, layer_idx: int) -> List[Any]:
        key_sv_task_id, other_task_id = self.compress_data[layer_idx]
        key_sv_data = self.kv_manager.retrieve_by_task_id(task_id=key_sv_task_id)
        other_data = self.kv_manager.retrieve_by_task_id(task_id=other_task_id)

        if layer_idx == 1:
            self._build_key_sv_dedup_groups(key_sv_data)

        merged_payloads = []
        for idx, other_payload in enumerate(other_data):
            key_sv_index = idx
            if layer_idx > 2 and hasattr(self, "key_sv_idx"):
                key_sv_index = self.key_sv_idx[idx]

            if key_sv_index < 0 or key_sv_index >= len(key_sv_data):
                merged_payloads.append({})
                continue
            merged_payloads.append(
                self._merge_split_ours_payload(
                    key_sv_data[key_sv_index],
                    other_payload,
                )
            )
        return merged_payloads

    def _materialize_compress_data(self, layer_idx: int) -> List[Any]:
        if self._is_ours_compress():
            return self._materialize_ours_with_retry(layer_idx)
        return self.kv_manager.retrieve_by_task_id(task_id=self.compress_data[layer_idx])

    def _compact_reuse_hits(self, payloads: List[Any]) -> List[Any]:
        if not self.has_valid_reuse():
            return []

        expected_len = min(
            len(self.active_hash_text),
            len(self.active_indices),
            len(payloads),
        )

        kept_hashes: List[str] = []
        kept_indices: List[List[int]] = []
        kept_payloads: List[Any] = []
        missing_hashes: List[str] = []
        missing_indices: List[List[int]] = []

        for hash_text, index, payload in zip(
            self.active_hash_text[:expected_len],
            self.active_indices[:expected_len],
            payloads[:expected_len],
            strict=False,
        ):
            if self._is_empty_reuse_payload(payload):
                missing_hashes.append(hash_text)
                missing_indices.append(list(index))
                continue
            kept_hashes.append(hash_text)
            kept_indices.append(list(index))
            kept_payloads.append(payload)

        if expected_len < len(self.active_hash_text):
            for hash_text, index in zip(
                self.active_hash_text[expected_len:],
                self.active_indices[expected_len:],
                strict=False,
            ):
                missing_hashes.append(hash_text)
                missing_indices.append(list(index))

        self.active_hash_text = kept_hashes
        self.active_indices = kept_indices
        self.missing_hash_text = missing_hashes
        self.missing_indices = missing_indices
        self.indices = self.get_effective_indices()

        profile_log(
            "compact_reuse_hits: "
            f"requested={len(self.requested_hash_text)} "
            f"payloads={len(payloads)} "
            f"kept={len(kept_hashes)} "
            f"missing={len(missing_hashes)}"
        )
        for missing_hash, missing_index in zip(self.missing_hash_text, self.missing_indices):
            profile_log(f"compact_reuse_miss: {missing_hash} , index {missing_index}")

        if not self.has_valid_reuse():
            self.compress_data = [None for _ in range(self.layer_num)]
            self.ping_pong_cache = None

        return kept_payloads

    def _offload_missing_chunks(
        self,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        layer_idx: int,
        blend_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self.missing_hash_text:
            return
        profile_log(f"key shape {key_tensor.shape} , missing indices {self.missing_indices}")
        for text_hash, indices in zip(
            self.missing_hash_text,
            self.missing_indices,
            strict=False,
        ):
            start, end = indices
            self.kv_manager.offload_layer_data(
                get_kvcache_filename(
                    text_hash,
                    layer_idx=layer_idx,
                    device=self.device,
                    compress_type=self.compress_type,
                ),
                (
                    key_tensor[:, start:end, :, :],
                    value_tensor[:, start:end, :, :],
                ),
                layer_idx=layer_idx,
                group_uuid=self._resolve_group_uuid(blend_meta),
            )

    def _resolve_group_uuid(
        self, blend_meta: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        if blend_meta is None:
            return None
        if blend_meta.get("group_uuid") is not None:
            return str(blend_meta["group_uuid"])
        if blend_meta.get("request_id") is not None:
            return str(blend_meta["request_id"])
        return None

    def _transfer_page(self, layer_idx: int , blend_meta: Dict[str, Any] ):
        kvcaches = blend_meta["kvcaches"]
        slot_mapping = blend_meta["slot_mapping"]

        if self.kv[layer_idx] is None:
            return

        kv = torch.cat(self.kv[layer_idx], dim=0).contiguous().reshape(2, -1, self.num_heads_kv * self.head_dim)

        # Get KV cache capacity (num_blocks * block_size)
        # kvcaches shape: [2, num_blocks, block_size, num_heads, head_dim]
        kv_capacity = kvcaches[layer_idx].shape[1] * kvcaches[layer_idx].shape[2]
        kv_len = kv.shape[1]
        slot_len = slot_mapping.shape[0]

        # Truncate to minimum length if mismatched
        if kv_len != slot_len:
            min_len = min(kv_len, slot_len)
            kv = kv[:, :min_len, :]
            slot_mapping = slot_mapping[:min_len]

        # Filter out-of-bounds slot indices
        valid_mask = slot_mapping < kv_capacity
        if not valid_mask.all():
            valid_indices = valid_mask.nonzero(as_tuple=True)[0]
            slot_mapping = slot_mapping[valid_indices]
            kv = kv[:, valid_indices, :]

        if kv.shape[1] == 0:
            self.kv[layer_idx] = None
            return

        lmc_ops.single_layer_kv_transfer(
            kv,
            kvcaches[layer_idx],
            slot_mapping,
            False,
            False,
            (kvcaches[0].shape[0] == 2)
        )
    
        self.kv[layer_idx] = None

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

    def save_all_kv_tensors(self, text_hash:str,  scores = None):
        """
        Save the key and value tensors to the disk.
        hash_str: the hash string of the text
        """ 

        for idx , kv in enumerate(self.kv):
            self.kv_manager.store_data(
                get_kvcache_filename(
                    text_hash,
                    layer_idx=idx,
                    device=self.device,
                    compress_type=self.compress_type,
                ),
                kv,
                scores,
                layer_idx=idx,
            )
 
    def store_chunks_kv(self, store_text_hashs: List , store_indices: List, layer_idx: int):

        for text_hash , indices in zip(store_text_hashs , store_indices):
            key , value = self.kv[layer_idx]
            key = key[:, indices[0]:indices[1], :, :]
            value = value[:, indices[0]:indices[1], :, :]

            # slice the chunk kv cache
            self.kv_manager.offload_compress_data(
                get_kvcache_filename(
                    text_hash,
                    layer_idx=layer_idx,
                    device=self.device,
                    compress_type=self.compress_type,
                ),
                (key, value),
                layer_idx=layer_idx,
            )

    def prefetch_chunk_kv(self, text_hash: list , indices: list , kv_len: int , layer_idx: int):
        """
        Prefetch the kv cache from the SSD to the CPU.
        text_hash: list of hash strings
        """
        if layer_idx >= self.layer_num:
            return

        if layer_idx == 1:
            self._initialize_reuse_request(text_hash, indices, kv_len)

        if not self.has_valid_reuse():
            return

        retrieve_start = time.perf_counter()
        if self._is_ours_compress():
            self.compress_data[layer_idx] = self._submit_ours_retrieve(layer_idx)
        else:
            self.compress_data[layer_idx] = self.kv_manager.retrieve_keys(
                self._base_reuse_paths(layer_idx)
            )
        retrieve_end = time.perf_counter()
        profile_log(
            "prefetch_chunk_kv: retrieve_keys for layer "
            f"{layer_idx} ({len(self.active_hash_text)} hashes) took "
            f"{(retrieve_end - retrieve_start) * 1000:.2f}ms"
        )

        if not self._is_ours_compress():
            self.compress_data[layer_idx] = self._materialize_compress_data(layer_idx)
            if layer_idx == 1:
                self.compress_data[layer_idx] = self._compact_reuse_hits(
                    self.compress_data[layer_idx]
                )
        elif layer_idx == 1:
            self.compress_data[layer_idx] = self._materialize_single_retrieve_result(
                layer_idx, self.compress_data[layer_idx]
            )
            self.compress_data[layer_idx] = self._compact_reuse_hits(
                self.compress_data[layer_idx]
            )
            if self.has_valid_reuse():
                self._build_key_sv_dedup_groups(self.compress_data[layer_idx])

    def get_reuse_kv(self, layer_idx: int):
        if self.reuse_kv_finished[layer_idx]:
            profile_log(
                f"get_reuse_kv: {self._req_prefix()}layer {layer_idx} already prepared, skipping."
            )
            return

        if self._is_ours_compress() and not self._all_ours_payloads_complete(
            self.compress_data[layer_idx]
        ):
            retrieve_task_start = time.perf_counter()
            self.compress_data[layer_idx] = self._materialize_compress_data(layer_idx)
            retrieve_task_end = time.perf_counter()
            profile_log(
                f"get_reuse_kv: {self._req_prefix()}materialize complete OURS payloads for layer {layer_idx} took {(retrieve_task_end - retrieve_task_start) * 1000:.2f}ms"
            )
        elif not isinstance(self.compress_data[layer_idx], list):
            retrieve_task_start = time.perf_counter()
            self.compress_data[layer_idx] = self._materialize_compress_data(layer_idx)
            retrieve_task_end = time.perf_counter()
            profile_log(
                f"get_reuse_kv: {self._req_prefix()}retrieve_by_task_id for layer {layer_idx} took {(retrieve_task_end - retrieve_task_start) * 1000:.2f}ms"
            )
        
        if layer_idx == 1:
            self.compress_data[layer_idx] = self._compact_reuse_hits(
                self.compress_data[layer_idx]
            )

        decompress_time = 0.0
        copy_time = 0.0
        idx = 0
        for id, compressed_data in enumerate(self.compress_data[layer_idx]):
            profile_log(f"id {id}: ,compressed data keys {compressed_data.keys() if isinstance(compressed_data, dict) else 'N/A'}")

            decompress_start = time.perf_counter()
            if not compressed_data:
                continue
            if isinstance(compressed_data, dict) and {"key", "value"}.issubset(
                compressed_data
            ):
                key = compressed_data["key"]
                value = compressed_data["value"]
            else:
                key , value = self.kv_manager.compressor.decompress(compressed_data, kv_len=self.indices[idx][1] - self.indices[idx][0])
            decompress_time += time.perf_counter() - decompress_start
            profile_log(f"retrieved key shape {key.shape} , original indices {self.indices[idx]}")
            copy_start = time.perf_counter()
            self.ping_pong_cache[layer_idx % 2]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(key.reshape(self.ping_pong_cache[layer_idx % 2]["key"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)
            self.ping_pong_cache[layer_idx % 2]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].copy_(value.reshape(self.ping_pong_cache[layer_idx % 2]["value"][: , self.indices[idx][0] : self.indices[idx][1], :].shape) , non_blocking=True)
            idx += 1
            copy_time += time.perf_counter() - copy_start

        profile_log(
            f"get_reuse_kv: {self._req_prefix()}layer {layer_idx} decompressed {len(self.compress_data[layer_idx])} chunks in {decompress_time * 1000:.2f}ms "
            f"and copied to buffers in {copy_time * 1000:.2f}ms"
        )

        reshape_start = time.perf_counter()
        self.ping_pong_cache[layer_idx % 2]["key"] = self.ping_pong_cache[layer_idx % 2]["key"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
        self.ping_pong_cache[layer_idx % 2]["value"] = self.ping_pong_cache[layer_idx % 2]["value"].reshape(self.batch_size, -1, self.num_heads_kv, self.head_dim)
        reshape_end = time.perf_counter()
        profile_log(
            f"get_reuse_kv: {self._req_prefix()}reshape and finalize buffers for layer {layer_idx} took {(reshape_end - reshape_start) * 1000:.2f}ms"
        )

        self.compress_data[layer_idx] = None
        self.reuse_kv_finished[layer_idx] = True

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
        total_start = time.perf_counter()
        self.active_request_id = self._resolve_group_uuid(blend_meta)

        len_k = pre_rope_key.size(1)
        if not self.initialized:
            self.init(pre_rope_query, pre_rope_key)

        positions = torch.arange(0, len_k, device=self.device) + self.lengths[layer_idx]

        rope_start = time.perf_counter()
        post_rope_query, post_rope_key = self.position_embedding(
            pre_rope_query.contiguous(), 
            pre_rope_key.contiguous(), 
            positions.unsqueeze(0).expand(self.batch_size, -1)
        )
        rope_end = time.perf_counter()

        # step 1 : compute the attention
        attn_start = time.perf_counter()
        o = self.attention.prefill(post_rope_query, post_rope_key, value)
        attn_end = time.perf_counter()
        o = o.unsqueeze(0)
        
        offload_start = time.perf_counter()
        if blend_meta["state"] == "store":
            # Offload each finished prefill layer to CPU through the KV manager's
            # dedicated transfer stream before it is retrieved again later.
            self.kv_manager.offload_layer_data(
                get_kvcache_filename(
                    blend_meta["hash_text"],
                    layer_idx=layer_idx,
                    device=self.device,
                    compress_type=self.compress_type,
                ),
                (pre_rope_key , value),
                layer_idx=layer_idx,
                group_uuid=self._resolve_group_uuid(blend_meta),
            )
        elif blend_meta["state"] == "retrieve":
            self._offload_missing_chunks(
                pre_rope_key, value, layer_idx, blend_meta
            )
        offload_end = time.perf_counter()
        # step 2 : offload the kv cache to manager
        self.update_kv(post_rope_key, value, layer_idx)

        self.lengths[layer_idx] += len_k

        transfer_start = time.perf_counter()
        self._transfer_page(layer_idx , blend_meta)
        transfer_end = time.perf_counter()
        assert o.size(1) == post_rope_query.size(1)

        profile_log(
            f"prefill: {self._req_prefix()}"
            f"layer={layer_idx} rope={(rope_end - rope_start) * 1000:.2f}ms "
            f"attn={(attn_end - attn_start) * 1000:.2f}ms "
            f"offload={(offload_end - offload_start) * 1000:.2f}ms "
            f"transfer={(transfer_end - transfer_start) * 1000:.2f}ms "
            f"total={(time.perf_counter() - total_start) * 1000:.2f}ms"
        )

        return o
    
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
        total_start = time.perf_counter()
        self.active_request_id = self._resolve_group_uuid(blend_meta)
        assert query.size(1) == blend_meta["input_len"]

        reuse_start = time.perf_counter()
        self.get_reuse_kv(layer_idx)
        reuse_end = time.perf_counter()
        blender = CacheBlend(blend_meta , device=query.device)
  
        blend_forward_start = time.perf_counter()
        recomputed_idx = blender.blend_forward(
            query, key, value, (
                self.ping_pong_cache[layer_idx % 2]["key"], 
                self.ping_pong_cache[layer_idx % 2]["value"]
            ),
            self.active_indices
        )
        blend_forward_end = time.perf_counter()

        # step 3: compute the attention
        prefill_start = time.perf_counter()
        o = self.prefill(query, key, value, layer_idx ,blend_meta)
        prefill_end = time.perf_counter()

        o = o[:, recomputed_idx, :, :]
        profile_log(
            f"prefill_select_token: {self._req_prefix()}"
            f"layer={layer_idx} get_reuse_kv={(reuse_end - reuse_start) * 1000:.2f}ms "
            f"blend_forward={(blend_forward_end - blend_forward_start) * 1000:.2f}ms "
            f"prefill={(prefill_end - prefill_start) * 1000:.2f}ms "
            f"total={(time.perf_counter() - total_start) * 1000:.2f}ms"
        )
        return o, recomputed_idx

    
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
        total_start = time.perf_counter()
        self.active_request_id = self._resolve_group_uuid(blend_meta)

        # step1: concatenate the key and value
        kv_len = blend_meta["input_len"]

        reuse_start = time.perf_counter()
        self.get_reuse_kv(layer_idx)
        reuse_end = time.perf_counter()
        # Shard along the head dimension
        assert self.ping_pong_cache is not None, "!!! should retrieve all layers' kv in prefill_select_token"
        retrieve_layer_key = self.ping_pong_cache[layer_idx % 2]["key"]
        retrieve_layer_value = self.ping_pong_cache[layer_idx % 2]["value"]

        assert pre_rope_key.size(2) == self.num_heads_kv
        retrieve_layer_key[:, positions, ...] = pre_rope_key
        retrieve_layer_value[:, positions, ...] = value
    
        # rotary the query and key
        rope_start = time.perf_counter()
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
        rope_end = time.perf_counter()
        
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

        attn_start = time.perf_counter()
        out = self.attention.prefill(post_rope_query, post_rope_key, retrieve_layer_value , mask).unsqueeze(0)
        attn_end = time.perf_counter()

        offload_start = time.perf_counter()
        if blend_meta["state"] == "retrieve":
            self._offload_missing_chunks(
                retrieve_layer_key, retrieve_layer_value, layer_idx, blend_meta
            )
        offload_end = time.perf_counter()

        # For tensor parallel, we need to update KV with the full tensors for proper storage
        self.update_kv(post_rope_key, retrieve_layer_value, layer_idx)
        self.lengths[layer_idx] += kv_len

        assert out.size(1) == positions.size(0)
        transfer_start = time.perf_counter()
        self._transfer_page(layer_idx , blend_meta)
        transfer_end = time.perf_counter()

        profile_log(
            f"prefill_blend: {self._req_prefix()}"
            f"layer={layer_idx} get_reuse_kv={(reuse_end - reuse_start) * 1000:.2f}ms "
            f"rope={(rope_end - rope_start) * 1000:.2f}ms "
            f"attn={(attn_end - attn_start) * 1000:.2f}ms "
            f"offload={(offload_end - offload_start) * 1000:.2f}ms "
            f"transfer={(transfer_end - transfer_start) * 1000:.2f}ms "
            f"total={(time.perf_counter() - total_start) * 1000:.2f}ms"
        )

        return out
