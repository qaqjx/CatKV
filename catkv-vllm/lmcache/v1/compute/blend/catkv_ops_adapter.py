from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import os
import threading
import time
from typing import Any, Optional

import torch
from lmcache.v1.compute.blend.compress.abstract import CompressType
ENABLE_PROFILING = os.environ.get("LMCACHE_ENABLE_PROFILING", "0") == "1"

def profile_log(msg: str, *args, **kwargs):
    if ENABLE_PROFILING:
        print(f"[PROFILE] {msg}", flush=True)


def _env_value(name: str, default=None):
    value = os.environ.get(name)
    return default if value is None else value


def _env_flag(name: str, default: bool) -> bool:
    value = _env_value(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class CatKVOpsAdapterConfig:
    enabled: bool
    config_path: Optional[str]
    store_enabled: bool = True
    prefetch_enabled: bool = True
    ratio: float = 0.15
    num_workers: int = 32
    max_queue_bytes: int = 0
    dtype: torch.dtype = torch.bfloat16
    prefetch_workers: int = 16
    queue_log_path: Optional[str] = None
    queue_log_stdout: Optional[bool] = None
    queue_log_interval: Optional[float] = None


class CatKVOpsBlendAdapter:
    def __init__(self, config: CatKVOpsAdapterConfig):
        self.config = config
        self.available = False
        self.unavailable_reason: Optional[str] = None
        self._ops = None
        self._store = None
        self._scheduler = None
        self._split_prefetch_tasks: dict[Any, dict[str, Any]] = {}
        self._queue_log_path = (
            config.queue_log_path
            if config.queue_log_path is not None
            else _env_value("LMCACHE_CATKV_OPS_QUEUE_LOG")
        )
        self._queue_log_stdout = (
            config.queue_log_stdout
            if config.queue_log_stdout is not None
            else _env_flag("LMCACHE_CATKV_OPS_QUEUE_LOG_STDOUT", False)
        )
        self._queue_log_interval = (
            config.queue_log_interval
            if config.queue_log_interval is not None
            else float(
                _env_value("LMCACHE_CATKV_OPS_QUEUE_LOG_INTERVAL", "0.5")
            )
        )
        self._queue_monitor_stop = threading.Event()
        self._queue_monitor_thread: Optional[threading.Thread] = None

        if not config.enabled or not (config.store_enabled or config.prefetch_enabled):
            self.unavailable_reason = "disabled"
            return

        try:
            self._ops = importlib.import_module("catkv_ops")
        except ImportError as exc:
            self.unavailable_reason = f"import-error: {exc}"
            return
    
        num_workers = self._int_env(
            "LMCACHE_CATKV_OPS_NUM_WORKERS",
            config.num_workers,
        )
        prefetch_workers = self._int_env(
            "LMCACHE_CATKV_OPS_PREFETCH_WORKERS",
            config.prefetch_workers,
        )
        if config.store_enabled:
            self._store = self._ops.CPUMemoryStore(pin_memory=True)

        if config.store_enabled and config.config_path:
            self._store.enable_remote_upload(
                config.config_path,
                ratio=config.ratio,
                dtype=config.dtype,
                num_workers=num_workers,
                max_queue_bytes=config.max_queue_bytes,
            )
        if config.prefetch_enabled and config.config_path:
            self._scheduler = self._ops.S3Schedule(
                config.config_path, prefetch_workers
            )

        self.available = True
        self._start_queue_monitor()
        self.key = {}
        self.cpu_transfer_gpu_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _int_env(self, name: str, default: int) -> int:
        value = _env_value(name)
        if value is None:
            return default
        try:
            parsed = int(value)
        except ValueError:
            return default
        return parsed if parsed > 0 else default

    def _current_rss_bytes(self) -> Optional[int]:
        try:
            with open("/proc/self/statm", "r", encoding="utf-8") as statm:
                pages = int(statm.read().split()[1])
            return pages * os.sysconf("SC_PAGE_SIZE")
        except Exception:
            return None

    def _queue_snapshot(self, event: str, path: Optional[str] = None) -> dict[str, Any]:
        snapshot: dict[str, Any] = {
            "ts": time.time(),
            "event": event,
            "rss_bytes": self._current_rss_bytes(),
        }
        if path is not None:
            snapshot["path"] = path
        if self._store is None:
            return snapshot

        if hasattr(self._store, "remote_pending_count"):
            snapshot["remote_pending_count"] = self._store.remote_pending_count()
        if hasattr(self._store, "remote_current_queue_bytes"):
            snapshot["remote_queue_bytes"] = self._store.remote_current_queue_bytes()
        if hasattr(self._store, "remote_queue_stats"):
            snapshot["remote_queue_stats"] = self._store.remote_queue_stats()
        try:
            snapshot["local_store_entries"] = len(self._store)
        except Exception:
            pass
        return snapshot

    def _write_queue_log(self, event: str, path: Optional[str] = None) -> None:
        if not self._queue_log_path and not self._queue_log_stdout:
            return
        snapshot = self._queue_snapshot(event, path)
        serialized = json.dumps(snapshot, sort_keys=True)
        if self._queue_log_path:
            with open(self._queue_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(serialized + "\n")
        if self._queue_log_stdout:
            print(f"[CATKV_OPS_QUEUE] {serialized}", flush=True)

    def _queue_monitor_loop(self) -> None:
        while not self._queue_monitor_stop.wait(self._queue_log_interval):
            try:
                self._write_queue_log("tick")
            except Exception:
                pass

    def _start_queue_monitor(self) -> None:
        if (
            (not self._queue_log_path and not self._queue_log_stdout)
            or self._queue_monitor_thread is not None
        ):
            return
        self._write_queue_log("init")
        self._queue_monitor_thread = threading.Thread(
            target=self._queue_monitor_loop,
            name="catkv-ops-queue-monitor",
            daemon=True,
        )
        self._queue_monitor_thread.start()

    def supports(self, compress_type: CompressType) -> bool:
        return self.supports_store(compress_type) or self.supports_prefetch(
            compress_type
        )

    def supports_store(self, compress_type: CompressType) -> bool:
        return (
            self.available
            and self.config.store_enabled
            and self._store is not None
            and compress_type == CompressType.OURS
        )

    def supports_prefetch(self, compress_type: CompressType) -> bool:
        return (
            self.available
            and self.config.prefetch_enabled
            and self._scheduler is not None
            and compress_type == CompressType.OURS
        )

    def _device_to_index(self, device: str | torch.device) -> int:
        if isinstance(device, torch.device):
            if device.type != "cuda":
                return -1
            return 0 if device.index is None else device.index

        if device == "cpu":
            return -1
        if device.startswith("cuda:"):
            return int(device.split(":", maxsplit=1)[1])
        if device == "cuda":
            return 0
        return -1

    def _device_to_load_target(self, device: str | torch.device) -> str:
        device_index = self._device_to_index(device)
        if device_index < 0:
            return "cpu"
        return f"cuda:{device_index}"

    def offload(self, path: str, tensors: dict[str, torch.Tensor], group_uuid: str):
        if self._store is None:
            raise RuntimeError("catkv_ops CPUMemoryStore is unavailable")
        
        profile_log(f"offload: offloading path {path} with tensor keys {tensors.keys()} , shapes {[tensor.shape for tensor in tensors.values()]}")
        result = self._store.offload(path, tensors, group_uuid)
        self._write_queue_log("offload", path)
        return result

    def _split_remote_paths(self, paths: list[str]) -> list[str]:
        return [
            split_path
            for path in paths
            for split_path in (f"{path}_key_sv", f"{path}_other")
        ]

    def _build_split_prefetch_plan(
        self,
        all_paths: list[str],
        cpu_data: list[Any],
        dedup_key_sv_groups: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        miss_indices = [
            idx
            for idx, data in enumerate(cpu_data)
            if idx < len(all_paths) and not data
        ]
        if dedup_key_sv_groups is None:
            miss_paths = [all_paths[idx] for idx in miss_indices]
            return {
                "remote_paths": self._split_remote_paths(miss_paths),
                "miss_indices": miss_indices,
                "mode": "paired",
                "logical_path_count": len(miss_paths),
            }

        chunk_to_group = list(dedup_key_sv_groups.get("chunk_to_group", []))
        shared_key_sv_paths = list(
            dedup_key_sv_groups.get("shared_key_sv_paths", [])
        )
        unique_group_to_path: dict[int, str] = {}
        missing_groups: list[int] = []
        for idx in miss_indices:
            if idx >= len(chunk_to_group):
                continue
            group_idx = int(chunk_to_group[idx])
            if group_idx < 0 or group_idx in unique_group_to_path:
                continue
            unique_group_to_path[group_idx] = all_paths[idx]
            missing_groups.append(group_idx)

        key_sv_paths = []
        for group_idx in missing_groups:
            if 0 <= group_idx < len(shared_key_sv_paths):
                key_sv_paths.append(str(shared_key_sv_paths[group_idx]))
            else:
                key_sv_paths.append(unique_group_to_path[group_idx] + "_key_sv")
        other_paths = [all_paths[idx] + "_other" for idx in miss_indices]
        return {
            "remote_paths": key_sv_paths + other_paths,
            "miss_indices": miss_indices,
            "mode": "dedup_key_sv",
            "missing_groups": missing_groups,
            "chunk_to_group": chunk_to_group,
        }

    def _prefer_gpu_results(self, result):
        if not isinstance(result, tuple) or len(result) != 2:
            return result

        cpu_results, gpu_results = result
        normalized = []
        for cpu_result, gpu_result in zip(cpu_results, gpu_results, strict=False):
            normalized.append(gpu_result if gpu_result else cpu_result)
        if len(gpu_results) > len(cpu_results):
            normalized.extend(gpu_results[len(cpu_results) :])
        elif len(cpu_results) > len(gpu_results):
            normalized.extend(cpu_results[len(gpu_results) :])
        return normalized

    def _merge_split_payloads(self, result, logical_path_count: int):
        if not isinstance(result, list):
            return result
        if len(result) == logical_path_count:
            return result

        merged_payloads = []
        for idx in range(logical_path_count):
            key_sv_payload_idx = idx * 2
            other_payload_idx = key_sv_payload_idx + 1
            if other_payload_idx >= len(result):
                merged_payloads.append({})
                continue

            key_sv_payload = result[key_sv_payload_idx]
            other_payload = result[other_payload_idx]
            merged_payloads.append(
                self._merge_split_payload(key_sv_payload, other_payload)
            )
        return merged_payloads

    def _merge_split_payloads_with_plan(self, result, plan: dict[str, Any]):
        if not isinstance(result, list):
            return result

        mode = plan.get("mode")
        if mode != "dedup_key_sv":
            return self._merge_split_payloads(
                result, int(plan.get("logical_path_count", 0))
            )

        missing_groups = list(plan.get("missing_groups", []))
        miss_indices = list(plan.get("miss_indices", []))
        chunk_to_group = list(plan.get("chunk_to_group", []))
        key_sv_count = len(missing_groups)
        key_sv_by_group = {
            int(group_idx): result[offset]
            for offset, group_idx in enumerate(missing_groups)
            if offset < len(result)
        }
        other_payloads = result[key_sv_count:]

        merged_payloads = []
        for other_offset, original_idx in enumerate(miss_indices):
            if (
                original_idx >= len(chunk_to_group)
                or other_offset >= len(other_payloads)
            ):
                merged_payloads.append({})
                continue
            group_idx = int(chunk_to_group[original_idx])
            key_sv_payload = key_sv_by_group.get(group_idx)
            other_payload = other_payloads[other_offset]
            merged_payloads.append(
                self._merge_split_payload(key_sv_payload, other_payload)
            )
        return merged_payloads

    def _merge_split_payload(self, key_sv_payload, other_payload):
        if not key_sv_payload or not other_payload:
            return {}

        required_key_sv_keys = {
            "key_sv_quantized",
            "key_sv_meta",
            "key_residual_sv",
        }
        required_other_keys = {
            "u_quantized",
            "u_meta",
            "value_sv_quantized",
            "value_sv_meta",
            "value_residual_sv",
        }
        if not required_key_sv_keys.issubset(key_sv_payload) or not (
            required_other_keys.issubset(other_payload)
        ):
            return {}

        merged = dict(other_payload)
        merged.update(key_sv_payload)
        return merged

    def prefetch_remote(
        self,
        paths: list[str],
        device: str | torch.device,
        dedup_key_sv_groups: Optional[dict[str, Any]] = None,
    ):
        # CPUMemoryStore owns local CPU-hit transfer semantics. Missing entries
        # are fetched from split S3 objects below.
        if self._store is None:
            cpu_data = [{} for _ in paths]
        else:
            cpu_data = self._store.load_batch(
                paths, self._device_to_load_target(device)
            )
        for path, data in zip(paths, cpu_data, strict=False):
            if data:
                profile_log(f"prefetch_remote: loaded from CPU for path {path} with data keys {data.keys() if isinstance(data, dict) else 'N/A'}")
        task_id = paths[0]
        plan = self._build_split_prefetch_plan(paths, cpu_data, dedup_key_sv_groups)
        profile_log(
            "prefetch_remote: loaded from CPU for "
            f"task_id {task_id}, miss_indices: {plan['miss_indices']}"
        )
        if plan["remote_paths"]:
            task_id = self._scheduler.submit_batch_load_to_gpu(
                plan["remote_paths"], self._device_to_index(device)
            )
            self._split_prefetch_tasks[task_id] = plan
            self.key[task_id] = cpu_data , task_id
        else:
            self.key[task_id] = cpu_data, None
        return task_id

    def get_prefetch_result(self, task_id):
        result_from_cpu, task_flag = self.key.pop(task_id)
        profile_log(f"get_prefetch_result: retrieved from CPU for task_id {task_id} with flag {task_flag}")
        if task_flag is not None:
            result = self._prefer_gpu_results(
                self._scheduler.get_batch_load_to_gpu_result(task_id)
            )
            plan = self._split_prefetch_tasks.pop(task_id, {})
            result_from_s3 = self._merge_split_payloads_with_plan(result, plan)
            count = 0
            for idx,data in enumerate(result_from_cpu):
                if not data:
                    result_from_cpu[idx] = result_from_s3[count]
                    profile_log(f"get_prefetch_result: merged data for task_id {task_id} at index {idx}")
                    profile_log(f"get_prefetch_result: merged data keys {result_from_cpu[idx].keys() if isinstance(result_from_cpu[idx], dict) else 'N/A'} for task_id {task_id} at index {idx} , remote keys{result_from_s3[count].keys() if isinstance(result_from_s3[count], dict) else 'N/A'}")
                    count += 1
            self.key.pop(task_id,None)
        
        return result_from_cpu

    def shutdown(self) -> None:
        self._queue_monitor_stop.set()
        if self._queue_monitor_thread is not None:
            self._queue_monitor_thread.join(timeout=1.0)
            self._queue_monitor_thread = None
        self._write_queue_log("shutdown")
        if self._store is not None and hasattr(self._store, "wait_remote_all"):
            self._store.wait_remote_all()



__all__ = [
    "CatKVOpsAdapterConfig",
    "CatKVOpsBlendAdapter",
]
