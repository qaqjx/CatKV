"""
Offload worker process for KV cache compression and storage.

Architecture:
- Multiple worker processes running offload_worker_loop()
- Pre-allocated shared memory ring buffer for zero-copy tensor IPC
- Lightweight MetadataMsg via mp.Queue (no tensor references, fast pickle)
- Compresses using CPUCompressorV3 (pure CPU SVD + int4 quantization)
- Stores via DataCenter (CPUBufferPool + DiskIOManager)
- Sends back new key filenames via keys_ack_queue for keys_set sync
"""

from __future__ import annotations

import atexit
import os
import queue
import threading
from dataclasses import dataclass
from enum import Enum
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.multiprocessing as mp


class RequestType(Enum):
    STORE_DATA = "store_data"
    STORE_MULTI_DATA = "store_multi_data"
    OFFLOAD_COMPRESS = "offload_compress"
    SHUTDOWN = "shutdown"


@dataclass
class OffloadRequest:
    request_type: RequestType
    key: Optional[str] = None
    layer_idx: Optional[int] = None
    data: Optional[Any] = None
    score: Optional[Any] = None
    key_hash_list: Optional[List[str]] = None
    indices: Optional[Any] = None
    device: Optional[str] = None
    kv: Optional[Any] = None
    compress_config: Optional[Dict[str, Any]] = None


@dataclass
class MetadataMsg:
    """Lightweight metadata for IPC (no tensor references, fast pickle)."""
    request_type: RequestType
    slot_idx: int = -1
    seq_len: int = 0
    key: Optional[str] = None
    key_hash_list: Optional[List[str]] = None
    indices: Optional[Any] = None
    layer_idx: Optional[int] = None
    device: Optional[str] = None


class SharedBufferPool:
    """Pre-allocated shared memory buffer pool for zero-copy IPC."""

    def __init__(self, num_slots: int, max_seq_len: int, num_heads: int,
                 head_dim: int, dtype=torch.bfloat16, ctx=None):
        self.num_slots = num_slots
        self.max_seq_len = max_seq_len
        self._buffers: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(num_slots):
            key_buf = torch.empty(1, max_seq_len, num_heads, head_dim, dtype=dtype)
            val_buf = torch.empty(1, max_seq_len, num_heads, head_dim, dtype=dtype)
            key_buf.share_memory_()
            val_buf.share_memory_()
            self._buffers.append((key_buf, val_buf))

        # Use provided context or default
        if ctx is None:
            ctx = mp.get_context()
        self._free_slots: mp.Queue = ctx.Queue()
        for i in range(num_slots):
            self._free_slots.put(i)

    def acquire(self, timeout: Optional[float] = None) -> int:
        """Acquire a free slot index (blocks until available)."""
        return self._free_slots.get(timeout=timeout)

    def release(self, slot_idx: int) -> None:
        """Return a slot to the free pool."""
        self._free_slots.put(slot_idx)

    def get_buffers(self, slot_idx: int, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get slot buffers sliced to actual seq_len."""
        key_buf, val_buf = self._buffers[slot_idx]
        return key_buf[:, :seq_len], val_buf[:, :seq_len]

    def get_raw_buffers(self, slot_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get full slot buffers (for copy_ target)."""
        return self._buffers[slot_idx]


def offload_worker_loop(
    request_queue: mp.Queue,
    keys_ack_queue: mp.Queue,
    compressor_kwargs: Dict[str, Any],
    db_kwargs: Dict[str, Any],
    shared_pool: Optional[SharedBufferPool] = None,
    done_queue: Optional[mp.Queue] = None,
) -> None:
    """Child process entry point for compression + storage offload.

    When shared_pool is provided, expects MetadataMsg on request_queue
    and reads tensors directly from the shared buffer pool.
    Falls back to legacy OffloadRequest path when shared_pool is None.
    """
    import importlib.util, sys
    mod_name = "catkv.kv_manager.compress.cpu_compress_mp.compressor_v3"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        os.path.join(os.path.dirname(__file__), "compress", "cpu_compress_mp", "compressor_v3.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    CPUCompressorV3 = mod.CPUCompressorV3

    from catkv.kv_manager.db import DataCenter

    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)

    compressor = CPUCompressorV3(**compressor_kwargs)
    db = DataCenter(**db_kwargs)

    use_shared_pool = shared_pool is not None

    print(f"[OffloadWorker {os.getpid()}] Started, waiting for messages...")

    while True:
        msg = request_queue.get()

        if msg.request_type == RequestType.SHUTDOWN:
            print(f"[OffloadWorker {os.getpid()}] Received SHUTDOWN signal")
            break

        try:
            if use_shared_pool and isinstance(msg, MetadataMsg):
                _handle_metadata_msg(msg, shared_pool, done_queue,
                                     compressor, db, keys_ack_queue)
            else:
                # Legacy path for OffloadRequest
                if msg.request_type == RequestType.STORE_DATA:
                    _handle_store_data(msg, compressor, db, keys_ack_queue)
                elif msg.request_type == RequestType.STORE_MULTI_DATA:
                    _handle_store_multi_data(msg, compressor, db, keys_ack_queue)
                elif msg.request_type == RequestType.OFFLOAD_COMPRESS:
                    _handle_offload_compress(msg, compressor, db, keys_ack_queue)
        except Exception as e:
            print(f"[OffloadWorker] Error processing {msg.request_type}: {e}")
            # Release slot on error to avoid deadlock
            if use_shared_pool and isinstance(msg, MetadataMsg) and msg.slot_idx >= 0:
                done_queue.put(msg.slot_idx)

    print(f"[OffloadWorker {os.getpid()}] Starting cleanup...")
    db.clean()
    print(f"[OffloadWorker {os.getpid()}] Cleanup complete, exiting")


def _handle_metadata_msg(msg, shared_pool, done_queue, compressor, db, keys_ack_queue):
    """Handle a MetadataMsg by reading from shared buffer pool."""
    key_buf, val_buf = shared_pool.get_buffers(msg.slot_idx, msg.seq_len)
    # Clone to decouple from shared buffer before releasing slot
    key_data = key_buf.clone()
    val_data = val_buf.clone()
    done_queue.put(msg.slot_idx)

    if msg.request_type == RequestType.STORE_DATA:
        compressed = compressor.compress([key_data, val_data])
        flag = db.store_data(msg.key, compressed)
        if flag and "layer_0" in msg.key:
            keys_ack_queue.put(msg.key.split("/")[-1])

    elif msg.request_type == RequestType.STORE_MULTI_DATA:
        from catkv.kv_manager.utils import get_kvcache_filename
        compressed_list = compressor.compress_multi(
            [key_data, val_data], indices=msg.indices
        )
        key_hash_list = msg.key_hash_list
        if isinstance(key_hash_list, str):
            key_hash_list = [key_hash_list]
        for idx, key_hash in enumerate(key_hash_list):
            key = get_kvcache_filename(
                key_hash, layer_idx=msg.layer_idx, device=msg.device
            )
            flag = db.store_data(key, compressed_list[idx])
            if flag and "layer_0" in key:
                keys_ack_queue.put(key.split("/")[-1])

    elif msg.request_type == RequestType.OFFLOAD_COMPRESS:
        compressed = compressor.compress([key_data, val_data])
        flag = db.store_data(msg.key, compressed)
        if flag and "layer_0" in msg.key:
            keys_ack_queue.put(msg.key.split("/")[-1])


def _handle_store_data(request, compressor, db, keys_ack_queue):
    compressed = compressor.compress(request.data)
    flag = db.store_data(request.key, compressed)
    if flag and "layer_0" in request.key:
        keys_ack_queue.put(request.key.split("/")[-1])


def _handle_store_multi_data(request, compressor, db, keys_ack_queue):
    from catkv.kv_manager.utils import get_kvcache_filename

    compressed_list = compressor.compress_multi(request.kv, indices=request.indices)
    key_hash_list = request.key_hash_list
    if isinstance(key_hash_list, str):
        key_hash_list = [key_hash_list]

    for idx, key_hash in enumerate(key_hash_list):
        key = get_kvcache_filename(
            key_hash, layer_idx=request.layer_idx, device=request.device
        )
        flag = db.store_data(key, compressed_list[idx])
        if flag and "layer_0" in key:
            keys_ack_queue.put(key.split("/")[-1])


def _handle_offload_compress(request, compressor, db, keys_ack_queue):
    key_tensor = request.data[0]
    value_tensor = request.data[1]
    compressed = compressor.compress([key_tensor, value_tensor])
    flag = db.store_data(request.key, compressed)
    if flag and "layer_0" in request.key:
        keys_ack_queue.put(request.key.split("/")[-1])


class OffloadManager:
    """Main-process side manager for the offload worker processes.

    Uses SharedBufferPool for zero-copy tensor IPC when buffer_pool_slots > 0.
    Supports multiple worker processes for parallel compression.
    """

    def __init__(
        self,
        compressor_kwargs: Dict[str, Any],
        db_kwargs: Dict[str, Any],
        device: str = "cuda",
        num_workers: int = 1,
        queue_size: int = 64,
        buffer_pool_slots: int = 8,
        max_seq_len: int = 8192,
        num_heads: int = 8,
        head_dim: int = 128,
        dtype=torch.bfloat16,
    ):
        if num_workers <= 0:
            raise ValueError("num_workers must be positive")

        self._compressor_kwargs = compressor_kwargs
        self._db_kwargs = db_kwargs
        self._device = device
        self._num_workers = num_workers
        self._queue_size = queue_size if queue_size > 0 else max(num_workers * 2, 16)

        self._ctx = mp.get_context("fork")
        self._request_queue: Optional[mp.Queue] = None
        self._keys_ack_queue: Optional[mp.Queue] = None
        self._workers: List[mp.Process] = []
        self._started = False

        self._transfer_stream: Optional[torch.cuda.Stream] = None
        self._lock = Lock()

        # Shared buffer pool config
        self._pool_slots = buffer_pool_slots
        self._max_seq_len = max_seq_len
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._dtype = dtype
        self._shared_pool: Optional[SharedBufferPool] = None
        self._done_queue: Optional[mp.Queue] = None
        self._reclaim_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()

        atexit.register(self.shutdown)

    def start(self) -> None:
        if self._started:
            return
        self._request_queue = self._ctx.Queue(maxsize=self._queue_size)
        self._keys_ack_queue = self._ctx.Queue()

        if self._pool_slots > 0:
            self._shared_pool = SharedBufferPool(
                self._pool_slots, self._max_seq_len,
                self._num_heads, self._head_dim, self._dtype,
                ctx=self._ctx,
            )
            self._done_queue = self._ctx.Queue()
            self._shutdown_event.clear()
            self._reclaim_thread = threading.Thread(
                target=self._reclaim_loop, daemon=True
            )
            self._reclaim_thread.start()

        # Start multiple worker processes
        for _ in range(self._num_workers):
            worker = self._ctx.Process(
                target=offload_worker_loop,
                args=(
                    self._request_queue,
                    self._keys_ack_queue,
                    self._compressor_kwargs,
                    self._db_kwargs,
                    self._shared_pool,
                    self._done_queue,
                ),
            )
            worker.daemon = False
            worker.start()
            self._workers.append(worker)

        self._started = True

    def _reclaim_loop(self) -> None:
        """Background thread: reclaim done slots from child process."""
        while not self._shutdown_event.is_set():
            try:
                slot_idx = self._done_queue.get(timeout=0.1)
                self._shared_pool.release(slot_idx)
            except queue.Empty:
                continue

    def _ensure_stream(self) -> torch.cuda.Stream:
        if self._transfer_stream is None:
            self._transfer_stream = torch.cuda.Stream(device=self._device)
        return self._transfer_stream

    def _copy_to_slot(self, key_t: torch.Tensor, value_t: torch.Tensor) -> Tuple[int, int]:
        """Acquire a slot and copy key/value tensors into it. Returns (slot_idx, seq_len)."""
        seq_len = key_t.shape[1]
        slot_idx = self._shared_pool.acquire()
        raw_key, raw_val = self._shared_pool.get_raw_buffers(slot_idx)

        if key_t.device.type == "cuda":
            stream = self._ensure_stream()
            with torch.cuda.stream(stream):
                raw_key[:, :seq_len].copy_(key_t, non_blocking=True)
                raw_val[:, :seq_len].copy_(value_t, non_blocking=True)
            stream.synchronize()
        else:
            raw_key[:, :seq_len].copy_(key_t)
            raw_val[:, :seq_len].copy_(value_t)

        return slot_idx, seq_len

    def _tensors_to_shared_cpu(
        self, tensors: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Legacy path: copy tensors to shared CPU memory."""
        stream = self._ensure_stream()
        cpu_tensors = []
        with torch.cuda.stream(stream):
            for t in tensors:
                if t.device.type == "cuda":
                    cpu_t = torch.empty(
                        t.shape, dtype=t.dtype, device="cpu", pin_memory=True
                    )
                    cpu_t.copy_(t, non_blocking=True)
                else:
                    cpu_t = t.clone()
                cpu_tensors.append(cpu_t)
        stream.synchronize()
        for t in cpu_tensors:
            if not t.is_shared():
                t.share_memory_()
        return cpu_tensors

    def submit_store_data(self, key, data, score=None, layer_idx=None):
        if not self._started:
            self.start()
        key_t, value_t = data

        if self._shared_pool is not None:
            slot_idx, seq_len = self._copy_to_slot(key_t, value_t)
            msg = MetadataMsg(
                request_type=RequestType.STORE_DATA,
                slot_idx=slot_idx, seq_len=seq_len, key=key,
            )
            self._request_queue.put(msg)
        else:
            cpu_tensors = self._tensors_to_shared_cpu([key_t, value_t])
            request = OffloadRequest(
                request_type=RequestType.STORE_DATA,
                key=key, data=cpu_tensors, score=score, layer_idx=layer_idx,
            )
            self._request_queue.put(request)

    def submit_store_multi_data(
        self, key_hash_list, indices, layer_idx, device, kv
    ):
        if not self._started:
            self.start()
        key_t, value_t = kv

        # Convert indices to List[Tuple[int, int]] before serialization
        # to avoid torch.Size or tensor casting errors
        if indices is not None:
            try:
                indices_list = []
                for idx in indices:
                    if isinstance(idx, (list, tuple)):
                        indices_list.append((int(idx[0]), int(idx[1])))
                    elif isinstance(idx, torch.Size):
                        indices_list.append((int(idx[0]), int(idx[1])))
                    elif isinstance(idx, torch.Tensor):
                        indices_list.append((int(idx[0].item()), int(idx[1].item())))
                    else:
                        # Fallback: try to convert directly
                        indices_list.append((int(idx[0]), int(idx[1])))
            except Exception as e:
                print(f"[OffloadManager] Error converting indices: {e}")
                print(f"[OffloadManager] indices type: {type(indices)}, content: {indices}")
                raise
        else:
            indices_list = None

        if self._shared_pool is not None:
            slot_idx, seq_len = self._copy_to_slot(key_t, value_t)
            msg = MetadataMsg(
                request_type=RequestType.STORE_MULTI_DATA,
                slot_idx=slot_idx, seq_len=seq_len,
                key_hash_list=key_hash_list, indices=indices_list,
                layer_idx=layer_idx, device=device,
            )
            self._request_queue.put(msg)
        else:
            cpu_tensors = self._tensors_to_shared_cpu([key_t, value_t])
            request = OffloadRequest(
                request_type=RequestType.STORE_MULTI_DATA,
                key_hash_list=key_hash_list, indices=indices_list,
                layer_idx=layer_idx, device=device, kv=cpu_tensors,
            )
            self._request_queue.put(request)

    def submit_offload_compress(self, key, data):
        if not self._started:
            self.start()

        if self._shared_pool is not None:
            key_t, value_t = data[0], data[1]
            slot_idx, seq_len = self._copy_to_slot(key_t, value_t)
            msg = MetadataMsg(
                request_type=RequestType.OFFLOAD_COMPRESS,
                slot_idx=slot_idx, seq_len=seq_len, key=key,
            )
            self._request_queue.put(msg)
        else:
            for t in data:
                if not t.is_shared():
                    t.share_memory_()
            request = OffloadRequest(
                request_type=RequestType.OFFLOAD_COMPRESS,
                key=key, data=data,
            )
            self._request_queue.put(request)

    def drain_keys_ack(self) -> set:
        new_keys = set()
        if self._keys_ack_queue is None:
            return new_keys
        while True:
            try:
                key = self._keys_ack_queue.get_nowait()
                new_keys.add(key)
            except queue.Empty:
                break
        return new_keys

    def flush(self, timeout: float = 30.0) -> None:
        """Wait for all pending tasks to complete.

        This method blocks until the request queue is empty and all workers
        have finished processing their current tasks.

        Args:
            timeout: Maximum time to wait in seconds
        """
        if not self._started:
            return

        import time
        start_time = time.time()

        # Wait until request queue is empty
        while time.time() - start_time < timeout:
            if self._request_queue.empty():
                # Queue is empty, wait a bit more to ensure workers finish
                time.sleep(0.1)
                if self._request_queue.empty():
                    break
            time.sleep(0.01)

        # Drain any remaining keys_ack messages
        self.drain_keys_ack()

    def shutdown(self, wait: bool = True) -> None:
        if not self._started:
            return

        # Send shutdown signal to all workers
        if self._shared_pool is not None:
            for _ in self._workers:
                self._request_queue.put(
                    MetadataMsg(request_type=RequestType.SHUTDOWN)
                )
        else:
            for _ in self._workers:
                self._request_queue.put(
                    OffloadRequest(request_type=RequestType.SHUTDOWN)
                )

        if wait:
            for worker in self._workers:
                worker.join(timeout=5)  # Reduced from 30s to 5s
                if worker.is_alive():
                    print(f"[OffloadManager] Worker {worker.pid} did not exit cleanly, terminating...")
                    worker.terminate()
                    worker.join(timeout=2)
                    if worker.is_alive():
                        print(f"[OffloadManager] Worker {worker.pid} still alive after terminate, killing...")
                        worker.kill()

        # Stop reclaim thread
        self._shutdown_event.set()
        if self._reclaim_thread is not None:
            self._reclaim_thread.join(timeout=5)

        self._started = False
        self._request_queue = None
        self._keys_ack_queue = None
        self._workers.clear()
        self._shared_pool = None
        self._done_queue = None
        self._reclaim_thread = None
