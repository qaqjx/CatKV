"""
Perf benchmark: measure each stage of the offload data transfer pipeline.

Stages measured:
  1. pinned memory allocation
  2. CPU tensor copy (simulating GPU->CPU since we may not have GPU)
  3. share_memory_() call
  4. queue.put() serialization + IPC send
  5. queue.get() in child process (IPC receive + deserialization)
  6. compress in child process
  7. store in child process

Tests multiple tensor sizes to find scaling behavior.
"""

import os
import sys
import time

import torch
import torch.multiprocessing as mp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from catkv.kv_manager.offload_worker import OffloadRequest, RequestType
from catkv.kv_manager.db import OffloadMode
from catkv.kv_manager.disk.safe_tensor import IOMode


def _load_compressor_cls():
    import importlib.util
    mod_name = "catkv.kv_manager.compress.cpu_compress_mp.compressor_v3"
    spec = importlib.util.spec_from_file_location(
        mod_name,
        os.path.join(
            os.path.dirname(__file__),
            "compress", "cpu_compress_mp", "compressor_v3.py",
        ),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod.CPUCompressorV3


def worker_perf(request_queue, result_queue, compressor_kwargs, db_kwargs):
    """Child process: measure get + compress + store times."""
    CPUCompressorV3 = _load_compressor_cls()
    from catkv.kv_manager.db import DataCenter

    os.environ["OMP_NUM_THREADS"] = "4"
    torch.set_num_threads(4)

    compressor = CPUCompressorV3(**compressor_kwargs)
    db = DataCenter(**db_kwargs)

    while True:
        t_get_start = time.perf_counter()
        request = request_queue.get()
        t_get_end = time.perf_counter()

        if request.request_type == RequestType.SHUTDOWN:
            break

        t_compress_start = time.perf_counter()
        compressed = compressor.compress(request.data)
        t_compress_end = time.perf_counter()

        t_store_start = time.perf_counter()
        db.store_data(request.key, compressed)
        t_store_end = time.perf_counter()

        result_queue.put({
            "queue_get_ms": (t_get_end - t_get_start) * 1000,
            "compress_ms": (t_compress_end - t_compress_start) * 1000,
            "store_ms": (t_store_end - t_store_start) * 1000,
        })

    db.clean()


def bench_main_side(seq_len, num_heads, head_dim, num_iters=5):
    """Measure main-process side: alloc + copy + share_memory + put."""
    results = {
        "alloc_ms": [], "copy_ms": [], "share_memory_ms": [],
        "put_ms": [], "total_main_ms": [],
    }

    ctx = mp.get_context("fork")
    request_queue = ctx.Queue(maxsize=64)
    result_queue = ctx.Queue()

    compressor_kwargs = {"ratio": 0.2, "dtype": torch.bfloat16}
    db_kwargs = {
        "max_buffer_size": 2 * 1024 * 1024 * 1024,
        "offload_mode": OffloadMode.CPU,
        "io_mode": IOMode.SAFETENSOR,
        "device": "cpu",
    }

    worker = ctx.Process(
        target=worker_perf,
        args=(request_queue, result_queue, compressor_kwargs, db_kwargs),
    )
    worker.daemon = False
    worker.start()

    child_results = []

    for i in range(num_iters):
        src_key = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16)
        src_val = torch.randn(1, seq_len, num_heads, head_dim, dtype=torch.bfloat16)

        t_total_start = time.perf_counter()

        # Stage 1: pinned memory allocation
        t0 = time.perf_counter()
        cpu_key = torch.empty_like(src_key, device="cpu", pin_memory=True)
        cpu_val = torch.empty_like(src_val, device="cpu", pin_memory=True)
        t1 = time.perf_counter()
        results["alloc_ms"].append((t1 - t0) * 1000)

        # Stage 2: copy
        t0 = time.perf_counter()
        cpu_key.copy_(src_key)
        cpu_val.copy_(src_val)
        t1 = time.perf_counter()
        results["copy_ms"].append((t1 - t0) * 1000)

        # Stage 3: share_memory_
        t0 = time.perf_counter()
        cpu_key.share_memory_()
        cpu_val.share_memory_()
        t1 = time.perf_counter()
        results["share_memory_ms"].append((t1 - t0) * 1000)

        # Stage 4: queue.put
        request = OffloadRequest(
            request_type=RequestType.STORE_DATA,
            key=f"kvcache/perf_test-layer_0-device_cpu.bin",
            data=[cpu_key, cpu_val],
        )
        t0 = time.perf_counter()
        request_queue.put(request)
        t1 = time.perf_counter()
        results["put_ms"].append((t1 - t0) * 1000)

        t_total_end = time.perf_counter()
        results["total_main_ms"].append((t_total_end - t_total_start) * 1000)

        # Collect child result
        child_res = result_queue.get(timeout=300)
        child_results.append(child_res)

    # Shutdown
    request_queue.put(OffloadRequest(request_type=RequestType.SHUTDOWN))
    worker.join(timeout=15)

    return results, child_results


def print_results(seq_len, num_heads, head_dim, main_res, child_res):
    tensor_mb = seq_len * num_heads * head_dim * 2 * 2 / (1024 * 1024)
    print(f"\n{'='*70}")
    print(f"  seq_len={seq_len}, num_heads={num_heads}, head_dim={head_dim}")
    print(f"  Tensor size: {tensor_mb:.1f} MB (key+value)")
    print(f"{'='*70}")

    def avg(lst):
        return sum(lst) / len(lst) if lst else 0

    print(f"\n  --- Main Process (sender) ---")
    for stage in ["alloc_ms", "copy_ms", "share_memory_ms", "put_ms", "total_main_ms"]:
        vals = main_res[stage]
        print(f"  {stage:20s}: avg={avg(vals):8.2f} ms  "
              f"min={min(vals):8.2f}  max={max(vals):8.2f}")

    print(f"\n  --- Child Process (receiver) ---")
    for stage in ["queue_get_ms", "compress_ms", "store_ms"]:
        vals = [r[stage] for r in child_res]
        print(f"  {stage:20s}: avg={avg(vals):8.2f} ms  "
              f"min={min(vals):8.2f}  max={max(vals):8.2f}")

    total_child = [r["queue_get_ms"] + r["compress_ms"] + r["store_ms"] for r in child_res]
    print(f"  {'total_child_ms':20s}: avg={avg(total_child):8.2f} ms  "
          f"min={min(total_child):8.2f}  max={max(total_child):8.2f}")

    # Breakdown percentage
    print(f"\n  --- Breakdown (avg) ---")
    stages_main = ["alloc_ms", "copy_ms", "share_memory_ms", "put_ms"]
    stages_child = ["queue_get_ms", "compress_ms", "store_ms"]
    all_avgs = {}
    for s in stages_main:
        all_avgs[s] = avg(main_res[s])
    for s in stages_child:
        all_avgs[s] = avg([r[s] for r in child_res])
    total = sum(all_avgs.values())
    for s, v in all_avgs.items():
        pct = v / total * 100 if total > 0 else 0
        bar = "#" * int(pct / 2)
        print(f"  {s:20s}: {v:8.2f} ms ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    configs = [
        (512,  8, 128),
        (1024, 8, 128),
        (2048, 8, 128),
        (4096, 8, 128),
    ]
    num_iters = 3

    print("Offload Pipeline Performance Benchmark")
    print(f"Iterations per config: {num_iters}")

    for seq_len, num_heads, head_dim in configs:
        main_res, child_res = bench_main_side(
            seq_len, num_heads, head_dim, num_iters
        )
        print_results(seq_len, num_heads, head_dim, main_res, child_res)
