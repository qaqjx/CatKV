from __future__ import annotations

from pathlib import Path
import sys


REQUEST_RATE_DIR = (
    Path(__file__).resolve().parents[2] / "exp" / "request_rate"
)
if str(REQUEST_RATE_DIR) not in sys.path:
    sys.path.insert(0, str(REQUEST_RATE_DIR))

import test_large_prompt as profile_helpers  # noqa: E402


def test_extract_server_profile_handles_interleaved_request_logs():
    req_a = "cmpl-req-a-0"
    req_b = "cmpl-req-b-0"
    lines = [
        f"[PROFILE] scheduler.lookup: req={req_a} prompt_tokens=6000 took 1.00ms",
        f"[PROFILE] scheduler.lookup: req={req_b} prompt_tokens=6000 took 2.00ms",
        f"[PROFILE] start_load_kv: req={req_a} process_tokens 6000 tokens / 10 segments took 3.00ms",
        f"[PROFILE] start_load_kv: req={req_b} process_tokens 6000 tokens / 10 segments took 4.00ms",
        f"[PROFILE] start_load_kv: req={req_a} blender.blend total took 500.00ms",
        f"[PROFILE] arrival_to_first_blender_done: req={req_a} 800.00ms",
        f"[PROFILE] start_load_kv: req={req_a} total load-path took 550.00ms",
        f"[PROFILE] true_server_ttft: req={req_a} arrival_to_first_token=900.00ms queue=10.00ms prefill=700.00ms",
        f"[PROFILE] start_load_kv: req={req_b} blender.blend total took 650.00ms",
        f"[PROFILE] arrival_to_first_blender_done: req={req_b} 1000.00ms",
        f"[PROFILE] start_load_kv: req={req_b} total load-path took 680.00ms",
        f"[PROFILE] true_server_ttft: req={req_b} arrival_to_first_token=1100.00ms queue=12.00ms prefill=760.00ms",
    ]

    profile = profile_helpers._extract_server_profile(lines, req_a)

    assert profile is not None
    assert profile["scheduler_lookup_ms"] == 1.0
    assert profile["process_tokens_ms"] == 3.0
    assert profile["server_blend_envelope_ms"] == 500.0
    assert profile["server_blender_done_ms"] == 800.0
    assert profile["server_load_path_ms"] == 4.0
    assert profile["server_total_load_path_ms"] == 550.0
    assert profile["server_true_ttft_ms"] == 900.0
    assert profile["server_queue_ms"] == 10.0
    assert profile["server_prefill_ms"] == 500.0
    assert profile["server_scheduled_to_first_token_ms"] == 700.0


def test_extract_server_profile_ignores_unscoped_reuse_logs_from_other_requests():
    req_a = "cmpl-req-a-0"
    req_b = "cmpl-req-b-0"
    lines = [
        f"[PROFILE] scheduler.lookup: req={req_a} prompt_tokens=6000 took 1.00ms",
        f"[PROFILE] get_reuse_kv: req={req_b} layer 1 decompressed 10 chunks in 400.00ms and copied to buffers in 50.00ms",
        f"[PROFILE] get_reuse_kv: req={req_a} layer 1 decompressed 10 chunks in 12.00ms and copied to buffers in 3.00ms",
        f"[PROFILE] start_load_kv: req={req_a} process_tokens 6000 tokens / 10 segments took 3.00ms",
        f"[PROFILE] start_load_kv: req={req_a} blender.blend total took 500.00ms",
        f"[PROFILE] arrival_to_first_blender_done: req={req_a} 800.00ms",
        f"[PROFILE] start_load_kv: req={req_a} total load-path took 550.00ms",
        f"[PROFILE] true_server_ttft: req={req_a} arrival_to_first_token=900.00ms queue=10.00ms prefill=700.00ms",
    ]

    profile = profile_helpers._extract_server_profile(lines, req_a)

    assert profile is not None
    assert profile["decompress_ms"] == 12.0
    assert profile["copy_ms"] == 3.0
