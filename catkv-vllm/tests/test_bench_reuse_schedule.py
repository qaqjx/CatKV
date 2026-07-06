import asyncio
import pytest
import sys
from pathlib import Path
from contextlib import redirect_stdout
from io import StringIO

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "exp" / "request_rate"))
from exp.request_rate import bench_reuse


def test_fixed_interval_schedule_uses_inverse_rate_between_requests(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(bench_reuse.asyncio, "sleep", fake_sleep)

    async def collect():
        entries = ["req1", "req2", "req3", "req4"]
        scheduled = [
            entry
            async for entry in bench_reuse.fixed_interval_schedule(entries, request_rate=2.0)
        ]
        return entries, scheduled

    entries, scheduled = asyncio.run(collect())
    assert scheduled == entries
    assert sleeps == [0.5, 0.5, 0.5]


def test_fixed_interval_schedule_does_not_sleep_after_last_request(monkeypatch):
    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr(bench_reuse.asyncio, "sleep", fake_sleep)

    async def collect():
        return [
            entry
            async for entry in bench_reuse.fixed_interval_schedule(["only"], request_rate=1.0)
        ]

    scheduled = asyncio.run(collect())
    assert scheduled == ["only"]
    assert sleeps == []


def test_select_shared_requests_uses_prefix_and_renumbers_ordinals():
    shared_requests = [
        {"ordinal": 7, "prompt": [1, 2], "num_contexts": 2, "question": "q1"},
        {"ordinal": 8, "prompt": [3, 4], "num_contexts": 2, "question": "q2"},
        {"ordinal": 9, "prompt": [5, 6], "num_contexts": 2, "question": "q3"},
    ]

    selected = bench_reuse.select_shared_requests(shared_requests, 2)

    assert selected == [
        {"ordinal": 1, "prompt": [1, 2], "num_contexts": 2, "question": "q1"},
        {"ordinal": 2, "prompt": [3, 4], "num_contexts": 2, "question": "q2"},
    ]


def test_print_final_table_includes_first_choice_percentiles():
    rows = [
        {
            "method": "OURS",
            "rate": "1",
            "headers_avg": 6.0,
            "first_chunk_avg": 500.0,
            "avg": 480.0,
            "p50": 470.0,
            "p90": 650.0,
            "server_true_ttft_avg": 470.0,
            "server_blender_done_avg": 430.0,
            "success_rate": 1.0,
        }
    ]

    buf = StringIO()
    with redirect_stdout(buf):
        bench_reuse.print_final_table(rows)
    output = buf.getvalue()

    assert "1stP50" in output
    assert "1stP90" in output
    assert "470.00 ms" in output
    assert "650.00 ms" in output
