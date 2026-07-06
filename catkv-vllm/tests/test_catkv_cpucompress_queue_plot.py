from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


def load_plot_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "exp"
        / "request_rate"
        / "catkv_cpucompress_data"
        / "plot_queue_size.py"
    )
    if not module_path.exists():
        pytest.skip(f"missing optional plot helper: {module_path}")
    spec = importlib.util.spec_from_file_location("catkv_plot_queue_size", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_combined_plot_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "exp"
        / "request_rate"
        / "catkv_cpucompress_data"
        / "plot_queue_with_reference_cache_hit.py"
    )
    if not module_path.exists():
        pytest.skip(f"missing optional plot helper: {module_path}")
    spec = importlib.util.spec_from_file_location("catkv_plot_queue_with_cache", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_queue_plot_uses_reference_queue_memory_style(tmp_path, monkeypatch):
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    module = load_plot_module()
    module.plt = plt
    monkeypatch.setattr(plt, "close", lambda fig: None)

    rows = [
        {"time_s": 0.0, "queue_gb": 0.0},
        {"time_s": 1.0, "queue_gb": 2.0},
        {"time_s": 2.0, "queue_gb": 1.0},
    ]

    module.plot_with_matplotlib(rows, tmp_path / "queue.pdf", "Queue")

    fig = plt.gcf()
    try:
        ax = fig.axes[0]
        queue_lines = [
            line for line in ax.lines if line.get_label() == "Queue Memory"
        ]

        assert queue_lines
        assert queue_lines[0].get_color() == "darkblue"
        assert queue_lines[0].get_linewidth() >= 4
        assert ax.collections
        assert any(line.get_linestyle() == "--" for line in ax.lines)
        assert any(
            line.get_marker() == "o" and line.get_color() == "red"
            for line in ax.lines
        )
        assert ax.get_xlim()[0] == pytest.approx(0.0)
    finally:
        fig.clear()


def test_queue_series_trims_idle_time_before_first_request():
    module = load_plot_module()
    records = [
        {"ts": 100.0, "remote_queue_bytes": 0, "remote_pending_count": 0},
        {"ts": 101.0, "remote_queue_bytes": 0, "remote_pending_count": 0},
        {
            "ts": 102.0,
            "event": "tick",
            "remote_queue_bytes": module.BYTES_PER_GB,
            "remote_pending_count": 1,
        },
    ]

    rows = module.queue_series(records, start_ts=101.5)

    assert [row["time_s"] for row in rows] == [0.0, 0.5]
    assert rows[0]["event"] == "trim_start"
    assert rows[0]["queue_gb"] == 0.0
    assert rows[1]["queue_gb"] == 1.0


def test_first_request_start_ts_uses_first_reqid_line(tmp_path):
    module = load_plot_module()
    server_log = tmp_path / "server.log"
    server_log.write_text(
        "INFO 04-28 02:00:01 startup\n"
        "[2026-04-28 02:00:48,030] LMCache INFO: Reqid: cmpl-a-0\n"
        "[2026-04-28 02:00:49,143] LMCache INFO: Reqid: cmpl-b-0\n",
        encoding="utf-8",
    )

    assert module.first_request_start_ts(server_log) == module.parse_log_ts(
        "2026-04-28 02:00:48,030"
    )


def test_combined_plot_loads_cache_hit_rows_from_csv(tmp_path):
    module = load_combined_plot_module()
    cache_csv = tmp_path / "cache.csv"
    cache_csv.write_text(
        "idx,wall_time_s,cache_hit_ratio,queue_mb\n"
        "0,1.0,0.0,875\n"
        "1,2.0,0.5,1000\n"
        "2,3.0,-1.0,1000\n",
        encoding="utf-8",
    )

    rows = module.load_cache_hit_rows(cache_csv)

    assert rows == [
        {"request_index": 0, "cache_hit_ratio": 0.0, "cumulative_avg": 0.0},
        {"request_index": 1, "cache_hit_ratio": 0.5, "cumulative_avg": 0.25},
    ]


def test_combined_plot_default_canvas_is_compact():
    module = load_combined_plot_module()

    assert module.DEFAULT_FIGSIZE[0] <= 13
    assert module.DEFAULT_FIGSIZE[1] <= 4
    assert module.DEFAULT_DPI <= 160


def test_combined_plot_uses_scaled_reference_format():
    module = load_combined_plot_module()

    assert module.REFERENCE_SCALE == pytest.approx(1 / 3)
    assert module.REFERENCE_STYLE["axes.labelsize"] >= 18
    assert module.REFERENCE_LINEWIDTH >= 3
    assert module.REFERENCE_MARKERSIZE >= 6
