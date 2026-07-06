from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "exp"
    / "request_rate"
    / "run_cpu_compress_dedup_experiment.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "run_cpu_compress_dedup_experiment", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_output_paths_uses_named_artifacts(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)

    assert paths["queue_log"].name == "catkv_queue.jsonl"
    assert paths["client_results"].name == "client_results.json"
    assert paths["plot_png"].name == "catkv_cpu_memory_backlog.png"
    assert paths["summary"].name == "summary.json"


def test_build_lmcache_extra_config_carries_catkv_queue_settings(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)
    args = SimpleNamespace(catkv_num_workers=32, catkv_max_rss_gib=200.0)

    config = json.loads(module.build_lmcache_extra_config(args, paths))

    catkv_config = config["catkv_ops"]
    assert catkv_config["enabled"] is True
    assert catkv_config["store_enabled"] is True
    assert catkv_config["prefetch_enabled"] is True
    assert catkv_config["compress_type"] == "OURS"
    assert catkv_config["s3_config"] == str(module.RELEASE_ROOT / "csrc" / "config" / "s3.ini")
    assert catkv_config["num_workers"] == 32
    assert catkv_config["max_rss_gib"] == 200.0
    assert catkv_config["queue_log_path"] == str(paths["queue_log"])
    assert catkv_config["queue_log_interval"] == 0.2
    assert catkv_config["run_namespace"] == tmp_path.name


def test_build_server_env_exposes_catkv_repo_on_pythonpath(tmp_path):
    module = load_module()
    paths = module.build_output_paths(tmp_path)
    args = SimpleNamespace(
        cuda_visible_devices="1",
        catkv_num_workers=32,
        catkv_max_rss_gib=200.0,
    )

    env = module.build_server_env(args, paths)

    assert str(module.ROOT) in env["PYTHONPATH"].split(":")
    assert env["LMCACHE_ENABLE_PROFILING"] == "1"
    assert env["LMCACHE_USE_CATKV_OPS"] == "1"
    assert env["LMCACHE_CATKV_OPS_STORE"] == "1"
    assert env["LMCACHE_CATKV_OPS_PREFETCH"] == "1"
    assert env["LMCACHE_CATKV_OPS_S3_CONFIG"] == str(module.RELEASE_ROOT / "csrc" / "config" / "s3.ini")
    assert env["LMCACHE_CATKV_OPS_NUM_WORKERS"] == "32"
    assert env["LMCACHE_CATKV_OPS_MAX_RSS_GIB"] == "200.0"
    assert env["LMCACHE_CATKV_OPS_QUEUE_LOG"] == str(paths["queue_log"])
    assert env["LMCACHE_CATKV_OPS_QUEUE_LOG_INTERVAL"] == "0.2"
