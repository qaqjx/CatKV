from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "blend_kv_v1"
    / "run_wikimqa_outputs.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location("run_wikimqa_outputs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configure_environment_respects_none_compress_type(monkeypatch):
    module = load_module()
    args = module.parse_args.__wrapped__ if hasattr(module.parse_args, "__wrapped__") else None
    namespace = type(
        "Args",
        (),
        {
            "cuda_visible_devices": "1",
            "catkv_s3_config": "s3.ini",
            "catkv_num_workers": 32,
            "catkv_max_rss_gib": 200.0,
            "compress_type": "NONE",
        },
    )()

    module.configure_environment(namespace)

    assert module.os.environ["LMCACHE_COMPRESS_TYPE"] == "NONE"
