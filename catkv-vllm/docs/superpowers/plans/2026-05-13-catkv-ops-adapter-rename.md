# catkv_ops Adapter Rename Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rename the LMCache CatKV integration from `xj_project` naming to `catkv_ops` naming while preserving existing compatibility paths.

**Architecture:** The new primary module is `catkv_ops_adapter.py`, with `CatKVOpsAdapterConfig` and `CatKVOpsBlendAdapter` as the canonical API. Existing `xj_project_adapter.py`, `XJProjectAdapterConfig`, `XJProjectBlendAdapter`, `extra_config["xj_project"]`, and `LMCACHE_XJ_*` remain as compatibility aliases or fallbacks. Runtime behavior for OURS store, prefetch, split path expansion, and payload merge stays unchanged.

**Tech Stack:** Python 3, PyTorch, pytest, LMCache blend code, `catkv_ops.CPUMemoryStore`, `catkv_ops.S3Schedule`.

---

## File Structure

- Create `lmcache/v1/compute/blend/catkv_ops_adapter.py`: canonical adapter implementation and canonical class names.
- Replace `lmcache/v1/compute/blend/xj_project_adapter.py`: compatibility shim that re-exports canonical classes and old aliases.
- Modify `lmcache/v1/compute/blend/kvmanager.py`: import canonical adapter, prefer CatKV config/env names, keep old support-method aliases.
- Modify `lmcache/v1/compute/blend/context_manager.py`: prefer `catkv_ops_config` naming, keep `xj_project_config` constructor compatibility, and check canonical support method first.
- Modify `lmcache/v1/compute/blend/taotie_blender.py`: read `extra_config["catkv_ops"]` first and fall back to `extra_config["xj_project"]`.
- Modify `lmcache/v1/compute/blend/compress/our.py`: import fused dequant functions from `catkv_ops`.
- Modify `tests/v1/test_taotie_xj_project_adapter.py`: cover new imports, compatibility imports, real package import name, new env precedence, and legacy env fallback.
- Modify `tests/v1/test_taotie_blend_empty_s3.py`: update fakes or assertions only where needed to cover canonical support methods.
- Modify `tests/v1/test_cpu_compress_dedup_runner.py` and `exp/request_rate/run_cpu_compress_dedup_experiment.py`: emit canonical `catkv_ops` config/env while preserving old config/env for compatibility.

---

### Task 1: Adapter API Tests

**Files:**
- Modify: `tests/v1/test_taotie_xj_project_adapter.py`

- [ ] **Step 1: Write failing tests for canonical and compatibility imports**

Add imports near the existing imports:

```python
from lmcache.v1.compute.blend.catkv_ops_adapter import (
    CatKVOpsAdapterConfig,
    CatKVOpsBlendAdapter,
)
```

Add tests:

```python
def test_canonical_adapter_names_are_available():
    adapter = CatKVOpsBlendAdapter(
        CatKVOpsAdapterConfig(enabled=False, config_path=None)
    )

    assert adapter.available is False
    assert adapter.unavailable_reason == "disabled"


def test_legacy_xj_project_adapter_names_alias_canonical_names():
    assert XJProjectAdapterConfig is CatKVOpsAdapterConfig
    assert XJProjectBlendAdapter is CatKVOpsBlendAdapter
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py::test_canonical_adapter_names_are_available tests/v1/test_taotie_xj_project_adapter.py::test_legacy_xj_project_adapter_names_alias_canonical_names -q
```

Expected: fail with `ModuleNotFoundError: No module named 'lmcache.v1.compute.blend.catkv_ops_adapter'`.

- [ ] **Step 3: Commit the failing tests**

```bash
git add catkv-vllm/tests/v1/test_taotie_xj_project_adapter.py
git commit -m "test: cover catkv ops adapter canonical names"
```

---

### Task 2: Canonical Adapter Module

**Files:**
- Create: `lmcache/v1/compute/blend/catkv_ops_adapter.py`
- Modify: `lmcache/v1/compute/blend/xj_project_adapter.py`
- Test: `tests/v1/test_taotie_xj_project_adapter.py`

- [ ] **Step 1: Move implementation to canonical module**

Move the current adapter implementation into `catkv_ops_adapter.py`, rename:

```python
@dataclass(frozen=True)
class CatKVOpsAdapterConfig:
    ...


class CatKVOpsBlendAdapter:
    def __init__(self, config: CatKVOpsAdapterConfig):
        ...
```

Replace the eager `import catKV_ops` with lazy import inside `__init__`:

```python
try:
    self._ops = importlib.import_module("catkv_ops")
except ImportError as exc:
    self.unavailable_reason = f"import-error: {exc}"
    return
```

Use canonical queue/env fallbacks:

```python
self._queue_log_path = (
    config.queue_log_path
    if config.queue_log_path is not None
    else _env_first("LMCACHE_CATKV_OPS_QUEUE_LOG", "LMCACHE_XJ_QUEUE_LOG")
)
```

Add helper functions:

```python
def _env_first(primary: str, legacy: str, default=None):
    value = os.environ.get(primary)
    if value is not None:
        return value
    value = os.environ.get(legacy)
    return default if value is None else value


def _env_flag_first(primary: str, legacy: str, default: bool) -> bool:
    value = _env_first(primary, legacy)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}
```

Use these for worker, prefetch worker, queue log stdout, and queue interval.

- [ ] **Step 2: Add compatibility shim**

Replace `xj_project_adapter.py` contents with:

```python
from lmcache.v1.compute.blend.catkv_ops_adapter import (
    CatKVOpsAdapterConfig,
    CatKVOpsBlendAdapter,
)

XJProjectAdapterConfig = CatKVOpsAdapterConfig
XJProjectBlendAdapter = CatKVOpsBlendAdapter

__all__ = [
    "CatKVOpsAdapterConfig",
    "CatKVOpsBlendAdapter",
    "XJProjectAdapterConfig",
    "XJProjectBlendAdapter",
]
```

- [ ] **Step 3: Run adapter import tests**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py::test_canonical_adapter_names_are_available tests/v1/test_taotie_xj_project_adapter.py::test_legacy_xj_project_adapter_names_alias_canonical_names -q
```

Expected: pass.

- [ ] **Step 4: Commit canonical adapter module**

```bash
git add catkv-vllm/lmcache/v1/compute/blend/catkv_ops_adapter.py catkv-vllm/lmcache/v1/compute/blend/xj_project_adapter.py
git commit -m "refactor: rename xj adapter to catkv ops adapter"
```

---

### Task 3: Config and Env Precedence Tests

**Files:**
- Modify: `tests/v1/test_taotie_xj_project_adapter.py`
- Modify: `tests/v1/test_taotie_blend_empty_s3.py`

- [ ] **Step 1: Write failing adapter env precedence test**

Add to `test_taotie_xj_project_adapter.py`:

```python
def test_catkv_ops_env_overrides_legacy_xj_env(monkeypatch):
    class _FakeStore:
        def __init__(self, pin_memory=True):
            self.enable_calls = []

        def enable_remote_upload(self, *args, **kwargs):
            self.enable_calls.append((args, kwargs))

    class _FakeSchedule:
        def __init__(self, config_path, workers):
            self.workers = workers

    class _FakeOps:
        CPUMemoryStore = _FakeStore
        S3Schedule = _FakeSchedule

    monkeypatch.setattr(importlib, "import_module", lambda name: _FakeOps)
    monkeypatch.setenv("LMCACHE_XJ_NUM_WORKERS", "12")
    monkeypatch.setenv("LMCACHE_CATKV_OPS_NUM_WORKERS", "34")
    monkeypatch.setenv("LMCACHE_XJ_PREFETCH_WORKERS", "13")
    monkeypatch.setenv("LMCACHE_CATKV_OPS_PREFETCH_WORKERS", "35")

    adapter = CatKVOpsBlendAdapter(
        CatKVOpsAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    assert adapter._store.enable_calls[0][1]["num_workers"] == 34
    assert adapter._scheduler.workers == 35
```

- [ ] **Step 2: Write failing KV manager config precedence test**

Add a focused test that instantiates `KVCacheManager.__new__` where possible or patches adapter construction to assert `catkv_ops` config wins over `xj_project` config.

- [ ] **Step 3: Run new tests to verify failure**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py::test_catkv_ops_env_overrides_legacy_xj_env -q
```

Expected: fail because only `LMCACHE_XJ_*` is currently read.

- [ ] **Step 4: Commit failing tests**

```bash
git add catkv-vllm/tests/v1/test_taotie_xj_project_adapter.py catkv-vllm/tests/v1/test_taotie_blend_empty_s3.py
git commit -m "test: cover catkv ops config precedence"
```

---

### Task 4: Rewire KV Manager and Context Manager

**Files:**
- Modify: `lmcache/v1/compute/blend/kvmanager.py`
- Modify: `lmcache/v1/compute/blend/context_manager.py`
- Modify: `lmcache/v1/compute/blend/taotie_blender.py`

- [ ] **Step 1: Update imports and constructor naming**

Change `kvmanager.py` import to:

```python
from lmcache.v1.compute.blend.catkv_ops_adapter import (
    CatKVOpsAdapterConfig,
    CatKVOpsBlendAdapter,
)
```

Add helper:

```python
def _env_first(primary: str, legacy: str, default=None):
    value = os.environ.get(primary)
    if value is not None:
        return value
    value = os.environ.get(legacy)
    return default if value is None else value
```

Rename internal storage to `self.catkv_ops_config` while keeping
`self.xj_project_config` as an alias.

- [ ] **Step 2: Prefer canonical config and env names**

Use canonical config and env names:

```python
legacy_enabled = _config_flag(
    self.catkv_ops_config,
    "enabled",
    _env_flag("LMCACHE_USE_CATKV_OPS", _env_flag("LMCACHE_USE_XJ_PROJECT", False)),
)
catkv_store_enabled = _config_flag(
    self.catkv_ops_config,
    "store_enabled",
    _env_flag("LMCACHE_CATKV_OPS_STORE", _env_flag("LMCACHE_XJ_STORE", legacy_enabled)),
)
```

Apply the same pattern for prefetch, S3 config, queue log, stdout, interval,
num workers, and prefetch workers.

- [ ] **Step 3: Add canonical support methods and legacy proxies**

Add:

```python
def _supports_catkv_ops_pipeline(self) -> bool:
    adapter = getattr(self, "_catkv_ops_adapter", None)
    return adapter is not None and adapter.supports(self.compress_type)

def _supports_xj_project_pipeline(self) -> bool:
    return self._supports_catkv_ops_pipeline()
```

Repeat the same pattern for store and prefetch.

- [ ] **Step 4: Update context/blender config selection**

Add a helper in `taotie_blender.py`:

```python
def _get_catkv_ops_config(extra_config):
    return extra_config.get("catkv_ops") or extra_config.get("xj_project") or {}
```

Pass this config into `ContextManager`.

In `ContextManager`, prefer `catkv_ops_config` naming and make
`_uses_catkv_ops_prefetch()` check `_supports_catkv_ops_prefetch()` first, then
legacy `_supports_xj_project_prefetch()`.

- [ ] **Step 5: Run focused tests**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py tests/v1/test_taotie_blend_empty_s3.py -q
```

Expected: pass, except failures caused by missing local optional CUDA deps should be captured exactly.

- [ ] **Step 6: Commit manager rewiring**

```bash
git add catkv-vllm/lmcache/v1/compute/blend/kvmanager.py catkv-vllm/lmcache/v1/compute/blend/context_manager.py catkv-vllm/lmcache/v1/compute/blend/taotie_blender.py
git commit -m "refactor: prefer catkv ops blend config"
```

---

### Task 5: Fix CatKV Package Imports

**Files:**
- Modify: `lmcache/v1/compute/blend/compress/our.py`
- Test: `tests/v1/test_taotie_xj_project_adapter.py`

- [ ] **Step 1: Write failing package-name test**

Add a test asserting adapter import asks for `catkv_ops`:

```python
def test_adapter_imports_canonical_catkv_ops_package(monkeypatch):
    imported = []

    class _FakeOps:
        class CPUMemoryStore:
            def __init__(self, pin_memory=True):
                pass

            def enable_remote_upload(self, *args, **kwargs):
                pass

        class S3Schedule:
            def __init__(self, config_path, workers):
                pass

    def _fake_import_module(name: str):
        imported.append(name)
        assert name == "catkv_ops"
        return _FakeOps

    monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    adapter = CatKVOpsBlendAdapter(
        CatKVOpsAdapterConfig(enabled=True, config_path="/tmp/s3.ini")
    )

    assert adapter.available is True
    assert imported == ["catkv_ops"]
```

- [ ] **Step 2: Run test to verify failure before implementation**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py::test_adapter_imports_canonical_catkv_ops_package -q
```

Expected: fail until `catkv_ops_adapter.py` uses `importlib.import_module("catkv_ops")`.

- [ ] **Step 3: Update OURS import**

Change in `compress/our.py`:

```python
from catkv_ops import fused_dequant_u_transposed, fused_dequant_v_residual
```

- [ ] **Step 4: Run adapter tests**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py -q
```

Expected: pass.

- [ ] **Step 5: Commit package import fix**

```bash
git add catkv-vllm/lmcache/v1/compute/blend/compress/our.py catkv-vllm/tests/v1/test_taotie_xj_project_adapter.py
git commit -m "fix: import canonical catkv ops package"
```

---

### Task 6: Experiment Runner Compatibility

**Files:**
- Modify: `exp/request_rate/run_cpu_compress_dedup_experiment.py`
- Modify: `tests/v1/test_cpu_compress_dedup_runner.py`

- [ ] **Step 1: Write failing runner tests**

Update `test_build_lmcache_extra_config_carries_xj_queue_settings` to assert:

```python
catkv_config = config["catkv_ops"]
assert catkv_config["enabled"] is True
assert catkv_config["compress_type"] == "OURS"
assert config["xj_project"] == catkv_config
```

Update env test to assert canonical env names:

```python
assert env["LMCACHE_USE_CATKV_OPS"] == "1"
assert env["LMCACHE_CATKV_OPS_STORE"] == "1"
assert env["LMCACHE_CATKV_OPS_PREFETCH"] == "1"
assert env["LMCACHE_CATKV_OPS_S3_CONFIG"] == "/home/xujie/xj_project/config/s3.ini"
assert env["LMCACHE_CATKV_OPS_NUM_WORKERS"] == "32"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_cpu_compress_dedup_runner.py -q
```

Expected: fail because runner only emits `xj_project` and `LMCACHE_XJ_*`.

- [ ] **Step 3: Update runner**

Emit both config keys:

```python
catkv_ops_config = {
    "enabled": True,
    ...
}
return json.dumps({
    "catkv_ops": catkv_ops_config,
    "xj_project": catkv_ops_config,
})
```

Emit both canonical and legacy env names, keeping current values.

- [ ] **Step 4: Run tests**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_cpu_compress_dedup_runner.py -q
```

Expected: pass.

- [ ] **Step 5: Commit runner compatibility**

```bash
git add catkv-vllm/exp/request_rate/run_cpu_compress_dedup_experiment.py catkv-vllm/tests/v1/test_cpu_compress_dedup_runner.py
git commit -m "refactor: emit catkv ops experiment config"
```

---

### Task 7: Final Verification

**Files:**
- No new production edits expected.

- [ ] **Step 1: Run focused adapter and blend tests**

Run:

```bash
cd /home/xujie/catkv-release/catkv-vllm
python -m pytest tests/v1/test_taotie_xj_project_adapter.py tests/v1/test_taotie_blend_empty_s3.py tests/v1/test_cpu_compress_dedup_runner.py -q
```

Expected: pass, or report exact optional dependency/CUDA import failures.

- [ ] **Step 2: Search for stale incorrect import**

Run:

```bash
cd /home/xujie/catkv-release
rg -n "catKV_ops|from catKV_ops|import catKV_ops" catkv-vllm
```

Expected: no matches.

- [ ] **Step 3: Search for remaining legacy names and classify them**

Run:

```bash
cd /home/xujie/catkv-release
rg -n "xj_project|XJProject|LMCACHE_XJ|_supports_xj_project" catkv-vllm/lmcache/v1/compute/blend catkv-vllm/tests/v1 catkv-vllm/exp/request_rate
```

Expected: remaining matches are compatibility aliases, legacy env fallbacks, tests for fallback behavior, or unrelated historical/docs naming.

- [ ] **Step 4: Review diff**

Run:

```bash
cd /home/xujie/catkv-release
git status --short
git diff --stat
git diff -- catkv-vllm/lmcache/v1/compute/blend catkv-vllm/tests/v1 catkv-vllm/exp/request_rate
```

Expected: diff matches the spec scope.
