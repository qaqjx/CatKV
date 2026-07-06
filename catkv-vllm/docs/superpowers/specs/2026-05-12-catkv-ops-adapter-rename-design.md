# catkv_ops Adapter Rename Design

## Context

`catkv-vllm` still exposes the CatKV C++/CUDA integration through names that
refer to `xj_project`: `xj_project_adapter.py`, `XJProjectBlendAdapter`,
`extra_config["xj_project"]`, and `LMCACHE_XJ_*` environment variables. The
actual released package in this repository is `catkv_ops`, with Python exports
such as `CPUMemoryStore`, `S3Schedule`, `fused_dequant_u_transposed`, and
`fused_dequant_v_residual`.

The current adapter also imports `catKV_ops`, which does not match the package
name documented and built by `csrc/setup.py`. The rename should make the public
integration describe what it actually uses, while preserving existing
experiment and deployment configurations.

## Approach

Use a compatibility rename. The primary adapter file, classes, config key, and
environment variables move to `catkv_ops` naming. Existing `xj_project` names
remain as compatibility aliases so current scripts and tests continue to work.

This keeps the runtime behavior unchanged: `KVCacheManager` still delegates
OURS compression store and prefetch to the adapter when enabled, and the
adapter still presents merged OURS payloads to the existing decompressor.

## Components

- Rename the primary adapter module from
  `lmcache/v1/compute/blend/xj_project_adapter.py` to
  `lmcache/v1/compute/blend/catkv_ops_adapter.py`.
- Rename the primary adapter types to `CatKVOpsAdapterConfig` and
  `CatKVOpsBlendAdapter`.
- Leave a thin `xj_project_adapter.py` compatibility module that re-exports the
  new names and provides aliases for `XJProjectAdapterConfig` and
  `XJProjectBlendAdapter`.
- Update `KVCacheManager`, `ContextManager`, and blender construction to prefer
  `extra_config["catkv_ops"]`, falling back to `extra_config["xj_project"]`.
- Update support helpers to prefer names such as
  `_supports_catkv_ops_store()` and `_supports_catkv_ops_prefetch()`, while
  keeping the old `_supports_xj_project_*()` methods as proxy aliases.
- Update imports from `catKV_ops` to the actual `catkv_ops` package in both the
  adapter and OURS decompression helper.

## Configuration

New configuration is accepted under:

```json
{
  "catkv_ops": {
    "enabled": true,
    "store_enabled": true,
    "prefetch_enabled": true,
    "compress_type": "OURS",
    "s3_config": "/path/to/s3.ini"
  }
}
```

The old `xj_project` key remains supported with lower precedence. When both are
present, `catkv_ops` wins.

New environment variables use `LMCACHE_CATKV_OPS_*` names. Existing
`LMCACHE_XJ_*` variables remain supported as fallbacks. For example,
`LMCACHE_CATKV_OPS_S3_CONFIG` is preferred over `LMCACHE_XJ_S3_CONFIG`.

## Data Flow

Store flow remains:

1. `ContextManager` calls `KVCacheManager.offload_layer_data()`.
2. `KVCacheManager` checks whether the CatKV ops adapter supports OURS store.
3. `CatKVOpsBlendAdapter.offload()` forwards key/value tensors to
   `catkv_ops.CPUMemoryStore.offload()`.

Prefetch flow remains:

1. `ContextManager` builds logical cache paths for active reuse chunks.
2. `KVCacheManager.retrieve_keys()` delegates to
   `CatKVOpsBlendAdapter.prefetch_remote()` when CatKV ops prefetch is enabled.
3. The adapter loads any local CPU payloads and submits missing split remote
   paths to `catkv_ops.S3Schedule`.
4. `get_prefetch_result()` merges split OURS payloads back into one dictionary
   per logical cache path for the existing decompressor.

## Error Handling

If CatKV ops support is disabled, the adapter reports `disabled` as before. If
`catkv_ops` cannot be imported, the adapter records an import-error reason and
the KV manager falls back to the non-adapter path. Empty or malformed split
remote payloads are returned as empty dictionaries so existing reuse pruning
treats them as misses.

## Testing

Focused tests should cover:

- New adapter module and class imports.
- Legacy adapter module and class aliases.
- Importing the real `catkv_ops` package name rather than `catKV_ops`.
- `extra_config["catkv_ops"]` taking precedence over `extra_config["xj_project"]`.
- New `LMCACHE_CATKV_OPS_*` environment variables taking precedence over old
  `LMCACHE_XJ_*` variables.
- Existing store, prefetch, split-path expansion, and payload merge behavior.

The main verification commands are:

```bash
python -m pytest tests/v1/test_taotie_xj_project_adapter.py -q
python -m pytest tests/v1/test_taotie_blend_empty_s3.py -q
```
