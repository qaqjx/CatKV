import importlib
import sys
import types

import pytest

from lmcache.v1.compute.blend.compress.abstract import CompressType


def _import_kvmanager(monkeypatch):
    fake_catkv_ops = types.SimpleNamespace(
        fused_dequant_u_transposed=lambda *args, **kwargs: None,
        fused_dequant_v_residual=lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "catkv_ops", fake_catkv_ops)

    return importlib.import_module("lmcache.v1.compute.blend.kvmanager")


def test_vllm_exposes_only_none_and_ours_compress_types():
    assert {member.name for member in CompressType} == {"NONE", "OURS"}
    assert {member.value for member in CompressType} == {"None", "ours"}


def test_vllm_default_compress_type_is_ours(monkeypatch):
    monkeypatch.delenv("LMCACHE_COMPRESS_TYPE", raising=False)
    kvmanager = _import_kvmanager(monkeypatch)

    assert kvmanager.get_compress_type_from_env() == CompressType.OURS


def test_vllm_rejects_removed_compress_type(monkeypatch):
    monkeypatch.setenv("LMCACHE_COMPRESS_TYPE", "REMOVED_METHOD")
    kvmanager = _import_kvmanager(monkeypatch)

    with pytest.raises(ValueError, match="Unsupported compression type"):
        kvmanager.get_compress_type_from_env()
