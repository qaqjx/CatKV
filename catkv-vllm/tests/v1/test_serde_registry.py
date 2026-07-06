from pathlib import Path

import pytest


def test_removed_v1_compression_serde_files_are_not_present():
    serde_dir = Path("lmcache/v1/storage_backend/naive_serde")

    assert not (serde_dir / "cachegen_basics.py").exists()
    assert not (serde_dir / "cachegen_decoder.py").exists()
    assert not (serde_dir / "cachegen_encoder.py").exists()
    assert not (serde_dir / "kivi_serde.py").exists()


def test_v1_serde_rejects_removed_compression_serializers():
    from lmcache.v1.storage_backend.naive_serde import CreateSerde

    for serde_type in ("cachegen", "kivi"):
        with pytest.raises(ValueError, match="Invalid type"):
            CreateSerde(serde_type, metadata=None, config=None)


def test_removed_legacy_cachegen_serde_files_are_not_present():
    serde_dir = Path("lmcache/storage_backend/serde")

    assert not (serde_dir / "cachegen_basics.py").exists()
    assert not (serde_dir / "cachegen_decoder.py").exists()
    assert not (serde_dir / "cachegen_encoder.py").exists()


def test_legacy_serde_rejects_cachegen():
    from lmcache.storage_backend.serde import CreateSerde

    with pytest.raises(ValueError, match="Invalid serde type"):
        CreateSerde("cachegen", config=None, metadata=None)
