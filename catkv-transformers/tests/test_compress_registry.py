import pytest
from omegaconf import OmegaConf

from catkv.kv_manager.compress.abstract_compress import CompressType
from catkv.utils.config import ConfigManager


def test_transformers_exposes_only_none_and_ours_compress_types():
    assert {member.name for member in CompressType} == {"NONE", "OURS"}
    assert {member.value for member in CompressType} == {"None", "ours"}


def test_transformers_config_registry_only_names_public_compress_types():
    manager = ConfigManager()

    assert set(manager._strategy_registries["compress"]) == {"NONE", "OURS"}


def test_transformers_parse_compress_config_rejects_removed_methods():
    manager = ConfigManager()
    config = OmegaConf.create(
        {"compress_strategy": {"type": "REMOVED_METHOD"}}
    )

    with pytest.raises(ValueError, match="Unsupported compression type"):
        manager.parse_compress_config(config)
