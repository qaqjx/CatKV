from pathlib import Path

import pytest

from catkv.strategy.Cacheblend import CacheBlend
from catkv.strategy.Epic import EPIC
from catkv.strategy.Kvshare import KVShare
from catkv.strategy.abstract_blend import ProcessType
from catkv.strategy.utils import BlenderFactory
from catkv.utils.config import ConfigManager


def test_strategy_package_only_keeps_public_blenders():
    strategy_dir = Path(__file__).parents[1] / "catkv" / "strategy"
    public_modules = {path.name for path in strategy_dir.glob("*.py")}

    assert public_modules == {
        "__init__.py",
        "abstract_blend.py",
        "utils.py",
        "Cacheblend.py",
        "Epic.py",
        "Kvshare.py",
    }


def test_process_types_are_limited_to_public_strategies_and_default():
    assert {member.name for member in ProcessType} == {
        "DEFAULT",
        "CACHEBLEND",
        "EPIC",
        "KVSHARE",
    }


def test_select_strategy_registry_is_limited_to_public_strategies():
    manager = ConfigManager()

    assert set(manager._strategy_registries["select"]) == {
        "DEFAULT",
        "CACHEBLEND",
        "EPIC",
        "KVSHARE",
    }


@pytest.mark.parametrize(
    ("process_type", "expected_cls"),
    [
        (ProcessType.CACHEBLEND, CacheBlend),
        (ProcessType.EPIC, EPIC),
        (ProcessType.KVSHARE, KVShare),
    ],
)
def test_blender_factory_keeps_public_strategies(process_type, expected_cls):
    blend_meta = {
        "select_strategy": process_type,
        "select_config": {"recompute_ratio": 0.15, "recompute_num": 16},
        "device": "cpu",
    }

    assert isinstance(BlenderFactory().get_blender(0, blend_meta), expected_cls)


def test_blender_factory_rejects_removed_strategy():
    blend_meta = {
        "select_strategy": "REMOVED_STRATEGY",
        "select_config": {},
        "device": "cpu",
    }

    with pytest.raises(ValueError, match="Invalid blend type"):
        BlenderFactory().get_blender(0, blend_meta)
