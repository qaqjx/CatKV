from pathlib import Path

import pytest

from catkv.kv_manager.disk.safe_tensor import IOMode, SafeTensorDiskManager, create_io_manager


def test_disk_package_only_keeps_public_s3_and_safetensor_modules():
    disk_dir = Path(__file__).parents[1] / "catkv" / "kv_manager" / "disk"
    public_modules = {path.name for path in disk_dir.glob("*.py")}

    assert public_modules == {"s3_disk.py", "safe_tensor.py"}


def test_disk_io_modes_are_limited_to_s3_and_safetensor():
    assert {mode.name for mode in IOMode} == {"SAFETENSOR", "S3"}
    assert {mode.value for mode in IOMode} == {"safetensor", "s3"}


def test_disk_factory_rejects_removed_modes():
    with pytest.raises(ValueError, match="Unsupported IO mode"):
        create_io_manager("bin")


def test_disk_factory_keeps_safetensor_interface():
    assert isinstance(create_io_manager(IOMode.SAFETENSOR), SafeTensorDiskManager)
    assert isinstance(create_io_manager("safetensor"), SafeTensorDiskManager)
