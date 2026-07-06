from __future__ import annotations

from pathlib import PurePosixPath
from typing import Any

import torch


def _to_signed_int32(value: int) -> int:
    value &= 0xFFFFFFFF
    return value - (1 << 32) if value >= (1 << 31) else value


def uuid_to_tensor(uuid: str) -> torch.Tensor:
    hash_value = 1469598103934665603
    for byte in str(uuid).encode("utf-8"):
        hash_value ^= byte
        hash_value = (hash_value * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    high = _to_signed_int32((hash_value >> 32) & 0xFFFFFFFF)
    low = _to_signed_int32(hash_value & 0xFFFFFFFF)
    return torch.tensor([high, low], dtype=torch.int32)


def normalize_uuid(uuid: Any) -> torch.Tensor:
    if isinstance(uuid, torch.Tensor):
        return uuid.detach().cpu().to(torch.int32).contiguous().view(-1)
    return uuid_to_tensor(str(uuid))


def uuid_tensor_to_path_id(uuid: Any) -> str:
    if isinstance(uuid, str):
        return uuid
    if not isinstance(uuid, torch.Tensor):
        raise TypeError(f"uuid must be a string or tensor, got {type(uuid)!r}")
    values = uuid.detach().cpu().to(torch.int32).flatten().tolist()
    return "_".join(str(int(value)) for value in values)


def get_shared_key_sv_filename(uuid: Any, layer_idx: int, base_key: str) -> str:
    path = PurePosixPath(base_key)
    filename = f"{uuid_tensor_to_path_id(uuid)}_layer_{layer_idx}_key_sv"
    if str(path.parent) == ".":
        return filename
    return str(path.parent / filename)


def default_group_uuid_from_key(key: str) -> str:
    basename = PurePosixPath(key).name
    marker = "-layer_"
    if marker not in basename:
        return basename
    return basename.split(marker, maxsplit=1)[0]
