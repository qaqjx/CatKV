# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Tuple

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.storage_backend.naive_serde.naive_serde import (
    NaiveDeserializer,
    NaiveSerializer,
)
from lmcache.v1.storage_backend.naive_serde.serde import Deserializer, Serializer


def CreateSerde(
    serde_type: str,
    metadata: LMCacheEngineMetadata,
    config: LMCacheEngineConfig,
) -> Tuple[Serializer, Deserializer]:
    s: Optional[Serializer] = None
    d: Optional[Deserializer] = None

    if serde_type == "naive":
        s, d = NaiveSerializer(), NaiveDeserializer()
    else:
        raise ValueError(f"Invalid type: {serde_type}")

    return s, d


__all__ = [
    "Serializer",
    "Deserializer",
    "CreateSerde",
]
