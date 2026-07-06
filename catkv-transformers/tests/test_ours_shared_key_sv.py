import importlib
import os
import sys
import types

import torch

from catkv.kv_manager.compress.abstract_compress import CompressType
from catkv.kv_manager.kv_manager import KVCacheManager
from catkv.kv_manager.utils import get_kvcache_filename, uuid_to_tensor


def _uuid_path_id(uuid):
    if isinstance(uuid, str):
        return uuid
    values = uuid.detach().cpu().to(torch.int32).flatten().tolist()
    return "_".join(str(int(value)) for value in values)


def _shared_key_sv_filename(uuid, layer_idx, base_key):
    directory = os.path.dirname(base_key)
    filename = f"{_uuid_path_id(uuid)}_layer_{layer_idx}_key_sv"
    return os.path.join(directory, filename) if directory else filename


class _RecordingDB:
    def __init__(self):
        self.records = []

    def store_data(self, key, payload):
        self.records.append((key, payload))
        return True


class _FakeCompressor:
    def __init__(self, uuid):
        self.uuid = uuid
        self.calls = []

    def compress_multi(self, kv, indices=None, uuid=None):
        self.calls.append({"kv": kv, "indices": indices, "uuid": uuid})
        payload_uuid = uuid if uuid is not None else self.uuid
        payloads = []
        for idx, _ in enumerate(indices):
            payloads.append(
                {
                    "key_sv_quantized": torch.tensor([idx + 1]),
                    "key_sv_meta": torch.tensor([idx + 2]),
                    "key_residual_sv": torch.tensor([idx + 3]),
                    "u_quantized": torch.tensor([idx + 4]),
                    "u_meta": torch.tensor([idx + 5]),
                    "value_sv_quantized": torch.tensor([idx + 6]),
                    "value_sv_meta": torch.tensor([idx + 7]),
                    "value_residual_sv": torch.tensor([idx + 8]),
                    "uuid": payload_uuid,
                }
            )
        return payloads


class _RankedFakeCompressor:
    def __init__(self, uuid, key_ranks):
        self.uuid = uuid
        self.key_ranks = key_ranks

    def compress_multi(self, kv, indices=None, uuid=None):
        payload_uuid = uuid if uuid is not None else self.uuid
        payloads = []
        for idx, _ in enumerate(indices):
            key_rank = self.key_ranks[idx]
            payloads.append(
                {
                    "key_sv_quantized": torch.ones(1, key_rank, 2) * (idx + 1),
                    "key_sv_meta": torch.ones(1, key_rank, 4) * (idx + 2),
                    "key_residual_sv": torch.ones(1, key_rank, 2) * (idx + 3),
                    "u_quantized": torch.tensor([idx + 4]),
                    "u_meta": torch.tensor([idx + 5]),
                    "value_sv_quantized": torch.tensor([idx + 6]),
                    "value_sv_meta": torch.tensor([idx + 7]),
                    "value_residual_sv": torch.tensor([idx + 8]),
                    "uuid": payload_uuid,
                }
            )
        return payloads


def _build_kv_manager(uuid):
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.compress_type = CompressType.OURS
    manager.db = _RecordingDB()
    manager.compressor = _FakeCompressor(uuid)
    manager.keys_set = set()
    return manager


def test_store_multi_data_keeps_per_chunk_key_sv_for_layer_two():
    uuid = torch.tensor([11, 22], dtype=torch.int32)
    manager = _build_kv_manager(uuid)

    manager.store_multi_data(
        ["chunk-a", "chunk-b"],
        indices=[[0, 2], [2, 4]],
        layer_idx=2,
        device="cpu",
        kv=("key", "value"),
    )

    paths = [path for path, _ in manager.db.records]
    assert paths == [
        get_kvcache_filename("chunk-a", layer_idx=2, device="cpu") + "_key_sv",
        get_kvcache_filename("chunk-a", layer_idx=2, device="cpu") + "_other",
        get_kvcache_filename("chunk-b", layer_idx=2, device="cpu") + "_key_sv",
        get_kvcache_filename("chunk-b", layer_idx=2, device="cpu") + "_other",
    ]


def test_store_multi_data_saves_one_shared_key_sv_after_layer_two():
    uuid = torch.tensor([11, 22], dtype=torch.int32)
    manager = _build_kv_manager(uuid)

    manager.store_multi_data(
        ["chunk-a", "chunk-b"],
        indices=[[0, 2], [2, 4]],
        layer_idx=3,
        device="cpu",
        kv=("key", "value"),
    )

    paths = [path for path, _ in manager.db.records]
    passed_uuid = manager.compressor.calls[0]["uuid"]
    assert passed_uuid is not None
    shared_path = _shared_key_sv_filename(
        passed_uuid,
        layer_idx=3,
        base_key=get_kvcache_filename("chunk-a", layer_idx=3, device="cpu"),
    )
    assert paths == [
        shared_path,
        get_kvcache_filename("chunk-a", layer_idx=3, device="cpu") + "_other",
        get_kvcache_filename("chunk-b", layer_idx=3, device="cpu") + "_other",
    ]
    assert manager.db.records[1][1]["uuid"].equal(passed_uuid)
    assert manager.db.records[2][1]["uuid"].equal(passed_uuid)


def test_store_multi_data_saves_largest_rank_shared_key_sv_after_layer_two():
    uuid = torch.tensor([11, 22], dtype=torch.int32)
    manager = _build_kv_manager(uuid)
    manager.compressor = _RankedFakeCompressor(uuid, key_ranks=[1, 3])

    manager.store_multi_data(
        ["chunk-a", "chunk-b"],
        indices=[[0, 2], [2, 4]],
        layer_idx=3,
        device="cpu",
        kv=("key", "value"),
    )

    shared_payload = manager.db.records[0][1]
    assert shared_payload["key_sv_quantized"].shape[-2] == 3
    assert shared_payload["key_sv_quantized"][0, 0, 0].item() == 2


class _FakeRetrieveKVManager:
    def __init__(self):
        self.compress_type = CompressType.OURS
        self.seen_keys = []

    def retrieve_keys(self, keys):
        keys = list(keys)
        self.seen_keys.append(keys)
        return f"task-{len(self.seen_keys)}"


def test_prefetch_chunk_kv_uses_shared_key_sv_paths_after_layer_two():
    utils_pkg = types.ModuleType("catkv.utils")
    utils_pkg.__path__ = []
    attention_mod = types.ModuleType("catkv.utils.attention")
    attention_mod.AdaptiveKVCacheAttention = type(
        "AdaptiveKVCacheAttention",
        (),
        {"__init__": lambda self, phase: None},
    )
    sys.modules.pop("catkv.kv_manager.context_manager", None)
    sys.modules["catkv.utils"] = utils_pkg
    sys.modules["catkv.utils.attention"] = attention_mod
    ContextManager = importlib.import_module(
        "catkv.kv_manager.context_manager"
    ).ContextManager

    manager = ContextManager.__new__(ContextManager)
    manager.layer_num = 5
    manager.batch_size = 1
    manager.num_heads_kv = 1
    manager.head_dim = 2
    manager.dtype = torch.float32
    manager.device = "cpu"
    manager.kv_manager = _FakeRetrieveKVManager()
    manager.compress_data = [None for _ in range(manager.layer_num)]
    manager.key_sv_filter_idx = [0]
    manager.key_sv_idx = [0, 0]
    manager.key_sv_uuid_path_ids = ["11_22"]

    manager.prefetch_chunk_kv(
        ["chunk-a", "chunk-b"],
        [[0, 2], [2, 4]],
        kv_len=4,
        layer_idx=3,
    )

    assert manager.kv_manager.seen_keys == [
        [
            _shared_key_sv_filename(
                "11_22",
                layer_idx=3,
                base_key=get_kvcache_filename("chunk-a", layer_idx=3, device="cpu"),
            )
        ],
        [
            get_kvcache_filename("chunk-a", layer_idx=3, device="cpu") + "_other",
            get_kvcache_filename("chunk-b", layer_idx=3, device="cpu") + "_other",
        ],
    ]


class _RecordingStoreMultiKVManager:
    def __init__(self):
        self.calls = []

    def store_multi_data(self, text_hash, indices=None, layer_idx=None, device=None, kv=None, group_uuid=None):
        self.calls.append(
            {
                "text_hash": text_hash,
                "indices": indices,
                "layer_idx": layer_idx,
                "device": device,
                "kv": kv,
                "group_uuid": group_uuid,
            }
        )


def test_save_all_kv_tensors_uses_one_group_uuid_for_all_layers():
    utils_pkg = types.ModuleType("catkv.utils")
    utils_pkg.__path__ = []
    attention_mod = types.ModuleType("catkv.utils.attention")
    attention_mod.AdaptiveKVCacheAttention = type(
        "AdaptiveKVCacheAttention",
        (),
        {"__init__": lambda self, phase: None},
    )
    sys.modules.pop("catkv.kv_manager.context_manager", None)
    sys.modules["catkv.utils"] = utils_pkg
    sys.modules["catkv.utils.attention"] = attention_mod
    ContextManager = importlib.import_module(
        "catkv.kv_manager.context_manager"
    ).ContextManager

    manager = ContextManager.__new__(ContextManager)
    manager.device = "cpu"
    manager.kv = ["layer-0-kv", "layer-1-kv", "layer-2-kv"]
    manager.kv_manager = _RecordingStoreMultiKVManager()

    manager.save_all_kv_tensors(["chunk-a", "chunk-b"], [[0, 2], [2, 4]])

    group_uuids = [call["group_uuid"] for call in manager.kv_manager.calls]
    assert len(group_uuids) == 3
    assert all(isinstance(group_uuid, torch.Tensor) for group_uuid in group_uuids)
    assert group_uuids[0].equal(group_uuids[1])
    assert group_uuids[0].equal(group_uuids[2])
    assert group_uuids[0].equal(uuid_to_tensor("chunk-a,chunk-b"))
