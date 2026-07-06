import uuid

import torch

import catkv_ops


S3_CONFIG = "/home/xujie/catkv-release/csrc/config/s3.ini"


def _expected_uuid_tensor(uuid: str) -> torch.Tensor:
    h = 1469598103934665603
    for byte in uuid.encode("utf-8"):
        h ^= byte
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    high = (h >> 32) & 0xFFFFFFFF
    low = h & 0xFFFFFFFF

    def to_int32(value):
        return value - (1 << 32) if value >= (1 << 31) else value

    return torch.tensor([to_int32(high), to_int32(low)], dtype=torch.int32)


def _kv_tensor(seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, 64, 2, 32), generator=generator, dtype=torch.bfloat16)


def test_compress_multi_uses_group_uuid_as_payload_uuid():
    compressor = catkv_ops.CatKVCompressor(ratio=0.5, dtype=torch.bfloat16)
    key = torch.randn(1, 4, 1, 4, dtype=torch.float32)
    value = torch.randn(1, 4, 1, 4, dtype=torch.float32)

    payloads = compressor.compress_multi(
        [key, value],
        [(0, 2), (2, 4)],
        ["request-uuid", "request-uuid"],
    )

    expected = _expected_uuid_tensor("request-uuid")
    assert len(payloads) == 2
    assert torch.equal(payloads[0]["uuid"], expected)
    assert torch.equal(payloads[1]["uuid"], expected)
    assert payloads[0]["uuid"].nbytes == 8
    assert payloads[1]["uuid"].nbytes == 8


def test_compress_multi_shares_largest_rank_key_sv_for_group_uuid():
    compressor = catkv_ops.CatKVCompressor(ratio=0.5, dtype=torch.bfloat16)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(123)
    key = torch.randn((1, 8, 2, 8), generator=generator, dtype=torch.float32)
    value = torch.randn((1, 8, 2, 8), generator=generator, dtype=torch.float32)

    payloads = compressor.compress_multi(
        [key, value],
        [(0, 2), (2, 8)],
        ["request-uuid", "request-uuid"],
    )

    key_ranks = [payload["key_sv_quantized"].shape[-2] for payload in payloads]
    value_ranks = [payload["value_sv_quantized"].shape[-2] for payload in payloads]
    assert value_ranks[0] < value_ranks[1]
    assert key_ranks == [value_ranks[1], value_ranks[1]]


def test_shared_key_sv_path_uses_uuid_tensor_and_layer():
    uuid = torch.tensor([11, 22], dtype=torch.int32)

    path = catkv_ops.shared_key_sv_path(
        uuid,
        3,
        "vllm/kvcache/OURS/chunk-a-layer_3-device_cuda:0.bin",
    )

    assert path == "vllm/kvcache/OURS/11_22_layer_3_key_sv"


def test_remote_group_save_writes_split_payload_with_shared_sv():
    prefix = f"vllm/kvcache/OURS/pytest_shared_sv_remote/{uuid.uuid4().hex}"
    group_uuid = f"{prefix}/request-a"
    expected_uuid = _expected_uuid_tensor(group_uuid)
    store = catkv_ops.CPUMemoryStore(True, 1)
    store.enable_remote_upload(S3_CONFIG, 0.5, torch.bfloat16, 2, 0, False)
    manager = catkv_ops.S3Manager(S3_CONFIG)

    paths = [
        f"{prefix}/chunk_{idx}_layer_3-device_cuda:0.bin"
        for idx in range(2)
    ]
    for idx, path in enumerate(paths):
        store.offload(
            path,
            {"key": _kv_tensor(1000 + idx), "value": _kv_tensor(2000 + idx)},
            group_uuid,
        )
    for path in paths:
        store.wait_remote_ready(path, 120.0)

    shared_path = catkv_ops.shared_key_sv_path(expected_uuid, 3, paths[0])
    assert manager.exists(shared_path)
    assert manager.exists(paths[0] + "_other")
    assert manager.exists(paths[1] + "_other")
    assert not manager.exists(paths[0] + "_key_sv")
    assert not manager.exists(paths[1] + "_key_sv")
    assert torch.equal(manager.load(paths[0] + "_other")["uuid"], expected_uuid)
    assert torch.equal(manager.load(paths[1] + "_other")["uuid"], expected_uuid)
    assert torch.equal(manager.load(shared_path)["uuid"], expected_uuid)


def test_remote_group_queue_is_layer_aware_but_keeps_payload_uuid():
    prefix = f"vllm/kvcache/OURS/pytest_layer_aware_group/{uuid.uuid4().hex}"
    group_uuid = f"{prefix}/request-a"
    expected_uuid = _expected_uuid_tensor(group_uuid)
    store = catkv_ops.CPUMemoryStore(True, 1)
    store.enable_remote_upload(S3_CONFIG, 0.5, torch.bfloat16, 2, 0, False)
    manager = catkv_ops.S3Manager(S3_CONFIG)

    paths = []
    for chunk in range(2):
        for layer in (2, 3):
            path = f"{prefix}/chunk_{chunk}_layer_{layer}-device_cuda:0.bin"
            paths.append(path)
            store.offload(
                path,
                {
                    "key": _kv_tensor(3000 + layer * 10 + chunk),
                    "value": _kv_tensor(4000 + layer * 10 + chunk),
                },
                group_uuid,
            )
    for path in paths:
        store.wait_remote_ready(path, 120.0)

    stats = store.remote_queue_stats()
    assert stats["max_group_size"] == 2
    assert stats["grouped_dispatches"] == 2

    for chunk in range(2):
        layer2_path = f"{prefix}/chunk_{chunk}_layer_2-device_cuda:0.bin"
        layer3_path = f"{prefix}/chunk_{chunk}_layer_3-device_cuda:0.bin"
        assert manager.exists(layer2_path + "_key_sv")
        assert manager.exists(layer2_path + "_other")
        assert not manager.exists(layer3_path + "_key_sv")
        assert manager.exists(layer3_path + "_other")
        assert torch.equal(manager.load(layer2_path + "_other")["uuid"], expected_uuid)
        assert torch.equal(manager.load(layer3_path + "_other")["uuid"], expected_uuid)

    shared_layer3 = catkv_ops.shared_key_sv_path(
        expected_uuid,
        3,
        f"{prefix}/chunk_0_layer_3-device_cuda:0.bin",
    )
    assert manager.exists(shared_layer3)
