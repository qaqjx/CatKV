from __future__ import annotations

import asyncio
import json

from exp.request_rate.catkv_memory_pressure import (
    annotate_requests_with_queue_memory,
    build_completion_payload,
    build_prompt_token_ids,
    merge_server_profile,
    load_prompts,
)


def test_load_prompts_repeats_dataset_when_requested(tmp_path):
    dataset_path = tmp_path / "mini.jsonl"
    samples = [
        {"id": "a", "contexts": ["ctx-a"], "question": "q-a"},
        {"id": "b", "contexts": ["ctx-b"], "question": "q-b"},
    ]
    with dataset_path.open("w", encoding="utf-8") as dataset:
        for sample in samples:
            dataset.write(json.dumps(sample) + "\n")

    prompts = load_prompts(str(dataset_path), limit=5, sep=" # # ", repeat_dataset=True)

    assert len(prompts) == 5
    assert [prompt["id"] for prompt in prompts] == ["a", "b", "a", "b", "a"]


def test_load_prompts_repeats_only_limited_source_subset(tmp_path):
    dataset_path = tmp_path / "mini.jsonl"
    samples = [
        {"id": "a", "contexts": ["ctx-a"], "question": "q-a"},
        {"id": "b", "contexts": ["ctx-b"], "question": "q-b"},
        {"id": "c", "contexts": ["ctx-c"], "question": "q-c"},
    ]
    with dataset_path.open("w", encoding="utf-8") as dataset:
        for sample in samples:
            dataset.write(json.dumps(sample) + "\n")

    prompts = load_prompts(
        str(dataset_path),
        limit=5,
        sep=" # # ",
        repeat_dataset=True,
        source_limit=2,
    )

    assert [prompt["id"] for prompt in prompts] == ["a", "b", "a", "b", "a"]


def test_load_prompts_honors_source_offset(tmp_path):
    dataset_path = tmp_path / "mini.jsonl"
    samples = [
        {"id": "a", "contexts": ["ctx-a"], "question": "q-a"},
        {"id": "b", "contexts": ["ctx-b"], "question": "q-b"},
        {"id": "c", "contexts": ["ctx-c"], "question": "q-c"},
    ]
    with dataset_path.open("w", encoding="utf-8") as dataset:
        for sample in samples:
            dataset.write(json.dumps(sample) + "\n")

    prompts = load_prompts(
        str(dataset_path),
        limit=2,
        sep=" # # ",
        source_limit=2,
        source_offset=1,
    )

    assert [prompt["id"] for prompt in prompts] == ["b", "c"]


def test_annotate_requests_with_queue_memory_uses_latest_prior_snapshot():
    results = [
        {"id": "r1", "sent_at_wall": 100.4},
        {"id": "r2", "sent_at_wall": 101.6},
    ]
    queue_records = [
        {"ts": 100.0, "remote_queue_bytes": 1024, "remote_pending_count": 1},
        {"ts": 101.0, "remote_queue_bytes": 2048, "remote_pending_count": 3},
        {"ts": 102.0, "remote_queue_bytes": 4096, "remote_pending_count": 5},
    ]

    annotate_requests_with_queue_memory(results, queue_records)

    assert results[0]["arrival_queue_bytes"] == 1024
    assert results[0]["arrival_pending_count"] == 1
    assert results[1]["arrival_queue_bytes"] == 2048
    assert results[1]["arrival_pending_count"] == 3


def test_build_prompt_token_ids_preserves_separator_boundaries():
    class _FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            mapping = {
                " # # ": [90, 91, 92],
                "ctx-a": [1, 2],
                "ctx-b": [3, 4],
                "q": [5],
            }
            return list(mapping[text])

    tokenizer = _FakeTokenizer()

    tokens = build_prompt_token_ids(
        ["ctx-a", "ctx-b"],
        "q",
        tokenizer=tokenizer,
        sep_tokens=[90, 91, 92],
    )

    assert tokens == [90, 91, 92, 1, 2, 90, 91, 92, 3, 4, 90, 91, 92, 5]


def test_build_completion_payload_uses_token_ids_via_prompt_field():
    sample = {
        "id": "x",
        "prompt": "raw prompt should not be used",
        "prompt_token_ids": [1, 2, 3],
        "num_contexts": 2,
    }

    payload = build_completion_payload(
        model="mistralai/Mistral-7B-Instruct-v0.3",
        sample=sample,
        max_tokens=8,
    )

    assert payload["prompt"] == [1, 2, 3]
    assert payload["max_tokens"] == 8


def test_send_one_includes_generated_text():
    from exp.request_rate.catkv_memory_pressure import send_one

    class _FakeResponse:
        status = 200

        def __init__(self):
            self._chunks = iter(
                [
                    b'data: {"choices":[{"text":"Hello"}]}\n',
                    b'data: {"choices":[{"text":" World"}]}\n',
                    b'data: [DONE]\n',
                ]
            )
            self.content = self
            self.headers = {}

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePostContext:
        def __init__(self):
            self.response = _FakeResponse()

        async def __aenter__(self):
            return await self.response.__aenter__()

        async def __aexit__(self, exc_type, exc, tb):
            return await self.response.__aexit__(exc_type, exc, tb)

    class _FakeSession:
        def post(self, api_url, json):
            return _FakePostContext()

    result = asyncio.run(
        send_one(
            _FakeSession(),
            api_url="http://127.0.0.1:12345/v1/completions",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            sample={
                "id": "req-1",
                "prompt": "prompt",
                "prompt_token_ids": None,
                "num_contexts": 1,
            },
            max_tokens=8,
        )
    )

    assert result["success"] is True
    assert result["generated_text"] == "Hello World"


def test_merge_server_profile_flattens_prefill_metrics():
    result = {"id": "req-1", "success": True}
    server_profile = {
        "server_true_ttft_ms": 123.0,
        "server_queue_ms": 4.0,
        "server_prefill_ms": 33.0,
        "server_scheduled_to_first_token_ms": 56.0,
        "server_load_path_ms": 33.0,
        "server_total_load_path_ms": 44.0,
        "server_blend_core_ms": 21.0,
    }

    merge_server_profile(result, server_profile)

    assert result["server_profile"] == server_profile
    assert result["server_true_ttft_ms"] == 123.0
    assert result["server_queue_ms"] == 4.0
    assert result["server_prefill_ms"] == 33.0
    assert result["server_ttft_no_queue_ms"] == 119.0
    assert result["server_scheduled_to_first_token_ms"] == 56.0
    assert result["server_prefill_compute_ms"] == 21.0
    assert result["server_prefill_path_ms"] == 33.0
    assert result["server_total_load_path_ms"] == 44.0


def test_send_one_merges_server_profile(monkeypatch):
    from exp.request_rate import catkv_memory_pressure as module

    class _FakeResponse:
        status = 200

        def __init__(self):
            self._chunks = iter(
                [
                    b'data: {"id":"req-server","choices":[{"text":"Hello"}]}\n',
                    b'data: [DONE]\n',
                ]
            )
            self.content = self
            self.headers = {}

        def __aiter__(self):
            return self

        async def __anext__(self):
            try:
                return next(self._chunks)
            except StopIteration as exc:
                raise StopAsyncIteration from exc

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class _FakePostContext:
        def __init__(self):
            self.response = _FakeResponse()

        async def __aenter__(self):
            return await self.response.__aenter__()

        async def __aexit__(self, exc_type, exc, tb):
            return await self.response.__aexit__(exc_type, exc, tb)

    class _FakeSession:
        def post(self, api_url, json):
            return _FakePostContext()

    async def _fake_collect(request_id):
        assert request_id == "req-server"
        return {
            "server_true_ttft_ms": 120.0,
            "server_queue_ms": 5.0,
            "server_prefill_ms": 34.0,
            "server_scheduled_to_first_token_ms": 60.0,
            "server_load_path_ms": 34.0,
            "server_total_load_path_ms": 45.0,
            "server_blend_core_ms": 22.0,
        }

    monkeypatch.setattr(
        module.profile_helpers,
        "maybe_collect_server_profile",
        _fake_collect,
    )

    result = asyncio.run(
        module.send_one(
            _FakeSession(),
            api_url="http://127.0.0.1:12345/v1/completions",
            model="mistralai/Mistral-7B-Instruct-v0.3",
            sample={
                "id": "req-2",
                "prompt": "prompt",
                "prompt_token_ids": None,
                "num_contexts": 1,
            },
            max_tokens=8,
            server_profile_log_path="server.log",
        )
    )

    assert result["server_true_ttft_ms"] == 120.0
    assert result["server_queue_ms"] == 5.0
    assert result["server_prefill_ms"] == 34.0
    assert result["server_ttft_no_queue_ms"] == 115.0
    assert result["server_scheduled_to_first_token_ms"] == 60.0
    assert result["server_prefill_compute_ms"] == 22.0
    assert result["server_prefill_path_ms"] == 34.0
    assert result["server_total_load_path_ms"] == 45.0
