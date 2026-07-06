#!/usr/bin/env python3
"""Warm all WikiMQA contexts, then write per-request generations to JSONL."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
RELEASE_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = RELEASE_ROOT / "dataset" / "wikimqa_s.jsonl"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
DEFAULT_S3_CONFIG = RELEASE_ROOT / "csrc" / "config" / "s3.ini"
BLEND_SPECIAL_STR = " # # "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--compress-type", default="OURS")
    parser.add_argument("--output", default=None)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=10000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--catkv-s3-config", default=DEFAULT_S3_CONFIG)
    parser.add_argument("--catkv-num-workers", type=int, default=32)
    parser.add_argument("--catkv-max-rss-gib", type=float, default=200.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = BLEND_SPECIAL_STR
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = "0.15"
    os.environ["LMCACHE_BLEND_MIN_TOKENS"] = "64"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
    os.environ["LMCACHE_COMPRESS_TYPE"] = str(args.compress_type).upper()
    os.environ["LMCACHE_USE_CATKV_OPS"] = "1"
    os.environ["LMCACHE_CATKV_OPS_S3_CONFIG"] = str(args.catkv_s3_config)
    os.environ["LMCACHE_CATKV_OPS_NUM_WORKERS"] = str(args.catkv_num_workers)
    os.environ["LMCACHE_CATKV_OPS_MAX_RSS_GIB"] = str(args.catkv_max_rss_gib)


def load_samples(path: str, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as dataset:
        for line in dataset:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def unique_contexts(samples: list[dict[str, Any]]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for sample in samples:
        for context in sample.get("contexts") or []:
            if context not in seen:
                seen.add(context)
                ordered.append(context)
    return ordered


def output_path(args: argparse.Namespace) -> Path:
    if args.output:
        return Path(args.output)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_name = args.model.split("/")[-1].replace(".", "_")
    return DEFAULT_OUTPUT_DIR / f"wikimqa_{model_name}_{timestamp}.jsonl"


def build_prompt_tokens(sample: dict[str, Any], tokenizer, sep_tokens: list[int]) -> list[int]:
    prefix = (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
    )
    suffix = (
        "\nAnswer the question based on the given passages. Answer the question "
        "within 5 words. Only give me the answer and do not output any other words.\n\n"
        f"Question: {sample.get('question', '')}\nAnswer:"
    )
    tokens = tokenizer.encode(prefix, add_special_tokens=False)
    for context in sample.get("contexts") or []:
        tokens.extend(sep_tokens)
        tokens.extend(tokenizer.encode(context, add_special_tokens=False))
    tokens.extend(sep_tokens)
    tokens.extend(tokenizer.encode(suffix, add_special_tokens=False))
    return tokens


def normalize_answers(sample: dict[str, Any]) -> list[str]:
    answer = sample.get("answer", [])
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer)]


def exact_match(prediction: str, answers: list[str]) -> bool:
    normalized = prediction.strip().lower()
    return any(normalized == answer.strip().lower() for answer in answers)


def contains_match(prediction: str, answers: list[str]) -> bool:
    normalized = prediction.strip().lower()
    return any(answer.strip().lower() in normalized for answer in answers)


@contextlib.contextmanager
def build_llm(args: argparse.Namespace):
    from lmcache.integration.vllm.utils import ENGINE_NAME
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from vllm import LLM
    from vllm.config import KVTransferConfig
    from vllm.engine.arg_utils import EngineArgs

    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    engine_args = EngineArgs(
        model=args.model,
        kv_transfer_config=kv_transfer_config,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        enforce_eager=True,
        tensor_parallel_size=1,
    )
    llm = LLM(**asdict(engine_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


def run_generate(llm, prompt_tokens: list[int], sampling_params):
    started = time.perf_counter()
    outputs = llm.generate(
        prompts={"prompt_token_ids": prompt_tokens},
        sampling_params=sampling_params,
        use_tqdm=False,
    )
    latency = time.perf_counter() - started
    output = outputs[0]
    generated_text = output.outputs[0].text if output.outputs else ""
    return generated_text, latency, output


def warmup_contexts(llm, contexts: list[str], tokenizer, sampling_params) -> None:
    total = len(contexts)
    for index, context in enumerate(contexts, start=1):
        tokens = tokenizer.encode(context, add_special_tokens=False)
        _, latency, _ = run_generate(llm, tokens, sampling_params)
        print(
            f"[warmup {index}/{total}] tokens={len(tokens)} latency={latency:.2f}s",
            flush=True,
        )


def main() -> None:
    args = parse_args()
    configure_environment(args)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(BLEND_SPECIAL_STR, add_special_tokens=False)
    samples = load_samples(args.dataset, args.num_samples)
    if not samples:
        raise RuntimeError(f"No samples loaded from {args.dataset}")

    out_path = output_path(args)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    warmup_params = SamplingParams(temperature=0, max_tokens=args.warmup_max_tokens)
    request_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print(f"model={args.model}")
    print(f"dataset={args.dataset} samples={len(samples)}")
    print(f"output={out_path}")
    print(f"cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"compress_type={os.environ.get('LMCACHE_COMPRESS_TYPE')}")
    print(f"sep_tokens={sep_tokens}")

    with build_llm(args) as llm:
        if not args.skip_warmup:
            contexts = unique_contexts(samples)
            print(f"warming unique contexts={len(contexts)}", flush=True)
            warmup_contexts(llm, contexts, tokenizer, warmup_params)

        with out_path.open("w", encoding="utf-8") as output_file:
            for index, sample in enumerate(samples, start=1):
                prompt_tokens = build_prompt_tokens(sample, tokenizer, sep_tokens)
                prediction, latency, request_output = run_generate(
                    llm, prompt_tokens, request_params
                )
                answers = normalize_answers(sample)
                record = {
                    "id": sample.get("id", index - 1),
                    "prediction": prediction.strip(),
                    "generated_text": prediction,
                    "answers": answers,
                    "exact_match": exact_match(prediction, answers),
                    "contains_match": contains_match(prediction, answers),
                    "question": sample.get("question", ""),
                    "latency_s": latency,
                    "prompt_tokens": len(prompt_tokens),
                    "completion_tokens": len(request_output.outputs[0].token_ids)
                    if request_output.outputs
                    else 0,
                    "model": args.model,
                    "dataset": str(args.dataset),
                    "mode": "offline-cacheblend-warm-all-contexts",
                    "compress_type": str(args.compress_type).upper(),
                }
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_file.flush()
                print(
                    f"[sample {index}/{len(samples)}] id={record['id']} "
                    f"prediction={record['prediction']!r} latency={latency:.2f}s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
