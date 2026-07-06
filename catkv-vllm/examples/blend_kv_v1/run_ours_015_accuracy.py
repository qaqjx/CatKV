#!/usr/bin/env python3
"""Evaluate CatKV CacheBlend OURS at 0.15 on QA datasets with vLLM."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import string
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
RELEASE_ROOT = Path(__file__).resolve().parents[3]
CURRENT_DIR = Path(__file__).resolve().parent
DEFAULT_DATASETS = {
    "needle_multi_rag": RELEASE_ROOT / "dataset" / "needle_multi_rag.jsonl",
    "wikimqa": RELEASE_ROOT / "dataset" / "wikimqa_s.jsonl",
    "musique": RELEASE_ROOT / "dataset" / "musique_s.jsonl",
}
DEFAULT_DATASET_ORDER = ["needle_multi_rag", "wikimqa", "musique"]
DEFAULT_OUTPUT_ROOT = CURRENT_DIR / "results" / "ours_0_15_accuracy"
DEFAULT_S3_CONFIG = RELEASE_ROOT / "csrc" / "config" / "s3.ini"
BLEND_SPECIAL_STR = " # # "


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASET_ORDER,
        help=f"Dataset names to run. Defaults to: {', '.join(DEFAULT_DATASET_ORDER)}",
    )
    parser.add_argument(
        "--dataset-path",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Override or add a dataset path. Can be passed multiple times.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--compress-type", default="OURS")
    parser.add_argument("--compress-ratio", type=float, default=0.15)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--warmup-max-tokens", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=16000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--catkv-s3-config", default=str(DEFAULT_S3_CONFIG))
    parser.add_argument("--catkv-num-workers", type=int, default=32)
    parser.add_argument("--catkv-max-rss-gib", type=float, default=200.0)
    parser.add_argument("--skip-warmup", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def configure_environment(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_visible_devices)
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"
    os.environ["LMCACHE_ENABLE_BLENDING"] = "True"
    os.environ["LMCACHE_BLEND_SPECIAL_STR"] = BLEND_SPECIAL_STR
    os.environ["LMCACHE_USE_LAYERWISE"] = "True"
    os.environ["LMCACHE_BLEND_CHECK_LAYERS"] = "1"
    os.environ["LMCACHE_BLEND_RECOMPUTE_RATIOS"] = str(args.compress_ratio)
    os.environ["LMCACHE_BLEND_MIN_TOKENS"] = "64"
    os.environ["LMCACHE_LOCAL_CPU"] = "True"
    os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
    os.environ["LMCACHE_COMPRESS_TYPE"] = str(args.compress_type).upper()
    os.environ["LMCACHE_USE_CATKV_OPS"] = "1"
    os.environ["LMCACHE_CATKV_OPS_S3_CONFIG"] = str(args.catkv_s3_config)
    os.environ["LMCACHE_CATKV_OPS_NUM_WORKERS"] = str(args.catkv_num_workers)
    os.environ["LMCACHE_CATKV_OPS_MAX_RSS_GIB"] = str(args.catkv_max_rss_gib)


def dataset_paths(args: argparse.Namespace) -> dict[str, Path]:
    paths = dict(DEFAULT_DATASETS)
    for override in args.dataset_path:
        if "=" not in override:
            raise ValueError(f"--dataset-path must use NAME=PATH format: {override}")
        name, value = override.split("=", 1)
        paths[name] = Path(value).expanduser().resolve()
    return paths


def load_samples(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as dataset:
        for line in dataset:
            line = line.strip()
            if not line:
                continue
            samples.append(json.loads(line))
            if limit is not None and len(samples) >= limit:
                break
    return samples


def normalize_answers(sample: dict[str, Any]) -> list[str]:
    answer = sample.get("answer", [])
    if isinstance(answer, list):
        return [str(item) for item in answer]
    return [str(answer)]


def parse_generation(text: str) -> str:
    first_word = text.strip().lower().split()[0] if text.strip() else ""
    if first_word == "yes":
        return "Yes"
    if first_word == "no":
        return "No"
    return text


def normalize_answer(text: str) -> str:
    def remove_articles(value: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", value)

    def white_space_fix(value: str) -> str:
        return " ".join(value.split())

    def remove_punc(value: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in value if ch not in exclude)

    text = text.lstrip("\n").split("\n")[0]
    return white_space_fix(remove_articles(remove_punc(text.lower())))


def compute_f1(prediction: str, answers: list[str]) -> float:
    prediction = parse_generation(prediction)
    pred_tokens = normalize_answer(prediction).split()
    scores = []
    for answer in answers:
        answer_tokens = normalize_answer(answer).split()
        common = Counter(answer_tokens) & Counter(pred_tokens)
        num_same = sum(common.values())
        if len(answer_tokens) == 0 or len(pred_tokens) == 0:
            scores.append(float(answer_tokens == pred_tokens))
        elif num_same == 0:
            scores.append(0.0)
        else:
            precision = num_same / len(pred_tokens)
            recall = num_same / len(answer_tokens)
            scores.append((2 * precision * recall) / (precision + recall))
    return max(scores) if scores else 0.0


def compute_em(prediction: str, answers: list[str]) -> int:
    prediction = parse_generation(prediction)
    normalized = normalize_answer(prediction)
    return int(any(normalized == normalize_answer(answer) for answer in answers))


def contains_match(prediction: str, answers: list[str]) -> bool:
    normalized = normalize_answer(prediction)
    return any(normalize_answer(answer) in normalized for answer in answers)


def build_prefix() -> str:
    return (
        "Answer the question based on the given passages. Only give me the answer "
        "and do not output any other words.\n\nThe following are given passages.\n"
    )


def build_suffix(question: str) -> str:
    return (
        "\nAnswer the question based on the given passages. Answer the question "
        "within 5 words. Only give me the answer and do not output any other words.\n\n"
        f"Question: {question}\nAnswer:"
    )


def prepare_prompt_tokens(
    sample: dict[str, Any],
    tokenizer,
    sep_tokens: list[int],
    max_model_len: int,
    max_tokens: int,
) -> dict[str, Any]:
    prefix_tokens = tokenizer.encode(build_prefix(), add_special_tokens=False)
    suffix_tokens = tokenizer.encode(
        build_suffix(str(sample.get("question", ""))), add_special_tokens=False
    )
    context_texts = [str(context) for context in sample.get("contexts") or []]
    context_token_ids = [
        tokenizer.encode(context, add_special_tokens=False) for context in context_texts
    ]

    def total_len(token_groups: list[list[int]]) -> int:
        return (
            len(prefix_tokens)
            + len(suffix_tokens)
            + len(sep_tokens) * (len(token_groups) + 1)
            + sum(len(tokens) for tokens in token_groups)
        )

    selected_contexts = list(context_texts)
    selected_token_ids = list(context_token_ids)
    while selected_token_ids and total_len(selected_token_ids) + max_tokens > max_model_len:
        selected_token_ids.pop()
        selected_contexts.pop()

    prompt_tokens = list(prefix_tokens)
    for context_tokens in selected_token_ids:
        prompt_tokens.extend(sep_tokens)
        prompt_tokens.extend(context_tokens)
    prompt_tokens.extend(sep_tokens)
    prompt_tokens.extend(suffix_tokens)

    if len(prompt_tokens) + max_tokens > max_model_len:
        raise ValueError(
            "Prompt does not fit max_model_len even after dropping contexts: "
            f"prompt_tokens={len(prompt_tokens)} max_tokens={max_tokens} "
            f"max_model_len={max_model_len}"
        )

    return {
        "prompt_tokens": prompt_tokens,
        "context_texts": selected_contexts,
        "context_token_ids": selected_token_ids,
        "context_token_lengths": [len(tokens) for tokens in selected_token_ids],
        "dropped_contexts": len(context_texts) - len(selected_contexts),
        "original_contexts": len(context_texts),
    }


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
    completion_tokens = len(output.outputs[0].token_ids) if output.outputs else 0
    return generated_text, latency, completion_tokens


def warmup_contexts(llm, context_token_ids: list[list[int]], sampling_params) -> list[dict[str, Any]]:
    records = []
    total = len(context_token_ids)
    for index, tokens in enumerate(context_token_ids, start=1):
        _, latency, completion_tokens = run_generate(llm, tokens, sampling_params)
        records.append(
            {
                "index": index,
                "total": total,
                "prompt_tokens": len(tokens),
                "completion_tokens": completion_tokens,
                "latency_s": latency,
            }
        )
    return records


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        output_file.flush()


class RunLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, message: str) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as log_file:
            log_file.write(line + "\n")


def existing_ids(path: Path) -> set[Any]:
    if not path.exists():
        return set()
    ids = set()
    with path.open("r", encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            try:
                ids.add(json.loads(line).get("id"))
            except json.JSONDecodeError:
                continue
    return ids


def summarize_records(records: list[dict[str, Any]], dataset: str, dataset_path: Path) -> dict[str, Any]:
    successes = [record for record in records if record.get("success")]
    failures = [record for record in records if not record.get("success")]
    f1_scores = [record["f1"] for record in successes]
    em_scores = [record["em"] for record in successes]
    latencies = [record["latency_s"] for record in successes]
    warmup_latencies = [
        item["latency_s"]
        for record in successes
        for item in record.get("warmup", [])
        if item.get("latency_s") is not None
    ]
    return {
        "dataset": dataset,
        "dataset_path": str(dataset_path),
        "total": len(records),
        "success": len(successes),
        "failed": len(failures),
        "f1": mean(f1_scores) if f1_scores else 0.0,
        "f1_percent": (mean(f1_scores) * 100) if f1_scores else 0.0,
        "em": mean(em_scores) if em_scores else 0.0,
        "em_percent": (mean(em_scores) * 100) if em_scores else 0.0,
        "latency_avg_s": mean(latencies) if latencies else None,
        "warmup_latency_avg_s": mean(warmup_latencies) if warmup_latencies else None,
        "errors": [
            {"id": record.get("id"), "error": record.get("error")}
            for record in failures
        ],
    }


def run_dataset(
    *,
    dataset: str,
    dataset_path: Path,
    samples: list[dict[str, Any]],
    llm,
    tokenizer,
    sep_tokens: list[int],
    args: argparse.Namespace,
    run_dir: Path,
    log: RunLogger,
    request_params,
    warmup_params,
) -> dict[str, Any]:
    jsonl_path = run_dir / f"{dataset}.jsonl"
    if not args.resume and jsonl_path.exists():
        jsonl_path.unlink()

    skipped_ids = existing_ids(jsonl_path) if args.resume else set()
    records: list[dict[str, Any]] = []
    if args.resume and jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as input_file:
            records = [json.loads(line) for line in input_file if line.strip()]

    log(f"dataset={dataset} path={dataset_path} samples={len(samples)} output={jsonl_path}")
    for index, sample in enumerate(samples, start=1):
        sample_id = sample.get("id", index - 1)
        if sample_id in skipped_ids:
            log(f"[{dataset} {index}/{len(samples)}] id={sample_id} skipped by resume")
            continue

        try:
            prepared = prepare_prompt_tokens(
                sample,
                tokenizer,
                sep_tokens,
                args.max_model_len,
                args.max_tokens,
            )
            warmup_records: list[dict[str, Any]] = []
            if not args.skip_warmup:
                log(
                    f"[{dataset} {index}/{len(samples)}] id={sample_id} "
                    f"warmup_contexts={len(prepared['context_token_ids'])} "
                    f"prompt_tokens={len(prepared['prompt_tokens'])}"
                )
                warmup_records = warmup_contexts(
                    llm,
                    prepared["context_token_ids"],
                    warmup_params,
                )

            prediction, latency, completion_tokens = run_generate(
                llm,
                prepared["prompt_tokens"],
                request_params,
            )
            answers = normalize_answers(sample)
            f1 = compute_f1(prediction, answers)
            em = compute_em(prediction, answers)
            record = {
                "dataset": dataset,
                "sample_index": index,
                "id": sample_id,
                "success": True,
                "prediction": prediction.strip(),
                "generated_text": prediction,
                "answers": answers,
                "question": sample.get("question", ""),
                "f1": f1,
                "f1_percent": f1 * 100,
                "em": em,
                "exact_match": bool(em),
                "contains_match": contains_match(prediction, answers),
                "latency_s": latency,
                "prompt_tokens": len(prepared["prompt_tokens"]),
                "completion_tokens": completion_tokens,
                "context_token_lengths": prepared["context_token_lengths"],
                "context_count": len(prepared["context_token_ids"]),
                "original_contexts": prepared["original_contexts"],
                "dropped_contexts": prepared["dropped_contexts"],
                "warmup": warmup_records,
                "model": args.model,
                "mode": "offline-vllm-cacheblend-warm-contexts-per-request",
                "compress_type": str(args.compress_type).upper(),
                "compress_ratio": args.compress_ratio,
            }
            log(
                f"[{dataset} {index}/{len(samples)}] id={sample_id} "
                f"f1={f1 * 100:.2f} em={em} latency={latency:.2f}s "
                f"prediction={record['prediction']!r}"
            )
        except Exception as exc:  # Keep full-dataset runs from losing earlier results.
            record = {
                "dataset": dataset,
                "sample_index": index,
                "id": sample_id,
                "success": False,
                "question": sample.get("question", ""),
                "answers": normalize_answers(sample),
                "error": repr(exc),
                "model": args.model,
                "mode": "offline-vllm-cacheblend-warm-contexts-per-request",
                "compress_type": str(args.compress_type).upper(),
                "compress_ratio": args.compress_ratio,
            }
            log(f"[{dataset} {index}/{len(samples)}] id={sample_id} error={exc!r}")

        append_jsonl(jsonl_path, record)
        records.append(record)
        summary = summarize_records(records, dataset, dataset_path)
        write_json(run_dir / f"{dataset}.summary.json", summary)

    summary = summarize_records(records, dataset, dataset_path)
    write_json(run_dir / f"{dataset}.summary.json", summary)
    log(
        f"dataset={dataset} done success={summary['success']}/{summary['total']} "
        f"f1={summary['f1_percent']:.2f} em={summary['em_percent']:.2f}"
    )
    return summary


def main() -> None:
    args = parse_args()
    configure_environment(args)

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    paths = dataset_paths(args)
    unknown = [dataset for dataset in args.datasets if dataset not in paths]
    if unknown:
        raise ValueError(f"Unknown datasets: {unknown}. Known datasets: {sorted(paths)}")

    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(args.output_root).expanduser().resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    log = RunLogger(run_dir / "run.log")

    run_config = {
        "model": args.model,
        "datasets": args.datasets,
        "dataset_paths": {name: str(paths[name]) for name in args.datasets},
        "compress_type": str(args.compress_type).upper(),
        "compress_ratio": args.compress_ratio,
        "max_tokens": args.max_tokens,
        "warmup_max_tokens": args.warmup_max_tokens,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cuda_visible_devices": str(args.cuda_visible_devices),
        "catkv_s3_config": str(args.catkv_s3_config),
        "catkv_num_workers": args.catkv_num_workers,
        "catkv_max_rss_gib": args.catkv_max_rss_gib,
        "skip_warmup": args.skip_warmup,
        "resume": args.resume,
        "env": {
            key: os.environ.get(key)
            for key in sorted(os.environ)
            if key.startswith("LMCACHE_") or key.startswith("CATKV_")
        },
    }
    write_json(run_dir / "run_config.json", run_config)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    sep_tokens = tokenizer.encode(BLEND_SPECIAL_STR, add_special_tokens=False)
    warmup_params = SamplingParams(temperature=0.0, max_tokens=args.warmup_max_tokens)
    request_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    log(f"run_dir={run_dir}")
    log(f"model={args.model}")
    log(f"datasets={args.datasets}")
    log(f"compress_type={os.environ.get('LMCACHE_COMPRESS_TYPE')}")
    log(f"compress_ratio={os.environ.get('LMCACHE_BLEND_RECOMPUTE_RATIOS')}")
    log(f"sep_tokens={sep_tokens}")

    summaries: dict[str, Any] = {}
    with build_llm(args) as llm:
        for dataset in args.datasets:
            samples = load_samples(paths[dataset], args.num_samples)
            if not samples:
                raise RuntimeError(f"No samples loaded from {paths[dataset]}")
            summaries[dataset] = run_dataset(
                dataset=dataset,
                dataset_path=paths[dataset],
                samples=samples,
                llm=llm,
                tokenizer=tokenizer,
                sep_tokens=sep_tokens,
                args=args,
                run_dir=run_dir,
                log=log,
                request_params=request_params,
                warmup_params=warmup_params,
            )
            write_json(run_dir / "summary.json", summaries)

    write_json(run_dir / "summary.json", summaries)
    log(f"all done summary={run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
