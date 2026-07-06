
import fnmatch, json, os
from pathlib import Path
from time import time
from typing import Any, Dict, List, Tuple, Callable

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from catkv.kv_manager.kv_manager import KVCacheManager

from exp.utils.metric import MetricType, metric_func
from exp.utils.prompt import (
    SPECIAL_TOKENS,
    combine_contexts,
    normalize_context,
    serialize_and_hash,
    check_and_discard_contexts,
)
from catkv.kv_manager.utils import get_kvcache_filename
from catkv.strategy.abstract_blend import ProcessType
from catkv.utils.request import Request
from catkv.utils.hack import hack_model
from catkv.utils.config import parse_compress_config, parse_select_config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

RELEASE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_S3_CONFIG = str(RELEASE_ROOT / "csrc" / "config" / "s3.ini")

def get_model_and_tokenizer(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_path,
                                            #   local_files_only=True
                                              )
    
    model = AutoModelForCausalLM.from_pretrained(
        config.path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
    )
    model = hack_model(model, **config)
    
    return model, tokenizer

def load_model(config: dict) -> Tuple[Any, Any, Any]:
    device = config.get("device", "cuda")
    
    config.model.tokenizer_path = config.model.path
    model, tokenizer = get_model_and_tokenizer(config.model, device)
    
    return model, tokenizer, config

def ensure_result_dir(result_path: str):
    result_dir = os.path.dirname(result_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir, exist_ok=True)

def get_kvcache_stats(config):

    # if "disk_dir" not in config:
    #     return {"kvcache_file_count": 0, "kvcache_total_size_gb": 0.0}

    import catkv.kv_manager.utils as kv_utils

    disk_dir = kv_utils.store_kvcache_dir

    if not os.path.exists(disk_dir):
        return {"kvcache_file_count": 0, "kvcache_total_size_gb": 0.0}
  
    pattern = f"*{config.device}*"
    files = [f for f in os.listdir(disk_dir) if fnmatch.fnmatch(f, pattern)]
    total_size = sum(os.path.getsize(os.path.join(disk_dir, f)) for f in files if os.path.isfile(os.path.join(disk_dir, f)))

    return {"kvcache_file_count": len(files), "kvcache_total_size_gb": round(total_size / (1024**3), 4)}

def clean_kvcache_files(config):

    pattern = f"*{config.device}*"
    import catkv.kv_manager.utils as kv_utils
    disk_dir = kv_utils.store_kvcache_dir
    
    if not os.path.exists(disk_dir):
        return
    for filename in os.listdir(disk_dir):
        if fnmatch.fnmatch(filename, pattern):
            file_path = os.path.join(disk_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

def save_result(result_path: str, result: dict):
    ensure_result_dir(result_path)
    with open(result_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")

def get_hash(
    tokenizer,
    prompt,
):
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKENS]})
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if tokenized_prompt[0] == tokenized_prompt[1]:
        tokenized_prompt = tokenized_prompt[1:]
    spec_id = tokenizer(SPECIAL_TOKENS).input_ids[-1]
        
    indices = [
        idx for idx, input_id in enumerate(tokenized_prompt) if input_id == spec_id
    ]    

    return serialize_and_hash(tokenized_prompt[indices[0] + 1 :])  

def get_pred(
    model,
    tokenizer,
    prompt,
    max_gen,
    meta=None,
    config=None,
):
    searcher = Request(model, tokenizer, config.device)
    # add the special token.
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKENS]})
    tokenized_prompt_id = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0]
    if tokenized_prompt_id[0] == tokenized_prompt_id[1]:
        tokenized_prompt_id = tokenized_prompt_id[1:]
    special_token_id = tokenizer(SPECIAL_TOKENS).input_ids[-1]
        
    chunks_indices = [
        idx for idx, input_id in enumerate(tokenized_prompt_id) if input_id == special_token_id
    ]    
    chunks_hash_value = []

    if len(chunks_indices) == 1:
        chunks_hash_value.append(serialize_and_hash(tokenized_prompt_id[chunks_indices[0] + 1 :]))
    else :
        for idx in range(0, len(chunks_indices), 2):
            chunks_hash_value.append(
                serialize_and_hash(
                    tokenized_prompt_id[chunks_indices[idx] + 1 : chunks_indices[idx + 1]]
                )
            )

    spec_count = (tokenized_prompt_id == special_token_id).sum().item()
    tokens_num = len(tokenizer.tokenize(prompt)) - spec_count

    chunks_indices = [x - i for i, x in enumerate(chunks_indices)]
    if len(chunks_indices) >  1:
        chunks_indices = [[chunks_indices[i], chunks_indices[i + 1]] for i in range(0, len(chunks_indices), 2)]
    else:
        if meta["state"] == "store" and "offset" not in meta:
            chunks_indices = [[ 0 , tokens_num]]
            tokenized_prompt_id = tokenized_prompt_id[-len(tokenizer.tokenize(prompt)):]
        else:
            chunks_indices = [[ chunks_indices[0] , tokens_num]]
    
    if meta is None:
        meta = dict()

    meta["indices"] = chunks_indices
    meta["hash_text"] = chunks_hash_value
    meta["phase"] = "prefill"
    meta["device"] = config.device

    tokenized_prompt_id = tokenized_prompt_id[tokenized_prompt_id != special_token_id]
    meta["input_len"] = tokenized_prompt_id.size(0)

    model.blend_meta = meta
    output, ttft = searcher.generate(
        input_ids=tokenized_prompt_id,
        max_new_length=max_gen,
        device=config.device,
    )

    searcher.clear()
    return output, ttft

def get_kvcache_size(
    model,
    tokenizer,
    prompt
):
    tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_TOKENS]})
    tokenized_prompt = tokenizer(
        prompt, truncation=False, return_tensors="pt"
    ).input_ids[0] 
    if tokenized_prompt[0] == tokenized_prompt[1]:
        tokenized_prompt = tokenized_prompt[1:]
    spec_id = tokenizer(SPECIAL_TOKENS).input_ids[-1]
        
    indices = [
        idx for idx, input_id in enumerate(tokenized_prompt) if input_id == spec_id
    ]    

    file_hash = serialize_and_hash(tokenized_prompt[indices[0] + 1 :])

    import catkv.kv_manager.utils as kv_utils
    dir = kv_utils.store_kvcache_dir

    total_size = 0
    for file_name in os.listdir(dir):
        if file_hash in file_name:
            file_path = os.path.join(dir, file_name)
            if os.path.isfile(file_path):
                total_size += os.path.getsize(file_path)

    return total_size

def get_chunk_hash(
    tokenizer,
    example,
):
    contexts = example.get("contexts", [])
    chunk_hashes = []
    for context in contexts:
        chunk_hashes.append(get_hash(tokenizer, normalize_context(context)))
    
    return chunk_hashes

def eval_transfer(
    tokenizer,
    eval_dataset: List[Dict] = None,
    config: dict = None,
):
    requests = []
    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Evaluating Transfer Size"
    ):
        requests.append(
            get_chunk_hash(
                tokenizer=tokenizer,
                example=example,
            )
        )
    from catkv.kv_manager.disk.s3_disk import S3DiskManager

    config_path = os.environ.get("CATKV_S3_CONFIG", DEFAULT_S3_CONFIG)
    s3_manager = S3DiskManager(config_path)    

    # warm up 
    for idx , request in enumerate(requests):
        for layer in range(1, 32):
            task_id = s3_manager.load_datas([get_kvcache_filename(text , layer_idx = layer , device = config.device) for text in request] , config.device)
            _ = s3_manager.load_task(task_id)
        if idx >= 2:
            break
        

    for request in requests:
        begin = time.perf_counter()
        for layer in range(1, 32):
            task_id = s3_manager.load_datas([get_kvcache_filename(text , layer_idx = layer , device = config.device) for text in request] , config.device)
            _ = s3_manager.load_task(task_id)
        end = time.perf_counter()
        print(f"Transfer time for request : {end - begin} seconds")


    return requests

def process_example(
    model,
    tokenizer,
    max_new_length: int,
    example,
    prompt_template: str = "",
    config: List[Dict] = None,
    post_process: Callable = None,
    metric_type: MetricType = MetricType.F1,
):

    contexts = example.get("contexts", [])
    question = example.get("question", "")

    process_type, select_config = parse_select_config(config)
    compress_type, compress_config = parse_compress_config(config)

    # Handle select strategy
    blend_meta = {
        "select_strategy": process_type,
        "select_config": select_config,
        "compress_type": compress_type,
        "compress_config": compress_config,
        "state": "normal"
    }

    contexts = check_and_discard_contexts(
        contexts,
        config.max_model_len - len(tokenizer.tokenize(question)) - 512,
        tokenizer,
    )
    if process_type == ProcessType.DEFAULT:
        user_prompt = "".join(contexts)
    else:
        user_prompt = combine_contexts(contexts)
    KVCacheManager.clean_instance()

    user_prompt = prompt_template.format(
        context=user_prompt,
        input=question
    )

    if config.chat_template is True:
        messages = [{"role": "user", "content": user_prompt}]

        if model.__class__.__name__ == "Qwen3ForCausalLM":
            user_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            user_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    # Generate final prediction
    output, ttft = get_pred(model, tokenizer, user_prompt, max_new_length, meta=blend_meta, config=config)

    if post_process is not None:
        pred = post_process(output[0])
    else:
        pred = output[0]

    score = metric_func(metric_type, pred, example["answer"], tokenizer)
    result = {
        "id": example["id"],
        "pred": pred,
        "TTFT": ttft,
        "answer": example["answer"],
        "score": score,
        "input_toks": len(tokenizer.tokenize(user_prompt)),
        "output_toks": len(tokenizer.tokenize(output[0]))
    }

    return result, score


def process_example_repobench(
    model,
    tokenizer,
    max_new_length: int,
    example,
    prompt_template: str = "",
    config: List[Dict] = None,
    post_process: Callable = None,
    metric_type: MetricType = MetricType.F1,
):

    contexts = example.get("contexts", [])
    question = example.get("question", "")

    process_type, select_config = parse_select_config(config)
    compress_type, compress_config = parse_compress_config(config)

    # Handle select strategy
    blend_meta = {
        "select_strategy": process_type,
        "select_config": select_config,
        "compress_type": compress_type,
        "compress_config": compress_config,
        "state": "normal"
    }
    # blend_meta["offset"] = False

    contexts = check_and_discard_contexts(
        contexts,
        config.max_model_len - len(tokenizer.tokenize(question)) - 512,
        tokenizer,
    )
    if process_type == ProcessType.DEFAULT:
        user_prompt = "".join(contexts)
    else:
        # Store contexts with temporary state change
        for context in contexts:
            blend_meta["state"] = "store"
            get_pred(model, tokenizer, normalize_context(context), 0, meta=blend_meta, config=config)
            blend_meta["state"] = "normal"
        user_prompt = combine_contexts(contexts)
    KVCacheManager.clean_instance()

    user_prompt = prompt_template.format(
        context=user_prompt,
        input=question
    )

    if config.chat_template is True:
        messages = [{"role": "user", "content": user_prompt}]

        if model.__class__.__name__ == "Qwen3ForCausalLM":
            user_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            user_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

    # Generate final prediction
    output, ttft = get_pred(model, tokenizer, user_prompt, max_new_length, meta=blend_meta, config=config)

    if post_process is not None:
        pred = post_process(output[0])
    else:
        pred = output[0]

    score = metric_func(metric_type, pred, example["answer"], tokenizer)
    result = {
        "id": example["id"],
        "pred": pred,
        "TTFT": ttft,
        "answer": example["answer"],
        "score": score,
        "input_toks": len(tokenizer.tokenize(user_prompt)),
        "output_toks": len(tokenizer.tokenize(output[0]))
    }

    return result, score


def count_kvcache_files( 
    model,
    tokenizer,
    example,
):
    contexts = example.get("contexts", [])

    kv_count = 0
    for context in contexts:
        kv_count += get_kvcache_size(model, tokenizer, normalize_context(context))
  
    return kv_count

def count_kvcache_for_dataset(
    tokenizer,
    eval_dataset: List[Dict],
    config: dict = None,
):
    kv_count = 0
    hash_set = {}
    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Counting KVCache"
    ):
        
        hash_request = get_chunk_hash(
            tokenizer=tokenizer,
            example=example,
        )

        for h in hash_request:
            if h not in hash_set:
                hash_set[h] = 1
            else:
                hash_set[h] =  hash_set[h] + 1
        
    from catkv.kv_manager.disk.s3_disk import S3DiskManager

    config_path = os.environ.get("CATKV_S3_CONFIG", DEFAULT_S3_CONFIG)
    s3_manager = S3DiskManager(config_path) 

    def count_kv_size(data):
        count = 0
        for chunk_data in data:
            for key in chunk_data.keys():
               count += chunk_data[key].element_size() * chunk_data[key].nelement()
        return count

    cache_size = 0
    for h in hash_set.keys():
        task_id = s3_manager.load_datas([get_kvcache_filename(h , layer_idx = layer , device = config.device) for layer in range(1,32)] , config.device)
        _, gpu_data = s3_manager.load_task(task_id)
        cache_size += count_kv_size(gpu_data) * hash_set[h]
    print(f"Total unique KVCache size: {cache_size / (1024**3)} GB")
    return kv_count

def run_evaluation_repobench(
    model,
    tokenizer,
    max_new_length: int = 512,
    prompt_template: str = "",
    result_path: str = None,
    eval_dataset: List[Dict] = None,
    config: List[Dict] = None,
    metric_type: MetricType = MetricType.F1,
    post_process: Callable = None,
) -> Tuple[List[Dict], List[float]]:
    results = []
    scores = []

    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Processing examples"
    ):
        if example.get("id") is None:
            example["id"] = idx

        # try:
        result, score = process_example_repobench(
            model=model,
            tokenizer=tokenizer,
            max_new_length=max_new_length,
            example=example,
            prompt_template=prompt_template,
            config=config,
            metric_type=metric_type,
            post_process=post_process,
        )
        # except Exception as e:
        #     print(f"Error processing example {example['id']}: {e}")
        #     continue

        kvcache_stats = get_kvcache_stats(config)
        result.update(kvcache_stats)
        
        results.append(result)
        scores.append(score)
 
        save_result(result_path, result)

    return results, scores

def run_evaluation(
    model,
    tokenizer,
    max_new_length: int = 512,
    prompt_template: str = "",
    result_path: str = None,
    eval_dataset: List[Dict] = None,
    config: List[Dict] = None,
    metric_type: MetricType = MetricType.F1,
    post_process: Callable = None,
) -> Tuple[List[Dict], List[float]]:
    results = []
    scores = []

    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Processing examples"
    ):
        if example.get("id") is None:
            example["id"] = idx

        try:
            result, score = process_example(
                model=model,
                tokenizer=tokenizer,
                max_new_length=max_new_length,
                example=example,
                prompt_template=prompt_template,
                config=config,
                metric_type=metric_type,
                post_process=post_process,
            )
        except Exception as e:
            print(f"Error processing example {example['id']}: {e}")
            continue

        kvcache_stats = get_kvcache_stats(config)
        result.update(kvcache_stats)
        
        results.append(result)
        scores.append(score)
 
        save_result(result_path, result)

    return results, scores

def run_evaluation_precompute_(
    model,
    tokenizer,
    max_new_length: int = 512,
    prompt_template: str = "",
    result_path: str = None,
    eval_dataset: List[Dict] = None,
    config: List[Dict] = None,
    metric_type: MetricType = MetricType.F1,
    post_process: Callable = None,
) -> Tuple[List[Dict], List[float]]:
    results = []
    scores = []

    chunk_set = set()
    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Processing examples"
    ):
        if example.get("id") is None:
            example["id"] = idx
        compress_type, compress_config = parse_compress_config(config)

        blend_meta = {
            "compress_type": compress_type,
            "compress_config": compress_config,
            "state": "store"
        }
        for chunk in example.get("contexts", []):
            if chunk in chunk_set:
                continue
            get_pred(model, tokenizer, normalize_context(chunk), 0, meta=blend_meta, config=config)
            chunk_set.add(chunk)
    print(f"Precomputed {len(chunk_set)} chunks so far.")
    
    import catkv.kv_manager.utils as kv_utils
    # get the kvcache size of dir
    dir = kv_utils.store_kvcache_dir
    total_size = 0
    # ensure the dir exist
    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    for file_name in os.listdir(dir):
        file_path = os.path.join(dir, file_name)
        if os.path.isfile(file_path):
            total_size += os.path.getsize(file_path)
    print(f"Total KVCache size after precomputation: {total_size / (1024**3)} GB")

    return results, scores

def run_evaluation_precompute(
    model,
    tokenizer,
    max_new_length: int = 512,
    prompt_template: str = "",
    result_path: str = None,
    eval_dataset: List[Dict] = None,
    config: List[Dict] = None,
    metric_type: MetricType = MetricType.F1,
    post_process: Callable = None,
) -> Tuple[List[Dict], List[float]]:
    results = []
    scores = []
    import catkv.kv_manager.utils as kv_utils
    kv_utils.store_kvcache_dir = os.path.join(kv_utils.store_kvcache_dir , config.compress_strategy["type"]) + "-grouped"
    
    # chunk_set = set()
    all_chunk_set = set()
    for idx, example in tqdm(
        enumerate(eval_dataset), total=len(eval_dataset), desc="Processing examples"
    ):
        chunk_set = set()

        if example.get("id") is None:
            example["id"] = idx
        compress_type, compress_config = parse_compress_config(config)

        for chunk in example.get("contexts", []):

            if chunk in all_chunk_set:
                continue
            chunk_set.add(chunk)
            all_chunk_set.add(chunk)
            
        print(f"Example {idx} has {len(chunk_set)} unique chunks, total unique chunks so far: {len(all_chunk_set)}")

    compress_config["group_size"] = 4
    blend_meta = {
        "compress_type": compress_type,
        "compress_config": compress_config,
        "state": "store"
    }
    blend_meta["offset"] = False
    # divide chunk_set into batches
    chunk_list = list(all_chunk_set)
    # shuffle the chunk_list
    import random
    random.shuffle(chunk_list)
    group_size = compress_config["group_size"]
    for i in tqdm(range(0, len(chunk_list), group_size), desc="Precomputing chunks"):
        batch_chunks = chunk_list[i:i+group_size]
        combined_prompt = combine_contexts(batch_chunks)
        get_pred(model, tokenizer, combined_prompt, 0, meta=blend_meta, config=config)

    print(f"Precomputed {len(all_chunk_set)} chunks so far.")

    return results, scores
