
import argparse
import os
from datetime import timedelta
import subprocess
import torch
import torch.distributed as dist
import warnings

# Suppress common warnings from dependencies
warnings.filterwarnings('ignore', message='.*pkg_resources is deprecated.*')
warnings.filterwarnings('ignore', message='.*Using slow pure-python SequenceMatcher.*')

from exp.pred import load_model, run_evaluation, run_evaluation_precompute, run_evaluation_precompute_, run_evaluation_repobench
from exp.utils.metric import MetricType
from exp.utils.prompt import load_json_file
from catkv.utils.config import (
    get_cache_dir,
    load_unified_config,
    get_output_path,
    get_dataset_maxlen,
    get_dataset_prompt,
    get_dataset_path
)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', "-m", type=str, required=True, help="Path to the model configuration file")
    parser.add_argument('--state', "-s" , type=str, default="normal", choices=["normal" , "reuse"], help="State of the model, normal or reuse")
    parser.add_argument('--task_config', "-t", type=str, required=True, help="Path to the task configuration file")
    return parser.parse_args(args)


def single_process_evaluation(args):
    """Single process evaluation (non-distributed)"""
    # Clean up any orphaned worker processes from previous runs
    print(f"Model Config: {args.model_config}")
    print(f"Task Config: {args.task_config}")
    
    # Load unified configuration
    config = load_unified_config(args.model_config, args.task_config)
    
    print("Loading model...")
    model, tokenizer, model_config = load_model(config)
    print("Model loaded successfully.")

    # init the store kvcache dir
    import catkv.kv_manager.utils as kv_utils

    # Process datasets
    for dataset_name in config.datasets:
        print(f"Processing dataset: {dataset_name}")
        if "disk_dir" in config:
            kv_utils.store_kvcache_dir =  os.path.join(config.disk_dir, dataset_name , get_cache_dir(config, dataset_name))
        else:
            kv_utils.store_kvcache_dir = "kvcache/global_blocks_data-"
    
        # Generate output path using unified configuration system
        output_path = get_output_path(config, dataset_name , "result-test")

        max_new_length = get_dataset_maxlen(dataset_name)
        prompt_template = get_dataset_prompt(dataset_name)
        
        dataset_path = get_dataset_path(dataset_name)
        dataset = load_json_file(dataset_path)
        
        # QA task
        if dataset_name in ["wikimqa", "needle-single", "needle-multi", "needle-multi-rag", "musique", "triviaqa", "hotpotqa"]:
            metric_type = MetricType.F1
        elif dataset_name in ["samsum", "multi_news"]:
            metric_type = MetricType.RL
        elif dataset_name in ["repobench-p"]:
            metric_type = MetricType.CS
        elif dataset_name in ["dureader"]:
            metric_type = MetricType.RLZH
    
        # For summarization tasks, we might need to adjust the prompt template
        config.chat_template = False if dataset_name in ["samsum", "triviaqa", "repobench-p"] else True

        if dataset_name == "samsum":
            def post_process(output):
                if "\n" in output:
                    return output.split("\n")[0].strip()
                return output
        else:
            post_process = None

        print("config:", config)
        
        if dataset_name == "repobench-p":
            results, score = run_evaluation_repobench(
                model=model,
                tokenizer=tokenizer,
                max_new_length=max_new_length,
                prompt_template=prompt_template,
                eval_dataset=dataset,
                result_path=output_path,
                config=config,
                metric_type=metric_type,
                post_process=post_process
            )
        else:
            results, score = run_evaluation_precompute_(
                    model=model,
                    tokenizer=tokenizer,
                    max_new_length=max_new_length,
                    prompt_template=prompt_template,
                    eval_dataset=dataset,
                    result_path=output_path,
                    config=config,
                    metric_type=metric_type,
                    post_process=post_process
            ) 
            
            results, score = run_evaluation(
                model=model,
                tokenizer=tokenizer,
                max_new_length=max_new_length,
                prompt_template=prompt_template,
                eval_dataset=dataset,
                result_path=output_path,
                config=config,
                metric_type=metric_type,
                post_process=post_process
            )

if __name__ == "__main__":
    args = parse_args()
    single_process_evaluation(args)


