"""
Configuration management utilities for catkv.
Handles unified configuration loading, merging, and output file naming.
"""

import json
import os
from typing import Dict, Any
from omegaconf import OmegaConf, DictConfig
from pathlib import Path


class ConfigManager:
    """Centralized configuration manager with extensible naming strategy."""
    
    def __init__(self, config_root: str = "config"):
        self.project_root = Path(__file__).resolve().parents[2]
        self.release_root = self.project_root.parent
        config_root_path = Path(config_root).expanduser()
        if not config_root_path.is_absolute():
            config_root_path = self.project_root / config_root_path
        self.config_root = config_root_path
        self.models_dir = self.config_root / "models"
        self.tasks_dir = self.config_root / "task"
        self._dataset_configs = self._load_dataset_configs()
        
        # Strategy registries
        self._strategy_registries = {
            'select': {
                "CACHEBLEND": ["recompute_ratio", "key_or_value", "deviation_type", "mask_type" , "ablation"],
                "EPIC": ["recompute_num","mask_type"], 
                "KVSHARE": ["recompute_ratio"],
                "DEFAULT": []
            },
            'compress': {
                "OURS": ["ratio"],
                "NONE": []
            }
        }

    def register_strategy(self, strategy_type: str, strategy_name: str, param_names: list):
        """Register a new strategy with its parameter names."""
        self._strategy_registries[strategy_type][strategy_name] = param_names
        
    def _load_dataset_configs(self) -> Dict[str, Any]:
        """Load all dataset configurations."""
        dataset_dir = self.config_root / "dataset"
        configs = {}
        
        for config_file in ["dataset2maxlen.json", "dataset2prompt.json", "dataset2path.json"]:
            config_type = config_file.split('2')[1].split('.')[0]  # Extract type from filename
            file_path = dataset_dir / config_file
            configs[config_type] = json.load(open(file_path)) if file_path.exists() else {}
                
        return configs
    
    def load_config(self, model_config_path: str, task_config_path: str) -> DictConfig:
        """Load and merge model and task configurations."""
        model_path = str(self.models_dir / model_config_path) if not model_config_path.startswith('/') else model_config_path
        task_path = str(self.tasks_dir / task_config_path) if not task_config_path.startswith('/') else task_config_path
        
        return OmegaConf.merge(OmegaConf.load(model_path), OmegaConf.load(task_path))
    
    def get_dataset_config(self, dataset_name: str, config_type: str, default=None):
        """Get specific dataset configuration value."""
        return self._dataset_configs.get(config_type, {}).get(dataset_name, default)

    def resolve_path(self, path: str | None) -> str | None:
        """Resolve config paths relative to the project/release roots."""
        if path is None:
            return None
        raw_path = os.path.expandvars(os.path.expanduser(str(path)))
        candidate = Path(raw_path)
        if candidate.is_absolute():
            return str(candidate)
        for base_dir in (self.project_root, self.release_root):
            resolved = (base_dir / candidate).resolve()
            if resolved.exists():
                return str(resolved)
        return str((self.project_root / candidate).resolve())
    
    def _get_strategy_params(self, strategy_config: DictConfig, strategy_type: str) -> list:
        """Extract strategy-specific parameters for filename."""
        strategy_name = strategy_config.type
        params = [strategy_name] if strategy_type == 'select' else ([f"compress={strategy_name}"] if strategy_name != "None" else [])
        
        param_names = self._strategy_registries[strategy_type].get(strategy_name.upper(), [])
        for param_name in param_names:
            if hasattr(strategy_config, param_name):
                params.append(f"{param_name}={getattr(strategy_config, param_name)}")
                
        return params
    
    def generate_cache_dirname(self, config: DictConfig) -> str:
        """Generate cache directory name based on configuration."""
        model_name = getattr(config.model, 'name_or_path', config.model.path)
        filename_parts = [model_name]
        
        filename_parts.extend(self._get_strategy_params(config.select_strategy, 'select'))

        if hasattr(config, 'compress_strategy'):
            filename_parts.extend(self._get_strategy_params(config.compress_strategy, 'compress'))
            
        return "@".join(filename_parts)

    def generate_output_filename(self, config: DictConfig) -> str:
        """Generate output filename based on configuration."""
        model_name = getattr(config.model, 'name_or_path', config.model.path)
        filename_parts = [model_name]
        
        filename_parts.extend(self._get_strategy_params(config.select_strategy, 'select'))

        if hasattr(config, 'compress_strategy'):
            filename_parts.extend(self._get_strategy_params(config.compress_strategy, 'compress'))
            
        return "@".join(filename_parts) + ".jsonl"
    
    def parse_compress_config(self, config: DictConfig) -> tuple:
        """Parse compress_strategy from config and return CompressType and config dict"""
        from catkv.kv_manager.compress.abstract_compress import CompressType
        
        compress_strategy = config.compress_strategy
        strategy_type = compress_strategy.type.upper()
        if strategy_type not in self._strategy_registries["compress"]:
            raise ValueError(
                f"Unsupported compression type: {compress_strategy.type}. "
                "Open-source CatKV supports only None and Ours."
            )
        
        param_names = self._strategy_registries['compress'].get(strategy_type, [])
        compress_config = {
            param_name: getattr(compress_strategy, param_name)
            for param_name in param_names 
            if hasattr(compress_strategy, param_name)
        }
        
        return CompressType[strategy_type], compress_config

    def parse_select_config(self, config: DictConfig) -> tuple:
        """Parse select_strategy from config and return ProcessType and config dict"""
        from catkv.strategy.abstract_blend import ProcessType
        
        select_strategy = config.select_strategy
        strategy_type = select_strategy.type.upper()
        
        param_names = self._strategy_registries['select'].get(strategy_type, [])
        select_config = {
            param_name: getattr(select_strategy, param_name)
            for param_name in param_names 
            if hasattr(select_strategy, param_name)
        }
        
        return ProcessType[strategy_type.upper()], select_config
    
    def create_output_path(self, config: DictConfig, dataset_name: str, base_dir: str = "result-final-revision-size-score") -> str:
        """Create full output path for results."""
        output_path = os.path.join(base_dir, dataset_name, self.generate_output_filename(config))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path
    
    def create_cache_dir(self, config: DictConfig, dataset_name: str, base_dir: str = "kvcache") -> str:
        """Create cache directory path."""
        cache_dirname = self.generate_cache_dirname(config)
        cache_dir = os.path.join(base_dir, dataset_name, cache_dirname)
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir


# Global config manager instance
config_manager = ConfigManager()

# Convenience functions
def load_unified_config(model_config_path: str, task_config_path: str) -> DictConfig:
    return config_manager.load_config(model_config_path, task_config_path)

def get_output_path(config: DictConfig, dataset_name: str , base_dir: str = "result-final-revision-size-score") -> str:
    return config_manager.create_output_path(config, dataset_name, base_dir)

def get_cache_dir(config: DictConfig, dataset_name: str) -> str:
    return config_manager.create_cache_dir(config, dataset_name)

def get_dataset_maxlen(dataset_name: str, default: int = 512) -> int:
    return config_manager.get_dataset_config(dataset_name, 'maxlen', default)

def get_dataset_prompt(dataset_name: str, default: str = "{context}\n\n{input}") -> str:
    return config_manager.get_dataset_config(dataset_name, 'prompt', default)

def get_dataset_path(dataset_name: str) -> str:
    return config_manager.resolve_path(
        config_manager.get_dataset_config(dataset_name, 'path')
    )

def parse_compress_config(config: DictConfig) -> tuple:
    return config_manager.parse_compress_config(config)

def parse_select_config(config: DictConfig) -> tuple:
    return config_manager.parse_select_config(config)

def register_select_strategy(strategy_name: str, param_names: list):
    config_manager.register_strategy('select', strategy_name, param_names)

def register_compress_strategy(strategy_name: str, param_names: list):
    config_manager.register_strategy('compress', strategy_name, param_names)
