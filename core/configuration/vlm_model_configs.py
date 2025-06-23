from typing import Dict, Optional, Type, Union, Any
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
import copy

from .base_vlm_configs import BaseVisionConfig, BaseLLMConfig
from .vlm_registry import VLMConfigRegistry
from .vlm_factory import VLMConfigFactory

logger = logging.get_logger(__name__)


# ------------------- 默認配置定義 (第一步) -------------------

# 所有模型共用的默認視覺配置
DEFAULT_VISION_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "image_size": 336,
    "patch_size": 14,
    "use_flash_attn": True
}

# LLaVA 模型的默認 LLM 配置
DEFAULT_LLAVA_LLM_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 32000
}

# Qwen-VL 模型的默認 LLM 配置
DEFAULT_QWEN_VL_LLM_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 151936
}

# Intern-VL 模型的默認 LLM 配置
DEFAULT_INTERN_VL_LLM_CONFIG = {
    "architectures": ["InternLM2ForCausalLM"],
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 32000
}

# Phi-3 模型的默認 LLM 配置
DEFAULT_PHI3_LLM_CONFIG = {
    "architectures": ["PhiForCausalLM"],
    "hidden_size": 3072,
    "intermediate_size": 8192,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "vocab_size": 32064,
    "max_position_embeddings": 4096,
    "rope_scaling": {"type": "linear", "factor": 2.0}
}


# 將所有默認配置組合到一個總字典中，方便註冊
VLM_DEFAULT_CONFIGS = {
    "llava": {
        "vision_config": DEFAULT_VISION_CONFIG,
        "llm_config": DEFAULT_LLAVA_LLM_CONFIG
    },
    "qwen-vl": {
        "vision_config": DEFAULT_VISION_CONFIG,
        "llm_config": DEFAULT_QWEN_VL_LLM_CONFIG
    },
    "intern-vl": {
        "vision_config": DEFAULT_VISION_CONFIG,
        "llm_config": DEFAULT_INTERN_VL_LLM_CONFIG
    },
    "phi3": {
        "vision_config": DEFAULT_VISION_CONFIG,
        "llm_config": DEFAULT_PHI3_LLM_CONFIG
    }
}



def deep_merge_dict(default: dict, custom: dict) -> dict:
    """
    深度合併兩個字典。
    `custom` 字典中的值會覆蓋 `default` 字典中的值。
    """
    merged = copy.deepcopy(default)
    for key, value in custom.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


class BaseVLMConfig(PretrainedConfig):
    """VLM 配置基類
    
    所有具體的 VLM 配置類都應該繼承這個基類。
    它提供了基本的配置驗證和模型特定配置的驗證。
    """
    model_type = "base_vlm"
    
    # 支持的語言模型架構
    SUPPORTED_LLM_ARCHITECTURES = {
        "llava": ["LlamaForCausalLM"],
        "qwen-vl": ["Qwen2ForCausalLM"],
        "intern-vl": ["InternLM2ForCausalLM"],
        "phi3": ["PhiForCausalLM"]
    }
    
    def __init__(
        self,
        vlm_type: str,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vlm_type = vlm_type

        # 步驟 1: 從註冊表獲取默認配置
        registry = VLMConfigRegistry()
        default_config = registry.get_default_config(self.vlm_type)
        if default_config is None:
            raise ValueError(
                f"No default config registered for VLM type: {self.vlm_type}. "
                f"Available types: {list(registry._default_configs.keys())}"
            )

        # 步驟 2: 深度合併配置
        # 將用戶傳入的配置覆蓋到默認配置上
        vision_config_final = deep_merge_dict(
            default_config.get("vision_config", {}),
            vision_config or {}
        )
        llm_config_final = deep_merge_dict(
            default_config.get("llm_config", {}),
            llm_config or {}
        )
        
        self.vision_config = vision_config_final
        self.llm_config = llm_config_final
        
        # 步驟 3: 現在可以安全地進行驗證
        self.validate_config()
    
    def validate_config(self):
        """驗證基本配置
        
        驗證順序：
        1. 基本配置（類型檢查等）
        2. 語言模型配置（包括架構）
        3. 視覺配置
        """
        # 1. 驗證基本配置
        if self.vlm_type not in ["llava", "qwen-vl", "intern-vl", "phi3"]:
            raise ValueError(f"Unsupported VLM type: {self.vlm_type}")
        
        if not isinstance(self.vision_config, dict):
            raise ValueError("vision_config must be a dictionary")
        
        if not isinstance(self.llm_config, dict):
            raise ValueError("llm_config must be a dictionary")
        
        # 2. 驗證語言模型配置
        if "architectures" not in self.llm_config:
            raise ValueError("llm_config must specify architectures")
        
        # 驗證架構是否支持
        architecture = self.llm_config["architectures"][0]
        if architecture not in self.SUPPORTED_LLM_ARCHITECTURES.get(self.vlm_type, []):
            raise ValueError(f"Unsupported LLM architecture: {architecture}")
        
        self._validate_llm_config()
        
        # 3. 驗證視覺配置
        self._validate_vision_config()
    
    def _validate_vision_config(self):
        """驗證具體的視覺配置，由子類實現"""
        raise NotImplementedError
    
    def _validate_llm_config(self):
        """驗證具體的語言模型配置，由子類實現"""
        raise NotImplementedError

class LLaVAConfig(BaseVLMConfig):
    """LLaVA 配置類
    
    實現了 LLaVA 模型的特定配置和驗證邏輯。
    """
    model_type = "llava"
    
    def __init__(
        self,
        vision_config: Dict = None,
        llm_config: Dict = None,
        **kwargs
    ):
        # 移除 kwargs 中可能存在的 vlm_type，避免重複傳遞
        kwargs.pop('vlm_type', None)
        super().__init__(
            vlm_type="llava",
            vision_config=vision_config,
            llm_config=llm_config,
            **kwargs
        )
    
    def _validate_vision_config(self):
        """驗證 LLaVA 的視覺配置"""
        required_keys = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for key in required_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required key '{key}' in vision_config")
    
    def _validate_llm_config(self):
        """驗證 LLaVA 的語言模型配置"""
        # 架構驗證已經在基類中完成
        pass

class QwenVLConfig(BaseVLMConfig):
    """Qwen-VL 配置類
    
    實現了 Qwen-VL 模型的特定配置和驗證邏輯。
    """
    model_type = "qwen-vl"
    
    # Qwen-VL 特定的語言模型配置要求
    REQUIRED_LLM_KEYS = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size"
    ]
    
    def __init__(
        self,
        vision_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        **kwargs
    ):
        # 移除 kwargs 中可能存在的 vlm_type，避免重複傳遞
        kwargs.pop('vlm_type', None)
        super().__init__(
            vlm_type="qwen-vl",
            vision_config=vision_config,
            llm_config=llm_config,
            **kwargs
        )
    
    def _validate_vision_config(self):
        """驗證 Qwen-VL 的視覺配置"""
        required_keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads"
        ]
        for key in required_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required key '{key}' in vision_config")
    
    def _validate_llm_config(self):
        """驗證 Qwen-VL 的語言模型配置
        
        驗證要求：
        1. 架構必須是 Qwen2ForCausalLM（由基類驗證）
        2. 必須包含所有必需的配置鍵
        3. 配置值必須符合 Qwen-VL 的要求
        """
        # 驗證必需的配置鍵
        for key in self.REQUIRED_LLM_KEYS:
            if key not in self.llm_config:
                raise ValueError(f"Missing required key '{key}' in llm_config for Qwen-VL")
        
        # 驗證配置值的有效性
        if self.llm_config["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if self.llm_config["num_hidden_layers"] <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.llm_config["num_attention_heads"] <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.llm_config["intermediate_size"] <= 0:
            raise ValueError("intermediate_size must be positive")
        
        # 驗證 hidden_size 是否能被 num_attention_heads 整除
        if self.llm_config["hidden_size"] % self.llm_config["num_attention_heads"] != 0:
            raise ValueError(
                f"hidden_size {self.llm_config['hidden_size']} must be divisible by "
                f"num_attention_heads {self.llm_config['num_attention_heads']}"
            )

class InternVLConfig(BaseVLMConfig):
    """Intern-VL 配置類
    
    實現了 Intern-VL 模型的特定配置和驗證邏輯。
    """
    model_type = "intern-vl"
    
    # Intern-VL 特定的語言模型配置要求
    REQUIRED_LLM_KEYS = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size"
    ]
    
    def __init__(
        self,
        vision_config: Optional[Dict] = None,
        llm_config: Optional[Dict] = None,
        **kwargs
    ):
        # 移除 kwargs 中可能存在的 vlm_type，避免重複傳遞
        kwargs.pop('vlm_type', None)
        super().__init__(
            vlm_type="intern-vl",
            vision_config=vision_config,
            llm_config=llm_config,
            **kwargs
        )
    
    def _validate_vision_config(self):
        """驗證 Intern-VL 的視覺配置"""
        required_keys = [
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "use_flash_attn"
        ]
        for key in required_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required key '{key}' in vision_config")
        
        # 驗證 use_flash_attn 的類型
        if not isinstance(self.vision_config["use_flash_attn"], bool):
            raise ValueError("use_flash_attn must be a boolean")
    
    def _validate_llm_config(self):
        """驗證 Intern-VL 的語言模型配置
        
        驗證要求：
        1. 架構必須是 InternLM2ForCausalLM（由基類驗證）
        2. 必須包含所有必需的配置鍵
        3. 配置值必須符合 Intern-VL 的要求
        """
        # 驗證必需的配置鍵
        for key in self.REQUIRED_LLM_KEYS:
            if key not in self.llm_config:
                raise ValueError(f"Missing required key '{key}' in llm_config for Intern-VL")
        
        # 驗證配置值的有效性
        if self.llm_config["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if self.llm_config["num_hidden_layers"] <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.llm_config["num_attention_heads"] <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.llm_config["intermediate_size"] <= 0:
            raise ValueError("intermediate_size must be positive")
        
        # 驗證 hidden_size 是否能被 num_attention_heads 整除
        if self.llm_config["hidden_size"] % self.llm_config["num_attention_heads"] != 0:
            raise ValueError(
                f"hidden_size {self.llm_config['hidden_size']} must be divisible by "
                f"num_attention_heads {self.llm_config['num_attention_heads']}"
            )
        
        # 根據架構進行特定驗證
        architecture = self.llm_config["architectures"][0]
        if architecture != "InternLM2ForCausalLM":
            raise ValueError("InternVL only supports InternLM2ForCausalLM architecture")

class Phi3Config(BaseVLMConfig):
    """Phi-3 VLM 配置類
    
    實現了 Phi-3 模型的特定配置和驗證邏輯。
    Phi-3 是一個基於 Transformer 的視覺語言模型，具有以下特點：
    1. 支持 RoPE 縮放，可以動態調整位置編碼
    2. 使用 Flash Attention 進行高效的注意力計算
    3. 具有特定的詞彙表大小和位置編碼限制
    """
    model_type = "phi3"
    
    # Phi-3 特定的語言模型配置要求
    REQUIRED_LLM_KEYS = [
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "intermediate_size",
        "vocab_size",
        "max_position_embeddings"  # Phi-3 特有的必需參數
    ]
    
    def __init__(
        self,
        vision_config: Optional[Dict[str, Any]] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        # 移除 kwargs 中可能存在的 vlm_type，避免重複傳遞
        kwargs.pop('vlm_type', None)
        super().__init__(
            vlm_type="phi3",
            vision_config=vision_config,
            llm_config=llm_config,
            **kwargs
        )
    
    def _validate_vision_config(self) -> None:
        """驗證 Phi-3 的視覺配置
        
        驗證要求：
        1. 必須包含所有必需的配置鍵
        2. use_flash_attn 必須是布爾值（如果提供）
        """
        required_keys = ["hidden_size", "num_hidden_layers", "num_attention_heads"]
        for key in required_keys:
            if key not in self.vision_config:
                raise ValueError(f"Missing required key '{key}' in vision_config")
        
        # Phi-3 特定的視覺配置驗證
        if "use_flash_attn" in self.vision_config:
            if not isinstance(self.vision_config["use_flash_attn"], bool):
                raise ValueError("use_flash_attn must be a boolean")
    
    def _validate_llm_config(self) -> None:
        """驗證 Phi-3 的語言模型配置
        
        驗證要求：
        1. 架構必須是 PhiForCausalLM（由基類驗證）
        2. 必須包含所有必需的配置鍵
        3. 配置值必須符合 Phi-3 的要求
        4. rope_scaling 必須符合特定格式（如果提供）
        """
        # 驗證必需的配置鍵
        for key in self.REQUIRED_LLM_KEYS:
            if key not in self.llm_config:
                raise ValueError(f"Missing required key '{key}' in llm_config for Phi-3")
        
        # 驗證配置值的有效性
        if self.llm_config["hidden_size"] <= 0:
            raise ValueError("hidden_size must be positive")
        if self.llm_config["num_hidden_layers"] <= 0:
            raise ValueError("num_hidden_layers must be positive")
        if self.llm_config["num_attention_heads"] <= 0:
            raise ValueError("num_attention_heads must be positive")
        if self.llm_config["intermediate_size"] <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.llm_config["vocab_size"] <= 0:
            raise ValueError("vocab_size must be positive")
        if self.llm_config["max_position_embeddings"] <= 0:
            raise ValueError("max_position_embeddings must be positive")
        
        # 驗證 hidden_size 是否能被 num_attention_heads 整除
        if self.llm_config["hidden_size"] % self.llm_config["num_attention_heads"] != 0:
            raise ValueError(
                f"hidden_size {self.llm_config['hidden_size']} must be divisible by "
                f"num_attention_heads {self.llm_config['num_attention_heads']}"
            )
        
        # Phi-3 特有的 rope_scaling 驗證
        if "rope_scaling" in self.llm_config:
            rope_scaling = self.llm_config["rope_scaling"]
            if not isinstance(rope_scaling, dict):
                raise ValueError("rope_scaling must be a dictionary")
            if "type" not in rope_scaling or "factor" not in rope_scaling:
                raise ValueError("rope_scaling must specify 'type' and 'factor'")
            if rope_scaling["type"] not in ["linear", "dynamic"]:
                raise ValueError("rope_scaling type must be 'linear' or 'dynamic'")
            if not isinstance(rope_scaling["factor"], (int, float)) or rope_scaling["factor"] <= 0:
                raise ValueError("rope_scaling factor must be a positive number")

# ------------------- 默認配置註冊 (第四步) -------------------
# 將模型類型字符串映射到對應的配置類
VLM_CONFIG_MAPPING = {
    "llava": LLaVAConfig,
    "qwen-vl": QwenVLConfig,
    "intern-vl": InternVLConfig,
    "phi3": Phi3Config
}


def register_default_vlm_configs():
    """
    將所有在 VLM_DEFAULT_CONFIGS 中定義的默認配置和對應的類註冊到註冊表中。
    """
    registry = VLMConfigRegistry()
    
    # 註冊默認配置字典
    for model_type, config_dict in VLM_DEFAULT_CONFIGS.items():
        if registry.get_default_config(model_type) is None:
            registry.register_default_config(model_type, config_dict)
            
    # 註冊配置類
    for model_type, config_class in VLM_CONFIG_MAPPING.items():
        if registry.get_vlm_config_class(model_type) is None:
            registry.register_vlm_config(model_type, config_class)

# 在模塊加載時執行註冊
register_default_vlm_configs() 