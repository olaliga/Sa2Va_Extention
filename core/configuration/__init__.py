"""VLM 配置系統

此模塊提供了視覺語言模型（VLM）的配置管理系統，包括：
1. 基礎配置類：用於定義視覺和語言模型的基本配置
2. 配置註冊表：管理配置類的註冊和檢索
3. 配置工廠：創建各種類型的配置實例
4. 具體模型配置：實現各種 VLM 模型的特定配置
"""

from .base_vlm_configs import BaseVisionConfig, BaseLLMConfig
from .vlm_registry import VLMConfigRegistry
from .vlm_factory import VLMConfigFactory
from .vlm_model_configs import (
    BaseVLMConfig,  # VLM 配置基類
    LLaVAConfig,    # LLaVA 模型配置
    QwenVLConfig,   # Qwen-VL 模型配置
    InternVLConfig, # Intern-VL 模型配置
    Phi3Config,     # Phi-3 模型配置
)

"""分割模型配置系統
"""
from .seg_config import (
    # 分割模型配置基類
    BaseSegConfig,  # 抽象基類，定義配置接口
    # 分割模型配置工廠函數
    create_seg_config,  # 創建分割模型配置的工廠函數
    # 分割模型配置註冊表
    SegConfigRegistry,  # 管理不同分割模型的默認配置
    # 分割模型配置工廠類
    SegConfigFactory,  # 創建具體配置實例的工廠類
    # 具體分割模型配置類
    SAM2Config,  # SAM2 模型配置類
    OtherSegConfig,  # 其他分割模型配置類
)
# from .sa2va_config import Sa2VAConfig

__all__ = [
    # VLM 基礎配置
    "BaseVisionConfig",
    "BaseLLMConfig",
    "VLMConfigRegistry",
    "VLMConfigFactory",
    
    # VLM 具體模型配置
    "BaseVLMConfig",    # VLM 配置基類
    "LLaVAConfig",      # LLaVA 模型配置
    "QwenVLConfig",     # Qwen-VL 模型配置
    "InternVLConfig",   # Intern-VL 模型配置
    "Phi3Config",       # Phi-3 模型配置
    
    # 分割模型配置
    "BaseSegConfig",  # 分割模型配置抽象基類
    "create_seg_config",  # 分割模型配置工廠函數
    "SegConfigRegistry",  # 分割模型配置註冊表
    "SegConfigFactory",  # 分割模型配置工廠類
    "SAM2Config",  # SAM2 模型配置類
    "OtherSegConfig",  # 其他分割模型配置類
    
    # Sa2VA 配置
    # "Sa2VAConfig",  # Sa2VA 模型配置類
]
