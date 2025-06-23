from typing import Dict, Optional, Type, Union, Any
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .base_vlm_configs import BaseVisionConfig, BaseLLMConfig
from .vlm_registry import VLMConfigRegistry

logger = logging.get_logger(__name__)



class VLMConfigFactory:
    """VLM 配置工廠
    
    用於創建各種類型的配置實例。實現工廠模式，根據配置類型創建對應的配置對象。
    
    Attributes:
        registry: 配置註冊表實例
    """
    def __init__(self):
        self.registry = VLMConfigRegistry()

    def create_vision_config(
        self,
        config_type: str,
        **kwargs
    ) -> Optional[BaseVisionConfig]:
        """創建視覺模型配置實例
        
        Args:
            config_type: 配置類型標識符
            **kwargs: 配置參數
        
        Returns:
            配置實例，如果配置類型未註冊則返回 None
        
        Raises:
            ValueError: 如果配置類型未註冊
        """
        config_class = self.registry.get_vision_config_class(config_type)
        if config_class is None:
            raise ValueError(
                f"Unknown vision config type: {config_type}. "
                f"Available types: {self.registry.list_vision_configs()}"
            )
        
        try:
            return config_class(**kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create vision config of type {config_type}: {str(e)}"
            )
            raise
    
    def create_llm_config(
        self,
        config_type: str,
        **kwargs
    ) -> Optional[BaseLLMConfig]:
        """創建語言模型配置實例
        
        Args:
            config_type: 配置類型標識符
            **kwargs: 配置參數
        
        Returns:
            配置實例，如果配置類型未註冊則返回 None
        
        Raises:
            ValueError: 如果配置類型未註冊
        """
        config_class = self.registry.get_llm_config_class(config_type)
        if config_class is None:
            raise ValueError(
                f"Unknown LLM config type: {config_type}. "
                f"Available types: {self.registry.list_llm_configs()}"
            )
        
        try:
            return config_class(**kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create LLM config of type {config_type}: {str(e)}"
            )
            raise
    
    def create_vlm_config(
        self,
        config_type: str,
        **kwargs
    ) -> Optional[PretrainedConfig]:
        """創建 VLM 配置實例
        
        Args:
            config_type: 配置類型標識符
            **kwargs: 配置參數
        
        Returns:
            配置實例，如果配置類型未註冊則返回 None
        
        Raises:
            ValueError: 如果配置類型未註冊
        """
        logger.info(f"Creating VLM config of type: {config_type}")
        config_class = self.registry.get_vlm_config_class(config_type)
        if config_class is None:
            raise ValueError(
                f"Unknown VLM config type: {config_type}. "
                f"Available types: {self.registry.list_vlm_configs()}"
            )
        
        try:
            return config_class(**kwargs)
        except Exception as e:
            logger.error(
                f"Failed to create VLM config of type {config_type}: {str(e)}"
            )
            raise
    
    def create_config_from_dict(
        self,
        config_dict: Dict[str, Any]
    ) -> Optional[PretrainedConfig]:
        """從字典創建配置實例
        
        Args:
            config_dict: 配置字典，必須包含 "model_type" 字段
        
        Returns:
            配置實例，如果配置類型未註冊則返回 None
        
        Raises:
            ValueError: 如果配置字典無效或配置類型未註冊
        """
        if not isinstance(config_dict, dict):
            raise ValueError("config_dict must be a dictionary")
        
        model_type = config_dict.get("model_type")
        if not model_type:
            raise ValueError("config_dict must contain 'model_type' field")
        
        # 嘗試創建 VLM 配置
        try:
            return self.create_vlm_config(model_type, **config_dict)
        except ValueError:
            pass
        
        # 嘗試創建視覺模型配置
        try:
            return self.create_vision_config(model_type, **config_dict)
        except ValueError:
            pass
        
        # 嘗試創建語言模型配置
        try:
            return self.create_llm_config(model_type, **config_dict)
        except ValueError:
            pass
        
        # 如果都失敗了，拋出錯誤
        available_types = (
            self.registry.list_vlm_configs() +
            self.registry.list_vision_configs() +
            self.registry.list_llm_configs()
        )
        raise ValueError(
            f"Unknown config type: {model_type}. "
            f"Available types: {available_types}"
        )
    
    def create_config_from_pretrained(
        self,
        pretrained_model_name_or_path: str,
        **kwargs
    ) -> Optional[PretrainedConfig]:
        """從預訓練模型創建配置實例
        
        Args:
            pretrained_model_name_or_path: 預訓練模型名稱或路徑
            **kwargs: 其他參數
        
        Returns:
            配置實例，如果配置類型未註冊則返回 None
        
        Raises:
            ValueError: 如果配置類型未註冊或加載失敗
        """
        try:
            # 嘗試加載配置
            config = PretrainedConfig.from_pretrained(
                pretrained_model_name_or_path,
                **kwargs
            )
            
            # 根據配置類型創建對應的配置實例
            return self.create_config_from_dict(config.to_dict())
        except Exception as e:
            logger.error(
                f"Failed to load config from {pretrained_model_name_or_path}: "
                f"{str(e)}"
            )
            raise 