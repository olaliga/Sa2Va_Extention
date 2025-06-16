from typing import Dict, List, Optional, Type, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

from .base_vlm_configs import BaseVisionConfig, BaseLLMConfig

logger = logging.get_logger(__name__)

class VLMConfigRegistry:
    """VLM 配置註冊表
    
    用於管理視覺語言模型的配置類註冊。實現單例模式，確保全局只有一個註冊表實例。
    
    Attributes:
        _instance: 單例實例
        _vision_configs: 視覺模型配置類註冊表
        _llm_configs: 語言模型配置類註冊表
        _vlm_configs: VLM 配置類註冊表
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._vision_configs: Dict[str, Type[BaseVisionConfig]] = {}
        self._llm_configs: Dict[str, Type[BaseLLMConfig]] = {}
        self._vlm_configs: Dict[str, Type[PretrainedConfig]] = {}
        self._initialized = True
    
    def register_vision_config(
        self,
        config_type: str,
        config_class: Type[BaseVisionConfig]
    ) -> None:
        """註冊視覺模型配置類
        
        Args:
            config_type: 配置類型標識符
            config_class: 配置類
        """
        if not issubclass(config_class, BaseVisionConfig):
            raise ValueError(
                f"config_class must be a subclass of BaseVisionConfig, "
                f"got {config_class}"
            )
        
        if config_type in self._vision_configs:
            logger.warning(
                f"Overwriting existing vision config registration: {config_type}"
            )
        
        self._vision_configs[config_type] = config_class
        logger.info(f"Registered vision config: {config_type}")
    
    def register_llm_config(
        self,
        config_type: str,
        config_class: Type[BaseLLMConfig]
    ) -> None:
        """註冊語言模型配置類
        
        Args:
            config_type: 配置類型標識符
            config_class: 配置類
        """
        if not issubclass(config_class, BaseLLMConfig):
            raise ValueError(
                f"config_class must be a subclass of BaseLLMConfig, "
                f"got {config_class}"
            )
        
        if config_type in self._llm_configs:
            logger.warning(
                f"Overwriting existing LLM config registration: {config_type}"
            )
        
        self._llm_configs[config_type] = config_class
        logger.info(f"Registered LLM config: {config_type}")
    
    def register_vlm_config(
        self,
        config_type: str,
        config_class: Type[PretrainedConfig]
    ) -> None:
        """註冊 VLM 配置類
        
        Args:
            config_type: 配置類型標識符
            config_class: 配置類
        """
        if not issubclass(config_class, PretrainedConfig):
            raise ValueError(
                f"config_class must be a subclass of PretrainedConfig, "
                f"got {config_class}"
            )
        
        if config_type in self._vlm_configs:
            logger.warning(
                f"Overwriting existing VLM config registration: {config_type}"
            )
        
        self._vlm_configs[config_type] = config_class
        logger.info(f"Registered VLM config: {config_type}")
    
    def get_vision_config_class(
        self,
        config_type: str
    ) -> Optional[Type[BaseVisionConfig]]:
        """獲取視覺模型配置類
        
        Args:
            config_type: 配置類型標識符
        
        Returns:
            配置類，如果未註冊則返回 None
        """
        return self._vision_configs.get(config_type)
    
    def get_llm_config_class(
        self,
        config_type: str
    ) -> Optional[Type[BaseLLMConfig]]:
        """獲取語言模型配置類
        
        Args:
            config_type: 配置類型標識符
        
        Returns:
            配置類，如果未註冊則返回 None
        """
        return self._llm_configs.get(config_type)
    
    def get_vlm_config_class(
        self,
        config_type: str
    ) -> Optional[Type[PretrainedConfig]]:
        """獲取 VLM 配置類
        
        Args:
            config_type: 配置類型標識符
        
        Returns:
            配置類，如果未註冊則返回 None
        """
        return self._vlm_configs.get(config_type)
    
    def list_vision_configs(self) -> List[str]:
        """列出所有註冊的視覺模型配置類型
        
        Returns:
            配置類型列表
        """
        return list(self._vision_configs.keys())
    
    def list_llm_configs(self) -> List[str]:
        """列出所有註冊的語言模型配置類型
        
        Returns:
            配置類型列表
        """
        return list(self._llm_configs.keys())
    
    def list_vlm_configs(self) -> List[str]:
        """列出所有註冊的 VLM 配置類型
        
        Returns:
            配置類型列表
        """
        return list(self._vlm_configs.keys())
    
    def unregister_vision_config(self, config_type: str) -> None:
        """取消註冊視覺模型配置類
        
        Args:
            config_type: 配置類型標識符
        """
        if config_type in self._vision_configs:
            del self._vision_configs[config_type]
            logger.info(f"Unregistered vision config: {config_type}")
    
    def unregister_llm_config(self, config_type: str) -> None:
        """取消註冊語言模型配置類
        
        Args:
            config_type: 配置類型標識符
        """
        if config_type in self._llm_configs:
            del self._llm_configs[config_type]
            logger.info(f"Unregistered LLM config: {config_type}")
    
    def unregister_vlm_config(self, config_type: str) -> None:
        """取消註冊 VLM 配置類
        
        Args:
            config_type: 配置類型標識符
        """
        if config_type in self._vlm_configs:
            del self._vlm_configs[config_type]
            logger.info(f"Unregistered VLM config: {config_type}")
    
    def clear(self) -> None:
        """清除所有註冊的配置類"""
        self._vision_configs.clear()
        self._llm_configs.clear()
        self._vlm_configs.clear()
        logger.info("Cleared all config registrations") 