from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union, Type, ClassVar
import json
import os
from abc import ABC, abstractmethod

class SegConfigRegistry:
    """
    分割模型配置註冊表
    用於管理和註冊不同類型的分割模型配置
    """
    _configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
    
    @classmethod
    def register(cls, model_type: str, default_config: Dict[str, Any]) -> None:
        """
        註冊新的分割模型配置
        
        Args:
            model_type (str): 模型類型
            default_config (Dict[str, Any]): 默認配置
        """
        cls._configs[model_type] = default_config
    
    @classmethod
    def get_default_config(cls, model_type: str) -> Dict[str, Any]:
        """
        獲取指定模型類型的默認配置
        
        Args:
            model_type (str): 模型類型
            
        Returns:
            Dict[str, Any]: 默認配置
            
        Raises:
            ValueError: 如果模型類型未註冊
        """
        if model_type not in cls._configs:
            raise ValueError(
                f"Model type '{model_type}' is not registered. "
                f"Available types: {list(cls._configs.keys())}"
            )
        return cls._configs[model_type]
    
    @classmethod
    def list_models(cls) -> list[str]:
        """
        列出所有已註冊的模型類型
        
        Returns:
            list[str]: 模型類型列表
        """
        return list(cls._configs.keys())

# 註冊默認的 SAM2 配置
SegConfigRegistry.register("sam2", {
    "image_encoder": {
        "type": "vit",
        "depth": 12,
        "embed_dim": 768,
        "num_heads": 12,
        "mlp_ratio": 4.0,
        "qkv_bias": True,
        "drop_rate": 0.0,
        "attn_drop_rate": 0.0,
        "drop_path_rate": 0.0,
        "norm_layer": "LayerNorm",
        "act_layer": "gelu",
        "use_abs_pos_emb": True,
        "use_rel_pos_bias": True,
        "window_size": 14,
        "global_attn_indexes": []
    },
    "prompt_encoder": {
        "embed_dim": 256,
        "image_embed_size": 1024,
        "input_image_size": 1024,
        "mask_in_chans": 16
    },
    "mask_decoder": {
        "transformer_dim": 256,
        "transformer_heads": 8,
        "transformer_depth": 2,
        "mlp_dim": 2048,
        "iou_head_depth": 3,
        "iou_head_hidden_dim": 256
    }
})

# 註冊其他分割模型的默認配置
SegConfigRegistry.register("other_seg", {
    "encoder": {
        "type": "resnet",
        "depth": 50
    },
    "decoder": {
        "type": "fpn",
        "in_channels": [256, 512, 1024, 2048]
    }
})

class BaseSegConfig(PretrainedConfig, ABC):
    """
    分割模型配置基類
    """
    model_type = "base_seg"
    
    def __init__(
        self,
        seg_type: str = "sam2",           # 提供默認值，允許無參數初始化
        seg_config: Optional[dict] = None, # 具體模型配置
        image_size: int = 1024,           # 圖像大小
        patch_size: int = 16,             # patch大小
        hidden_size: int = 768,           # 隱藏層大小
        num_hidden_layers: int = 12,      # 層數
        num_attention_heads: int = 12,    # 注意力頭數
        intermediate_size: int = 3072,    # 中間層大小
        hidden_act: str = "gelu",         # 激活函數
        initializer_range: float = 0.02,  # 初始化範圍
        layer_norm_eps: float = 1e-5,     # LayerNorm epsilon
        **kwargs
    ):
        super().__init__(**kwargs)
        self.seg_type = seg_type
        
        # 先驗證 seg_config 的類型
        if seg_config is not None and not isinstance(seg_config, dict):
            raise TypeError(f"seg_config must be a dictionary, got {type(seg_config)}")
        
        # 然後再進行合併
        if seg_config is None:
            seg_config = {}
        try:
            default_config = SegConfigRegistry.get_default_config(self.seg_type)
            self.seg_config = {**default_config, **seg_config}
        except ValueError as e:
            raise ValueError(
                f"Failed to initialize {self.seg_type} config: {str(e)}"
            )
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        
        # 驗證配置
        self.validate_config()
    
    @abstractmethod
    def _validate_custom_config(self) -> None:
        """
        驗證特定模型類型的自定義配置
        子類必須實現此方法
        """
        pass
    
    def validate_config(self) -> None:
        """
        驗證配置參數的有效性
        """
        # 驗證模型類型
        if self.seg_type not in SegConfigRegistry.list_models():
            raise ValueError(
                f"Invalid seg_type: {self.seg_type}. "
                f"Must be one of {SegConfigRegistry.list_models()}"
            )
        
        # 驗證基本參數
        self._validate_basic_params()
        
        # 驗證自定義配置
        self._validate_custom_config()
    
    def _validate_basic_params(self) -> None:
        """
        驗證基本配置參數
        
        驗證順序：
        1. 先驗證 patch_size，因為它是其他驗證的基礎
        2. 再驗證 image_size
        3. 然後驗證它們的整除關係
        4. 最後驗證其他參數
        """
        # 1. 驗證 patch 大小（必須先驗證）
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        
        # 2. 驗證圖像大小
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        
        # 3. 驗證圖像大小和 patch 大小的整除關係
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"patch_size {self.patch_size}"
            )
        
        # 4. 驗證隱藏層大小
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        # 5. 驗證注意力頭數
        if self.num_attention_heads <= 0:
            raise ValueError(
                f"num_attention_heads must be positive, "
                f"got {self.num_attention_heads}"
            )
        
        # 6. 驗證隱藏層大小和注意力頭數的整除關係
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        
        # 7. 驗證層數
        if self.num_hidden_layers <= 0:
            raise ValueError(
                f"num_hidden_layers must be positive, "
                f"got {self.num_hidden_layers}"
            )
        
        # 8. 驗證中間層大小
        if self.intermediate_size <= 0:
            raise ValueError(
                f"intermediate_size must be positive, "
                f"got {self.intermediate_size}"
            )
        
        # 9. 驗證激活函數
        valid_acts = ["gelu", "relu", "silu", "mish"]
        if self.hidden_act not in valid_acts:
            raise ValueError(
                f"Invalid hidden_act: {self.hidden_act}. "
                f"Must be one of {valid_acts}"
            )
        
        # 10. 驗證初始化範圍
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, "
                f"got {self.initializer_range}"
            )
        
        # 11. 驗證 LayerNorm epsilon
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be positive, "
                f"got {self.layer_norm_eps}"
            )

    @classmethod
    def _get_non_default_generation_parameters(cls) -> Dict[str, Any]:
        """
        重寫 transformers 的默認參數獲取方法
        
        由於我們的配置類需要 seg_type 參數，不能無參數實例化，
        所以直接返回空字典，避免創建默認實例。
        
        Returns:
            Dict[str, Any]: 空字典，表示沒有默認生成參數
        """
        return {}

    def to_diff_dict(self) -> Dict[str, Any]:
        """
        重寫 transformers 的差異字典生成方法
        
        由於我們的配置類需要 seg_type 參數，不能無參數實例化，
        所以直接返回當前配置的完整字典。
        
        Returns:
            Dict[str, Any]: 當前配置的完整字典
        """
        return self.to_dict()

class SAM2Config(BaseSegConfig):
    """
    SAM2 模型配置類
    """
    model_type = "sam2"
    
    def _validate_custom_config(self) -> None:
        """
        驗證 SAM2 特定的配置
        """
        if not isinstance(self.seg_config, dict):
            raise TypeError(
                f"seg_config must be a dictionary, got {type(self.seg_config)}"
            )
        
        required_keys = ["image_encoder", "prompt_encoder", "mask_decoder"]
        for key in required_keys:
            if key not in self.seg_config:
                raise ValueError(
                    f"Missing required key '{key}' in seg_config for SAM2"
                )

class OtherSegConfig(BaseSegConfig):
    """
    其他分割模型配置類
    """
    model_type = "other_seg"
    
    def _validate_custom_config(self) -> None:
        """
        驗證其他分割模型特定的配置
        """
        if not isinstance(self.seg_config, dict):
            raise TypeError(
                f"seg_config must be a dictionary, got {type(self.seg_config)}"
            )
        
        required_keys = ["encoder", "decoder"]
        for key in required_keys:
            if key not in self.seg_config:
                raise ValueError(
                    f"Missing required key '{key}' in seg_config for other segmentation model"
                )

class SegConfigFactory:
    """
    分割模型配置工廠類
    """
    _config_classes: ClassVar[Dict[str, Type[BaseSegConfig]]] = {
        "sam2": SAM2Config,
        "other_seg": OtherSegConfig
    }
    
    @classmethod
    def register_config(cls, model_type: str, config_class: Type[BaseSegConfig]) -> None:
        """
        註冊新的配置類
        
        Args:
            model_type (str): 模型類型
            config_class (Type[BaseSegConfig]): 配置類
        """
        if not issubclass(config_class, BaseSegConfig):
            raise TypeError(
                f"Config class must be a subclass of BaseSegConfig, "
                f"got {config_class}"
            )
        cls._config_classes[model_type] = config_class
    
    @classmethod
    def create_config(cls, model_type: str, **kwargs) -> BaseSegConfig:
        """
        創建指定類型的配置實例
        
        Args:
            model_type (str): 模型類型
            **kwargs: 配置參數
            
        Returns:
            BaseSegConfig: 配置實例
            
        Raises:
            ValueError: 如果模型類型未註冊
        """
        if model_type not in cls._config_classes:
            raise ValueError(
                f"Model type '{model_type}' is not registered. "
                f"Available types: {list(cls._config_classes.keys())}"
            )
        return cls._config_classes[model_type](seg_type=model_type, **kwargs)

def create_seg_config(*args, **kwargs) -> BaseSegConfig:
    """
    創建分割模型配置的工廠函數
    
    Args:
        *args: 位置參數
        **kwargs: 關鍵字參數，必須包含 seg_type 或使用默認值 "sam2"
        
    Returns:
        BaseSegConfig: 分割模型配置實例
        
    Examples:
        >>> # 創建 SAM2 配置
        >>> config = create_seg_config(seg_type="sam2")
        >>> # 創建自定義配置
        >>> config = create_seg_config(
        ...     seg_type="sam2",
        ...     seg_config={"image_encoder": {...}}
        ... )
    """
    # 從 kwargs 中取出 seg_type，並移除可能導致重複傳遞的參數
    model_type = kwargs.pop("seg_type", "sam2")
    kwargs.pop("model_type", None)  # 移除可能存在的 model_type，避免重複傳遞
    
    return SegConfigFactory.create_config(model_type, **kwargs)

