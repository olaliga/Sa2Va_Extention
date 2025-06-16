from typing import Dict, List, Optional, Union
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class BaseVisionConfig(PretrainedConfig):
    """視覺編碼器配置基類
    
    用於定義視覺編碼器的基本配置參數和驗證邏輯。
    
    Args:
        hidden_size (int): 隱藏層大小
        num_hidden_layers (int): 隱藏層數量
        num_attention_heads (int): 注意力頭數量
        intermediate_size (int): 中間層大小
        image_size (int): 輸入圖像大小
        patch_size (int): 圖像塊大小
        hidden_act (str, optional): 激活函數類型，默認為 "gelu"
        layer_norm_eps (float, optional): LayerNorm epsilon，默認為 1e-6
        dropout (float, optional): Dropout 比率，默認為 0.0
        attention_dropout (float, optional): 注意力 Dropout 比率，默認為 0.0
        initializer_range (float, optional): 初始化範圍，默認為 0.02
        **kwargs: 其他配置參數
    """
    model_type = "base_vision"
    
    def __init__(
        self,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        image_size: int,
        patch_size: int,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1e-6,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 基本參數
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.image_size = image_size
        self.patch_size = patch_size
        
        # 其他參數
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        
        # 驗證配置
        self.validate_config()
    
    def validate_config(self):
        """驗證配置參數的有效性"""
        # 驗證隱藏層大小
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        # 驗證層數
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        
        # 驗證注意力頭數
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        
        # 驗證中間層大小
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {self.intermediate_size}")
        
        # 驗證圖像大小
        if self.image_size <= 0:
            raise ValueError(f"image_size must be positive, got {self.image_size}")
        
        # 驗證圖像塊大小
        if self.patch_size <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")
        if self.image_size % self.patch_size != 0:
            raise ValueError(
                f"image_size {self.image_size} must be divisible by "
                f"patch_size {self.patch_size}"
            )
        
        # 驗證激活函數
        valid_acts = ["gelu", "relu", "silu", "mish"]
        if self.hidden_act not in valid_acts:
            raise ValueError(
                f"Invalid hidden_act: {self.hidden_act}. "
                f"Must be one of {valid_acts}"
            )
        
        # 驗證 LayerNorm epsilon
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be positive, "
                f"got {self.layer_norm_eps}"
            )
        
        # 驗證 Dropout 比率
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError(
                f"attention_dropout must be between 0 and 1, "
                f"got {self.attention_dropout}"
            )
        
        # 驗證初始化範圍
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, "
                f"got {self.initializer_range}"
            )

class BaseLLMConfig(PretrainedConfig):
    """語言模型配置基類
    
    用於定義語言模型的基本配置參數和驗證邏輯。
    
    Args:
        vocab_size (int): 詞表大小
        hidden_size (int): 隱藏層大小
        num_hidden_layers (int): 隱藏層數量
        num_attention_heads (int): 注意力頭數量
        intermediate_size (int): 中間層大小
        architectures (List[str]): 模型架構列表
        hidden_act (str, optional): 激活函數類型，默認為 "silu"
        max_position_embeddings (int, optional): 最大位置編碼長度，默認為 2048
        layer_norm_eps (float, optional): LayerNorm epsilon，默認為 1e-5
        dropout (float, optional): Dropout 比率，默認為 0.0
        attention_dropout (float, optional): 注意力 Dropout 比率，默認為 0.0
        initializer_range (float, optional): 初始化範圍，默認為 0.02
        **kwargs: 其他配置參數
    """
    model_type = "base_llm"
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_hidden_layers: int,
        num_attention_heads: int,
        intermediate_size: int,
        architectures: List[str],
        hidden_act: str = "silu",
        max_position_embeddings: int = 2048,
        layer_norm_eps: float = 1e-5,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        initializer_range: float = 0.02,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        # 基本參數
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.architectures = architectures
        
        # 其他參數
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.initializer_range = initializer_range
        
        # 驗證配置
        self.validate_config()
    
    def validate_config(self):
        """驗證配置參數的有效性"""
        # 驗證詞表大小
        if self.vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive, got {self.vocab_size}")
        
        # 驗證隱藏層大小
        if self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {self.hidden_size}")
        
        # 驗證層數
        if self.num_hidden_layers <= 0:
            raise ValueError(f"num_hidden_layers must be positive, got {self.num_hidden_layers}")
        
        # 驗證注意力頭數
        if self.num_attention_heads <= 0:
            raise ValueError(f"num_attention_heads must be positive, got {self.num_attention_heads}")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} must be divisible by "
                f"num_attention_heads {self.num_attention_heads}"
            )
        
        # 驗證中間層大小
        if self.intermediate_size <= 0:
            raise ValueError(f"intermediate_size must be positive, got {self.intermediate_size}")
        
        # 驗證模型架構
        if not isinstance(self.architectures, list) or not self.architectures:
            raise ValueError("architectures must be a non-empty list")
        valid_architectures = [
            "LlamaForCausalLM",
            "InternLM2ForCausalLM",
            "Phi3ForCausalLM",
            "Qwen2ForCausalLM"
        ]
        for arch in self.architectures:
            if arch not in valid_architectures:
                raise ValueError(
                    f"Invalid architecture: {arch}. "
                    f"Must be one of {valid_architectures}"
                )
        
        # 驗證激活函數
        valid_acts = ["silu", "gelu", "relu", "mish"]
        if self.hidden_act not in valid_acts:
            raise ValueError(
                f"Invalid hidden_act: {self.hidden_act}. "
                f"Must be one of {valid_acts}"
            )
        
        # 驗證位置編碼
        if self.max_position_embeddings <= 0:
            raise ValueError(
                f"max_position_embeddings must be positive, "
                f"got {self.max_position_embeddings}"
            )
        
        # 驗證 LayerNorm epsilon
        if self.layer_norm_eps <= 0:
            raise ValueError(
                f"layer_norm_eps must be positive, "
                f"got {self.layer_norm_eps}"
            )
        
        # 驗證 Dropout 比率
        if not 0 <= self.dropout <= 1:
            raise ValueError(f"dropout must be between 0 and 1, got {self.dropout}")
        if not 0 <= self.attention_dropout <= 1:
            raise ValueError(
                f"attention_dropout must be between 0 and 1, "
                f"got {self.attention_dropout}"
            )
        
        # 驗證初始化範圍
        if self.initializer_range <= 0:
            raise ValueError(
                f"initializer_range must be positive, "
                f"got {self.initializer_range}"
            ) 