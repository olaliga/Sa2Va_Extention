from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union
import json
import os
from .vlm_config import BaseVLMConfig
from .seg_config import BaseSegConfig

class Sa2VAConfig(PretrainedConfig):
    """
    Sa2VA 模型配置類
    """
    model_type = "sa2va"
    
    def __init__(
        self,
        vlm_config: Optional[BaseVLMConfig] = None,  # VLM 模型配置
        seg_config: Optional[BaseSegConfig] = None,  # 分割模型配置
        fusion_strategy: str = "attention",          # 特徵融合策略
        fusion_config: Dict[str, Any] = None,        # 融合策略配置
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vlm_config = vlm_config or BaseVLMConfig()
        self.seg_config = seg_config or BaseSegConfig()
        self.fusion_strategy = fusion_strategy
        self.fusion_config = fusion_config or {}
        
        # 驗證配置
        self.validate_config()
    
    def validate_config(self) -> None:
        """
        驗證配置參數的有效性
        """
        # 驗證 VLM 配置
        if not isinstance(self.vlm_config, BaseVLMConfig):
            raise TypeError(
                f"vlm_config must be an instance of BaseVLMConfig, "
                f"got {type(self.vlm_config)}"
            )
        
        # 驗證分割模型配置
        if not isinstance(self.seg_config, BaseSegConfig):
            raise TypeError(
                f"seg_config must be an instance of BaseSegConfig, "
                f"got {type(self.seg_config)}"
            )
        
        # 驗證融合策略
        valid_fusion_strategies = ["attention", "concat", "add", "cross_attention"]
        if self.fusion_strategy not in valid_fusion_strategies:
            raise ValueError(
                f"Invalid fusion_strategy: {self.fusion_strategy}. "
                f"Must be one of {valid_fusion_strategies}"
            )
        
        # 驗證融合配置
        self._validate_fusion_config()
        
        # 驗證模型兼容性
        self._validate_model_compatibility()
    
    def _validate_fusion_config(self) -> None:
        """
        驗證融合策略配置的有效性
        """
        if not isinstance(self.fusion_config, dict):
            raise TypeError(
                f"fusion_config must be a dictionary, "
                f"got {type(self.fusion_config)}"
            )
        
        # 根據不同的融合策略驗證特定的配置
        if self.fusion_strategy == "attention":
            required_keys = ["num_heads", "dropout"]
            for key in required_keys:
                if key not in self.fusion_config:
                    raise ValueError(
                        f"Missing required key '{key}' in fusion_config for attention fusion"
                    )
            
            # 驗證注意力頭數
            if self.fusion_config["num_heads"] <= 0:
                raise ValueError(
                    f"num_heads must be positive, "
                    f"got {self.fusion_config['num_heads']}"
                )
            
            # 驗證 dropout
            if not 0 <= self.fusion_config["dropout"] <= 1:
                raise ValueError(
                    f"dropout must be between 0 and 1, "
                    f"got {self.fusion_config['dropout']}"
                )
        
        elif self.fusion_strategy == "concat":
            if "dim" not in self.fusion_config:
                raise ValueError(
                    "Missing required key 'dim' in fusion_config for concat fusion"
                )
        
        elif self.fusion_strategy == "cross_attention":
            required_keys = ["num_heads", "dropout", "cross_attention_layers"]
            for key in required_keys:
                if key not in self.fusion_config:
                    raise ValueError(
                        f"Missing required key '{key}' in fusion_config for cross attention fusion"
                    )
            
            # 驗證注意力頭數
            if self.fusion_config["num_heads"] <= 0:
                raise ValueError(
                    f"num_heads must be positive, "
                    f"got {self.fusion_config['num_heads']}"
                )
            
            # 驗證 dropout
            if not 0 <= self.fusion_config["dropout"] <= 1:
                raise ValueError(
                    f"dropout must be between 0 and 1, "
                    f"got {self.fusion_config['dropout']}"
                )
            
            # 驗證交叉注意力層數
            if self.fusion_config["cross_attention_layers"] <= 0:
                raise ValueError(
                    f"cross_attention_layers must be positive, "
                    f"got {self.fusion_config['cross_attention_layers']}"
                )
    
    def _validate_model_compatibility(self) -> None:
        """
        驗證 VLM 和分割模型的兼容性
        """
        # 檢查隱藏層大小是否匹配
        if self.vlm_config.hidden_size != self.seg_config.hidden_size:
            raise ValueError(
                f"VLM and segmentation model must have the same hidden_size. "
                f"Got {self.vlm_config.hidden_size} and {self.seg_config.hidden_size}"
            )
        
        # 檢查注意力頭數是否匹配
        if self.vlm_config.num_attention_heads != self.seg_config.num_attention_heads:
            raise ValueError(
                f"VLM and segmentation model must have the same num_attention_heads. "
                f"Got {self.vlm_config.num_attention_heads} and {self.seg_config.num_attention_heads}"
            )
        
        # 檢查中間層大小是否匹配
        if self.vlm_config.intermediate_size != self.seg_config.intermediate_size:
            raise ValueError(
                f"VLM and segmentation model must have the same intermediate_size. "
                f"Got {self.vlm_config.intermediate_size} and {self.seg_config.intermediate_size}"
            )
        
        # 檢查激活函數是否匹配
        if self.vlm_config.hidden_act != self.seg_config.hidden_act:
            raise ValueError(
                f"VLM and segmentation model must have the same hidden_act. "
                f"Got {self.vlm_config.hidden_act} and {self.seg_config.hidden_act}"
            )
        
        # 檢查 LayerNorm epsilon 是否匹配
        if self.vlm_config.layer_norm_eps != self.seg_config.layer_norm_eps:
            raise ValueError(
                f"VLM and segmentation model must have the same layer_norm_eps. "
                f"Got {self.vlm_config.layer_norm_eps} and {self.seg_config.layer_norm_eps}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """
        將配置轉換為字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        output = super().to_dict()
        
        # 序列化 VLM 配置
        if isinstance(self.vlm_config, BaseVLMConfig):
            output["vlm_config"] = self.vlm_config.to_dict()
        else:
            output["vlm_config"] = {}
        
        # 序列化分割模型配置
        if isinstance(self.seg_config, BaseSegConfig):
            output["seg_config"] = self.seg_config.to_dict()
        else:
            output["seg_config"] = {}
        
        # 序列化融合配置
        if isinstance(self.fusion_config, dict):
            output["fusion_config"] = self.fusion_config
        else:
            output["fusion_config"] = {}
        
        return output
    
    def to_json_string(self, use_diff: bool = True) -> str:
        """
        將配置轉換為 JSON 字符串
        
        Args:
            use_diff (bool, optional): 是否只輸出與默認值不同的配置項。默認為 True。
            
        Returns:
            str: JSON 字符串
        """
        if use_diff:
            config_dict = self.to_diff_dict()
        else:
            config_dict = self.to_dict()
        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Sa2VAConfig":
        """
        從字典創建配置
        
        Args:
            config_dict (Dict[str, Any]): 配置字典
            **kwargs: 其他參數
            
        Returns:
            Sa2VAConfig: 配置實例
        """
        # 處理 VLM 配置
        vlm_config_dict = config_dict.pop("vlm_config", {})
        if isinstance(vlm_config_dict, dict):
            vlm_config = BaseVLMConfig.from_dict(vlm_config_dict)
        else:
            vlm_config = BaseVLMConfig()
        
        # 處理分割模型配置
        seg_config_dict = config_dict.pop("seg_config", {})
        if isinstance(seg_config_dict, dict):
            seg_config = BaseSegConfig.from_dict(seg_config_dict)
        else:
            seg_config = BaseSegConfig()
        
        # 處理融合配置
        fusion_config = config_dict.pop("fusion_config", {})
        if not isinstance(fusion_config, dict):
            fusion_config = {}
        
        # 創建配置實例
        config = cls(
            vlm_config=vlm_config,
            seg_config=seg_config,
            fusion_config=fusion_config,
            **config_dict
        )
        return config
    
    @classmethod
    def from_json_string(cls, json_string: str) -> "Sa2VAConfig":
        """
        從 JSON 字符串創建配置
        
        Args:
            json_string (str): JSON 字符串
            
        Returns:
            Sa2VAConfig: 配置實例
        """
        config_dict = json.loads(json_string)
        return cls.from_dict(config_dict)
    
    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        push_to_hub: bool = False,
        **kwargs
    ) -> None:
        """
        保存配置到文件
        
        Args:
            save_directory (Union[str, os.PathLike]): 保存目錄
            push_to_hub (bool, optional): 是否推送到 Hub。默認為 False。
            **kwargs: 其他參數
        """
        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")
        
        os.makedirs(save_directory, exist_ok=True)
        
        # 保存主配置
        output_config_file = os.path.join(save_directory, "config.json")
        self.to_json_file(output_config_file)
        
        # 保存 VLM 配置
        vlm_config_dir = os.path.join(save_directory, "vlm_config")
        self.vlm_config.save_pretrained(vlm_config_dir)
        
        # 保存分割模型配置
        seg_config_dir = os.path.join(save_directory, "seg_config")
        self.seg_config.save_pretrained(seg_config_dir)
        
        if push_to_hub:
            self.push_to_hub(save_directory, **kwargs)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        **kwargs
    ) -> "Sa2VAConfig":
        """
        從文件加載配置
        
        Args:
            pretrained_model_name_or_path (Union[str, os.PathLike]): 預訓練模型名稱或路徑
            **kwargs: 其他參數
            
        Returns:
            Sa2VAConfig: 配置實例
        """
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
        
        # 加載 VLM 配置
        vlm_config_path = os.path.join(pretrained_model_name_or_path, "vlm_config")
        if os.path.exists(vlm_config_path):
            vlm_config = BaseVLMConfig.from_pretrained(vlm_config_path)
        else:
            vlm_config = BaseVLMConfig()
        
        # 加載分割模型配置
        seg_config_path = os.path.join(pretrained_model_name_or_path, "seg_config")
        if os.path.exists(seg_config_path):
            seg_config = BaseSegConfig.from_pretrained(seg_config_path)
        else:
            seg_config = BaseSegConfig()
        
        # 更新配置字典
        config_dict["vlm_config"] = vlm_config
        config_dict["seg_config"] = seg_config
        
        return cls.from_dict(config_dict, **kwargs) 