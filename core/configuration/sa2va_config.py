from transformers import PretrainedConfig
from typing import Dict, Any, Optional, Union
import json
import os
from .seg_config import BaseSegConfig, create_seg_config
from .vlm_factory import VLMConfigFactory
from transformers.utils import logging

logger = logging.get_logger(__name__)

class Sa2VAConfig(PretrainedConfig):
    """
    Sa2VA 模型的頂層配置類。

    這個類作為整個 Sa2VA 模型的統一配置入口，負責動態管理和整合視覺語言模型 (VLM)
    和分割模型 (Segmentation) 的配置。

    用戶可以通過指定 `vlm_type` 和 `seg_type` 來快速創建一個完整的配置，同時也可以
    通過 `vlm_config` 和 `seg_config` 字典來覆蓋任意底層的默認參數。

    得益於與 Transformers 框架的深度整合，所有配置都可以通過 `save_pretrained` 和
    `from_pretrained` 方法進行可靠的序列化和反序列化。

    Args:
        vlm_type (str, optional):
            VLM 的模型類型，例如 "llava", "qwen-vl"。默認為 "llava"。
        seg_type (str, optional):
            分割模型的模型類型，例如 "sam2"。默認為 "sam2"。
        vlm_config (Optional[Dict[str, Any]], optional):
            一個字典，用於覆蓋默認的 VLM 配置參數。
        seg_config (Optional[Dict[str, Any]], optional):
            一個字典，用於覆蓋默認的分割模型配置參數。
        fusion_strategy (str, optional):
            VLM 和分割模型特徵的融合策略。默認為 "default"。
        **kwargs:
            其他傳遞給 `PretrainedConfig` 的參數。
    """

    model_type = "sa2va"
    
    def __init__(
        self,
        vlm_type: str = "llava",
        seg_type: str = "sam2",
        vlm_config: Optional[Dict[str, Any]] = None,
        seg_config: Optional[Dict[str, Any]] = None,
        fusion_strategy: str = "default",
        **kwargs
    ):
        # 步驟 1: 處理來自 `from_pretrained` 的嵌套字典。
        # 當從 JSON 文件加載時，`vlm_config` 和 `seg_config` 會作為字典出現在 kwargs 中。
        # 我們需要把它們彈出，以避免 PretrainedConfig 的基類 __init__ 方法錯誤地處理它們。
        vlm_config_from_kwargs = kwargs.pop("vlm_config", None)
        seg_config_from_kwargs = kwargs.pop("seg_config", None)

        # 優先使用來自 kwargs 的字典 (from_pretrained 的情況)，其次使用用戶直接傳入的字典。
        vlm_config_dict = vlm_config_from_kwargs if vlm_config_from_kwargs is not None else (vlm_config or {})
        seg_config_dict = seg_config_from_kwargs if seg_config_from_kwargs is not None else (seg_config or {})

        # 步驟 2: 確定真實的模型類型。
        # 同樣，優先從字典中獲取類型，這對於反序列化至關重要。
        effective_vlm_type = vlm_config_dict.get("vlm_type", vlm_type)
        effective_seg_type = seg_config_dict.get("seg_type", seg_type)

        # 步驟 3: 使用工廠動態創建子配置對象。
        # 這是整個設計的核心，將字典轉換為對應的、帶有驗證邏輯的配置類實例。
        self.vlm_config = VLMConfigFactory().create_vlm_config(effective_vlm_type, **vlm_config_dict)
        self.seg_config = create_seg_config(effective_seg_type, **seg_config_dict)

        # 步驟 4: 設置頂層配置的屬性。
        self.fusion_strategy = fusion_strategy
        
        # 為了方便訪問和確保序列化時類型信息不丟失，我們將類型也存儲在頂層。
        self.vlm_type = self.vlm_config.vlm_type
        self.seg_type = self.seg_config.seg_type

        # 步驟 5: 最後調用父類的構造函數。
        # 將剩餘的 kwargs 傳遞給父類，完成標準的初始化流程。
        super().__init__(**kwargs)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        將配置轉換為字典，確保類型信息被正確保存。
        
        Returns:
            Dict[str, Any]: 包含完整類型信息的配置字典
        """
        output = super().to_dict()
        
        # 確保類型信息被保存
        output["vlm_type"] = self.vlm_type
        output["seg_type"] = self.seg_type
        
        # 序列化子配置
        output["vlm_config"] = self.vlm_config.to_dict()
        output["seg_config"] = self.seg_config.to_dict()
        
        return output
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> "Sa2VAConfig":
        """
        從字典創建配置，使用與 __init__ 相同的邏輯。
        
        Args:
            config_dict (Dict[str, Any]): 配置字典
            **kwargs: 其他參數
            
        Returns:
            Sa2VAConfig: 配置實例
        """
        # 提取類型信息
        vlm_type = config_dict.pop("vlm_type", "llava")
        seg_type = config_dict.pop("seg_type", "sam2")
        
        # 提取子配置字典
        vlm_config_dict = config_dict.pop("vlm_config", {})
        seg_config_dict = config_dict.pop("seg_config", {})
        
        # 使用與 __init__ 相同的邏輯創建實例
        return cls(
            vlm_type=vlm_type,
            seg_type=seg_type,
            vlm_config=vlm_config_dict,
            seg_config=seg_config_dict,
            **config_dict
        ) 