import pytest
from typing import Dict, Optional
from transformers.configuration_utils import PretrainedConfig

from core.configuration import (
    BaseVLMConfig,
    LLaVAConfig,
    QwenVLConfig,
    InternVLConfig,
    Phi3Config,
    VLMConfigRegistry,
    VLMConfigFactory
)

# 測試用的默認視覺配置
DEFAULT_VISION_CONFIG = {
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "num_attention_heads": 16,
    "intermediate_size": 4096,
    "image_size": 224,
    "patch_size": 14,
}

# 為每個模型提供正確的默認語言模型配置
DEFAULT_LLAVA_LLM_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
}

DEFAULT_QWEN_VL_LLM_CONFIG = {
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
}

DEFAULT_INTERN_VL_LLM_CONFIG = {
    "architectures": ["InternLM2ForCausalLM"],
    "hidden_size": 4096,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "intermediate_size": 11008,
    "vocab_size": 32000
}

class TestBaseVLMConfig:
    """測試 VLM 配置基類
    
    注意：BaseVLMConfig 是一個抽象基類，不能直接實例化。
    我們通過測試其子類來間接測試基類的功能。
    """
    
    def test_base_class_is_abstract(self):
        """測試基類是抽象的，不能直接實例化"""
        with pytest.raises(NotImplementedError):
            # 嘗試直接實例化基類應該失敗
            # 提供必要的配置參數，但由於是抽象類，仍然應該拋出 NotImplementedError
            BaseVLMConfig(
                vlm_type="llava",
                vision_config=DEFAULT_VISION_CONFIG,
                llm_config=DEFAULT_LLAVA_LLM_CONFIG
            )
    
    def test_base_class_validation_in_subclass(self):
        """通過子類測試基類的驗證邏輯"""
        # 使用 LLaVAConfig 測試基類的驗證邏輯
        with pytest.raises(ValueError, match="vision_config must be a dictionary"):
            LLaVAConfig(vision_config="invalid")
        
        with pytest.raises(ValueError, match="llm_config must be a dictionary"):
            LLaVAConfig(llm_config="invalid")
        
        # 測試缺少架構配置
        invalid_llm_config = DEFAULT_LLAVA_LLM_CONFIG.copy()
        del invalid_llm_config["architectures"]
        with pytest.raises(ValueError, match="llm_config must specify architectures"):
            LLaVAConfig(llm_config=invalid_llm_config)
        
        # 測試無效的架構
        invalid_llm_config = DEFAULT_LLAVA_LLM_CONFIG.copy()
        invalid_llm_config["architectures"] = ["InvalidArchitecture"]
        with pytest.raises(ValueError, match="Unsupported LLM architecture"):
            LLaVAConfig(llm_config=invalid_llm_config)

class TestLLaVAConfig:
    """測試 LLaVA 配置類"""
    
    def test_init_with_valid_config(self):
        """測試使用有效配置初始化"""
        config = LLaVAConfig(
            vision_config=DEFAULT_VISION_CONFIG,
            llm_config=DEFAULT_LLAVA_LLM_CONFIG
        )
        assert config.model_type == "llava"
        assert config.vlm_type == "llava"
    
    def test_init_with_missing_vision_keys(self):
        """測試缺少必需的視覺配置鍵"""
        invalid_vision_config = DEFAULT_VISION_CONFIG.copy()
        del invalid_vision_config["hidden_size"]
        with pytest.raises(ValueError, match="Missing required key 'hidden_size'"):
            LLaVAConfig(
                vision_config=invalid_vision_config,
                llm_config=DEFAULT_LLAVA_LLM_CONFIG
            )

class TestQwenVLConfig:
    """測試 Qwen-VL 配置類"""
    
    def test_init_with_valid_config(self):
        """測試使用有效配置初始化"""
        config = QwenVLConfig(
            vision_config=DEFAULT_VISION_CONFIG,
            llm_config=DEFAULT_QWEN_VL_LLM_CONFIG
        )
        assert config.model_type == "qwen-vl"
        assert config.vlm_type == "qwen-vl"
    
    def test_init_with_missing_vision_keys(self):
        """測試缺少必需的視覺配置鍵"""
        invalid_vision_config = DEFAULT_VISION_CONFIG.copy()
        del invalid_vision_config["num_hidden_layers"]
        with pytest.raises(ValueError, match="Missing required key 'num_hidden_layers'"):
            QwenVLConfig(
                vision_config=invalid_vision_config,
                llm_config=DEFAULT_QWEN_VL_LLM_CONFIG
            )

class TestInternVLConfig:
    """測試 Intern-VL 配置類"""
    
    def test_init_with_valid_config(self):
        """測試使用有效配置初始化"""
        vision_config = DEFAULT_VISION_CONFIG.copy()
        vision_config["use_flash_attn"] = True
        config = InternVLConfig(
            vision_config=vision_config,
            llm_config=DEFAULT_INTERN_VL_LLM_CONFIG
        )
        assert config.vlm_type == "intern-vl"
        assert config.vision_config["use_flash_attn"] is True
    
    def test_init_with_missing_flash_attn(self):
        """測試缺少 flash attention 配置"""
        vision_config = DEFAULT_VISION_CONFIG.copy()
        with pytest.raises(ValueError, match="Missing required key 'use_flash_attn' in vision_config"):
            InternVLConfig(
                vision_config=vision_config,
                llm_config=DEFAULT_INTERN_VL_LLM_CONFIG
            )
    
    def test_create_qwen_vl_config(self):
        """測試創建 Qwen-VL 配置"""
        factory = VLMConfigFactory()
        config = factory.create_vlm_config(
            "qwen-vl",
            vision_config=DEFAULT_VISION_CONFIG,
            llm_config=DEFAULT_QWEN_VL_LLM_CONFIG
        )
        assert isinstance(config, QwenVLConfig)
    
    def test_create_intern_vl_config(self):
        """測試創建 Intern-VL 配置"""
        factory = VLMConfigFactory()
        vision_config = DEFAULT_VISION_CONFIG.copy()
        vision_config["use_flash_attn"] = True
        config = factory.create_vlm_config(
            "intern-vl",
            vision_config=vision_config,
            llm_config=DEFAULT_INTERN_VL_LLM_CONFIG
        )
        assert isinstance(config, InternVLConfig)

class TestPhi3Config:
    """測試 Phi-3 配置類"""
    
    def test_init_with_valid_config(self):
        """測試使用有效配置初始化"""
        vision_config = DEFAULT_VISION_CONFIG.copy()
        vision_config["use_flash_attn"] = True
        llm_config = DEFAULT_INTERN_VL_LLM_CONFIG.copy()  # 使用 Intern-VL 的配置作為基礎
        llm_config["architectures"] = ["PhiForCausalLM"]  # 修改為 Phi-3 的架構
        llm_config["max_position_embeddings"] = 2048  # 添加 Phi-3 特有的參數
        llm_config["rope_scaling"] = {  # 添加 Phi-3 特有的 RoPE 縮放配置
            "type": "linear",
            "factor": 2.0
        }
        
        config = Phi3Config(
            vision_config=vision_config,
            llm_config=llm_config
        )
        assert config.vlm_type == "phi3"
        assert config.vision_config["use_flash_attn"] is True
        assert config.llm_config["architectures"][0] == "PhiForCausalLM"
        assert config.llm_config["rope_scaling"]["type"] == "linear"
    
    def test_init_with_missing_vision_keys(self):
        """測試缺少必需的視覺配置鍵"""
        vision_config = DEFAULT_VISION_CONFIG.copy()
        del vision_config["hidden_size"]
        
        # 確保 llm_config 包含所有必需的參數
        llm_config = DEFAULT_INTERN_VL_LLM_CONFIG.copy()
        llm_config["architectures"] = ["PhiForCausalLM"]
        llm_config["max_position_embeddings"] = 2048  # 添加必需的參數
        
        with pytest.raises(ValueError, match="Missing required key 'hidden_size' in vision_config"):
            Phi3Config(
                vision_config=vision_config,
                llm_config=llm_config
            )
    
    def test_init_with_invalid_rope_scaling(self):
        """測試無效的 RoPE 縮放配置"""
        vision_config = DEFAULT_VISION_CONFIG.copy()
        vision_config["use_flash_attn"] = True
        llm_config = DEFAULT_INTERN_VL_LLM_CONFIG.copy()
        llm_config["architectures"] = ["PhiForCausalLM"]
        llm_config["max_position_embeddings"] = 2048
        
        # 測試無效的類型
        llm_config["rope_scaling"] = {"type": "invalid", "factor": 2.0}
        with pytest.raises(ValueError, match="rope_scaling type must be 'linear' or 'dynamic'"):
            Phi3Config(
                vision_config=vision_config,
                llm_config=llm_config
            )
        
        # 測試無效的因子
        llm_config["rope_scaling"] = {"type": "linear", "factor": -1.0}
        with pytest.raises(ValueError, match="rope_scaling factor must be a positive number"):
            Phi3Config(
                vision_config=vision_config,
                llm_config=llm_config
            )
    
    def test_create_phi3_config(self):
        """測試通過工廠創建 Phi-3 配置"""
        factory = VLMConfigFactory()
        vision_config = DEFAULT_VISION_CONFIG.copy()
        vision_config["use_flash_attn"] = True
        llm_config = DEFAULT_INTERN_VL_LLM_CONFIG.copy()
        llm_config["architectures"] = ["PhiForCausalLM"]
        llm_config["max_position_embeddings"] = 2048
        llm_config["rope_scaling"] = {
            "type": "linear",
            "factor": 2.0
        }
        
        config = factory.create_vlm_config(
            "phi3",
            vision_config=vision_config,
            llm_config=llm_config
        )
        assert isinstance(config, Phi3Config)
        assert config.vlm_type == "phi3"
