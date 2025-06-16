import unittest
from typing import Dict, List, Optional, Type, Union
import pytest

from transformers.configuration_utils import PretrainedConfig

from core.configuration import (
    BaseVisionConfig,
    BaseLLMConfig,
    VLMConfigRegistry,
    VLMConfigFactory,
)

class TestBaseVisionConfig(unittest.TestCase):
    """測試視覺模型配置基類"""
    
    def setUp(self):
        """設置測試環境"""
        self.valid_config = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "image_size": 224,
            "patch_size": 16,
        }
    
    def test_valid_config(self):
        """測試有效配置"""
        config = BaseVisionConfig(**self.valid_config)
        self.assertEqual(config.hidden_size, 768)
        self.assertEqual(config.num_hidden_layers, 12)
        self.assertEqual(config.num_attention_heads, 12)
        self.assertEqual(config.intermediate_size, 3072)
        self.assertEqual(config.image_size, 224)
        self.assertEqual(config.patch_size, 16)
    
    def test_invalid_hidden_size(self):
        """測試無效的隱藏層大小"""
        config_dict = self.valid_config.copy()
        config_dict["hidden_size"] = -1
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_num_layers(self):
        """測試無效的層數"""
        config_dict = self.valid_config.copy()
        config_dict["num_hidden_layers"] = 0
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_attention_heads(self):
        """測試無效的注意力頭數"""
        config_dict = self.valid_config.copy()
        config_dict["num_attention_heads"] = 5  # 不能被 hidden_size 整除
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_image_size(self):
        """測試無效的圖像大小"""
        config_dict = self.valid_config.copy()
        config_dict["image_size"] = 0
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_patch_size(self):
        """測試無效的圖像塊大小"""
        config_dict = self.valid_config.copy()
        config_dict["patch_size"] = 15  # 不能被 image_size 整除
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_hidden_act(self):
        """測試無效的激活函數"""
        config_dict = self.valid_config.copy()
        config_dict["hidden_act"] = "invalid_act"
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)
    
    def test_invalid_dropout(self):
        """測試無效的 Dropout 比率"""
        config_dict = self.valid_config.copy()
        config_dict["dropout"] = 1.5  # 超出範圍
        with self.assertRaises(ValueError):
            BaseVisionConfig(**config_dict)

class TestBaseLLMConfig(unittest.TestCase):
    """測試語言模型配置基類"""
    
    def setUp(self):
        """設置測試環境"""
        self.valid_config = {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "architectures": ["LlamaForCausalLM"],
        }
    
    def test_valid_config(self):
        """測試有效配置"""
        config = BaseLLMConfig(**self.valid_config)
        self.assertEqual(config.vocab_size, 32000)
        self.assertEqual(config.hidden_size, 2048)
        self.assertEqual(config.num_hidden_layers, 32)
        self.assertEqual(config.num_attention_heads, 32)
        self.assertEqual(config.intermediate_size, 8192)
        self.assertEqual(config.architectures, ["LlamaForCausalLM"])
    
    def test_invalid_vocab_size(self):
        """測試無效的詞表大小"""
        config_dict = self.valid_config.copy()
        config_dict["vocab_size"] = 0
        with self.assertRaises(ValueError):
            BaseLLMConfig(**config_dict)
    
    def test_invalid_architectures(self):
        """測試無效的模型架構"""
        config_dict = self.valid_config.copy()
        config_dict["architectures"] = ["InvalidArchitecture"]
        with self.assertRaises(ValueError):
            BaseLLMConfig(**config_dict)
    
    def test_empty_architectures(self):
        """測試空的模型架構列表"""
        config_dict = self.valid_config.copy()
        config_dict["architectures"] = []
        with self.assertRaises(ValueError):
            BaseLLMConfig(**config_dict)

class TestVLMConfigRegistry(unittest.TestCase):
    """測試 VLM 配置註冊表"""
    
    def setUp(self):
        """設置測試環境"""
        self.registry = VLMConfigRegistry()
        self.registry.clear()  # 清除所有註冊
    
    def test_singleton(self):
        """測試單例模式"""
        registry1 = VLMConfigRegistry()
        registry2 = VLMConfigRegistry()
        self.assertIs(registry1, registry2)
    
    def test_register_vision_config(self):
        """測試註冊視覺模型配置"""
        self.registry.register_vision_config("test_vision", BaseVisionConfig)
        self.assertIn("test_vision", self.registry.list_vision_configs())
        self.assertEqual(
            self.registry.get_vision_config_class("test_vision"),
            BaseVisionConfig
        )
    
    def test_register_llm_config(self):
        """測試註冊語言模型配置"""
        self.registry.register_llm_config("test_llm", BaseLLMConfig)
        self.assertIn("test_llm", self.registry.list_llm_configs())
        self.assertEqual(
            self.registry.get_llm_config_class("test_llm"),
            BaseLLMConfig
        )
    
    def test_register_invalid_vision_config(self):
        """測試註冊無效的視覺模型配置"""
        with self.assertRaises(ValueError):
            self.registry.register_vision_config("test", PretrainedConfig)
    
    def test_register_invalid_llm_config(self):
        """測試註冊無效的語言模型配置"""
        with self.assertRaises(ValueError):
            self.registry.register_llm_config("test", PretrainedConfig)
    
    def test_unregister_config(self):
        """測試取消註冊配置"""
        self.registry.register_vision_config("test", BaseVisionConfig)
        self.registry.unregister_vision_config("test")
        self.assertNotIn("test", self.registry.list_vision_configs())
    
    def test_clear(self):
        """測試清除所有註冊"""
        self.registry.register_vision_config("test1", BaseVisionConfig)
        self.registry.register_llm_config("test2", BaseLLMConfig)
        self.registry.clear()
        self.assertEqual(len(self.registry.list_vision_configs()), 0)
        self.assertEqual(len(self.registry.list_llm_configs()), 0)

class TestVLMConfigFactory(unittest.TestCase):
    """測試 VLM 配置工廠"""
    
    def setUp(self):
        """設置測試環境"""
        self.factory = VLMConfigFactory()
        self.registry = VLMConfigRegistry()
        self.registry.clear()  # 清除所有註冊
        
        # 註冊測試配置
        self.registry.register_vision_config("test_vision", BaseVisionConfig)
        self.registry.register_llm_config("test_llm", BaseLLMConfig)
        
        # 測試配置參數
        self.vision_config_dict = {
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "image_size": 224,
            "patch_size": 16,
        }
        self.llm_config_dict = {
            "vocab_size": 32000,
            "hidden_size": 2048,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "intermediate_size": 8192,
            "architectures": ["LlamaForCausalLM"],
        }
    
    def test_create_vision_config(self):
        """測試創建視覺模型配置"""
        config = self.factory.create_vision_config(
            "test_vision",
            **self.vision_config_dict
        )
        self.assertIsInstance(config, BaseVisionConfig)
        self.assertEqual(config.hidden_size, 768)
    
    def test_create_llm_config(self):
        """測試創建語言模型配置"""
        config = self.factory.create_llm_config(
            "test_llm",
            **self.llm_config_dict
        )
        self.assertIsInstance(config, BaseLLMConfig)
        self.assertEqual(config.vocab_size, 32000)
    
    def test_create_unknown_config(self):
        """測試創建未知配置"""
        with self.assertRaises(ValueError):
            self.factory.create_vision_config("unknown", **self.vision_config_dict)
    
    def test_create_config_from_dict(self):
        """測試從字典創建配置"""
        # 測試視覺模型配置
        vision_dict = self.vision_config_dict.copy()
        vision_dict["model_type"] = "test_vision"
        config = self.factory.create_config_from_dict(vision_dict)
        self.assertIsInstance(config, BaseVisionConfig)
        
        # 測試語言模型配置
        llm_dict = self.llm_config_dict.copy()
        llm_dict["model_type"] = "test_llm"
        config = self.factory.create_config_from_dict(llm_dict)
        self.assertIsInstance(config, BaseLLMConfig)
    
    def test_create_config_from_invalid_dict(self):
        """測試從無效字典創建配置"""
        with self.assertRaises(ValueError):
            self.factory.create_config_from_dict({"invalid": "dict"})
        
        with self.assertRaises(ValueError):
            self.factory.create_config_from_dict({"model_type": "unknown"})

if __name__ == "__main__":
    unittest.main() 