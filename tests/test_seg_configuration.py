import unittest
import json
import os
import tempfile
from typing import Dict, Any

from core.configuration import (
    SegConfigRegistry,
    SegConfigFactory,
    SAM2Config,
    OtherSegConfig,
    BaseSegConfig,
    create_seg_config
)

class TestSegConfigRegistry(unittest.TestCase):
    """測試分割模型配置註冊表"""
    
    def setUp(self):
        """設置測試環境"""
        # 保存原始配置
        self.original_configs = SegConfigRegistry._configs.copy()
        # 測試用配置
        self.test_config = {
            "encoder": {"type": "test_encoder"},
            "decoder": {"type": "test_decoder"}
        }
    
    def tearDown(self):
        """清理測試環境"""
        # 恢復原始配置
        SegConfigRegistry._configs.clear()
        SegConfigRegistry._configs.update(self.original_configs)
    
    def test_register_config(self):
        """測試註冊新配置"""
        # 註冊測試配置
        SegConfigRegistry.register("test_model", self.test_config)
        
        # 驗證配置已註冊
        self.assertIn("test_model", SegConfigRegistry._configs)
        self.assertEqual(
            SegConfigRegistry._configs["test_model"],
            self.test_config
        )
    
    def test_get_default_config(self):
        """測試獲取默認配置"""
        # 註冊測試配置
        SegConfigRegistry.register("test_model", self.test_config)
        
        # 獲取配置
        config = SegConfigRegistry.get_default_config("test_model")
        self.assertEqual(config, self.test_config)
    
    def test_get_invalid_config(self):
        """測試獲取無效配置"""
        with self.assertRaises(ValueError) as context:
            SegConfigRegistry.get_default_config("invalid_model")
        
        self.assertIn("invalid_model", str(context.exception))
        self.assertIn("not registered", str(context.exception))
    
    def test_list_models(self):
        """測試列出已註冊模型"""
        # 註冊測試配置
        SegConfigRegistry.register("test_model", self.test_config)
        
        # 獲取模型列表
        models = SegConfigRegistry.list_models()
        
        # 驗證列表內容
        self.assertIsInstance(models, list)
        self.assertIn("test_model", models)
        self.assertIn("sam2", models)  # 默認配置
        self.assertIn("other_seg", models)  # 默認配置
    
    def test_config_update(self):
        """測試更新已存在的配置"""
        # 註冊初始配置
        SegConfigRegistry.register("test_model", self.test_config)
        
        # 更新配置
        updated_config = {
            "encoder": {"type": "updated_encoder"},
            "decoder": {"type": "updated_decoder"}
        }
        SegConfigRegistry.register("test_model", updated_config)
        
        # 驗證配置已更新
        config = SegConfigRegistry.get_default_config("test_model")
        self.assertEqual(config, updated_config)

class TestBaseSegConfig(unittest.TestCase):
    """測試分割模型配置基類"""
    
    def setUp(self):
        """設置測試環境"""
        # 創建測試配置
        self.valid_config = {
            "seg_type": "sam2",
            "image_size": 1024,
            "patch_size": 16,
            "hidden_size": 768,
            "num_hidden_layers": 12,
            "num_attention_heads": 12,
            "intermediate_size": 3072,
            "hidden_act": "gelu",
            "initializer_range": 0.02,
            "layer_norm_eps": 1e-5
        }
    
    def test_basic_params_validation(self):
        """測試基本參數驗證"""
        # 測試無效的 patch 大小（必須先驗證，避免除零錯誤）
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", patch_size=0)
        self.assertIn("patch_size", str(context.exception))
        
        # 測試無效的圖像大小
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", image_size=-1)
        self.assertIn("image_size", str(context.exception))
        
        # 測試圖像大小不能被 patch 大小整除
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", image_size=1000, patch_size=16)
        self.assertIn("divisible", str(context.exception))
        
        # 測試無效的隱藏層大小
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", hidden_size=-768)
        self.assertIn("hidden_size", str(context.exception))
        
        # 測試無效的注意力頭數
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", num_attention_heads=0)
        self.assertIn("num_attention_heads", str(context.exception))
        
        # 測試隱藏層大小不能被注意力頭數整除
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", hidden_size=768, num_attention_heads=5)
        self.assertIn("divisible", str(context.exception))
        
        # 測試無效的激活函數
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", hidden_act="invalid_act")
        self.assertIn("hidden_act", str(context.exception))
        
        # 測試無效的初始化範圍
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", initializer_range=-0.02)
        self.assertIn("initializer_range", str(context.exception))
        
        # 測試無效的 LayerNorm epsilon
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", layer_norm_eps=-1e-5)
        self.assertIn("layer_norm_eps", str(context.exception))
        
        # 測試無效的層數
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", num_hidden_layers=0)
        self.assertIn("num_hidden_layers", str(context.exception))
        
        # 測試無效的中間層大小
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", intermediate_size=-3072)
        self.assertIn("intermediate_size", str(context.exception))
    
    def test_config_merge(self):
        """測試配置合併邏輯"""
        # 創建自定義配置
        custom_config = {
            "image_encoder": {
                "type": "custom_vit",
                "depth": 24
            }
        }
        
        # 創建配置實例
        config = SAM2Config(
            seg_type="sam2",
            seg_config=custom_config
        )
        
        # 驗證配置合併
        self.assertIn("image_encoder", config.seg_config)
        self.assertEqual(config.seg_config["image_encoder"]["type"], "custom_vit")
        self.assertEqual(config.seg_config["image_encoder"]["depth"], 24)
        # 驗證默認值保留
        self.assertIn("prompt_encoder", config.seg_config)
        self.assertIn("mask_decoder", config.seg_config)
    
    def test_serialization(self):
        """測試序列化和反序列化"""
        # 創建配置實例
        config = SAM2Config(**self.valid_config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存配置到臨時文件
            config_path = os.path.join(tmp_dir, "config.json")
            config.to_json_file(config_path)
            
            # 從文件加載配置
            new_config = SAM2Config.from_pretrained(tmp_dir)
            
            # 驗證配置是否正確加載
            self.assertEqual(new_config.seg_type, config.seg_type)
            self.assertEqual(new_config.image_size, config.image_size)
            self.assertEqual(new_config.seg_config, config.seg_config)
    
    def test_save_load(self):
        """測試配置保存和加載"""
        # 創建配置實例
        config = SAM2Config(**self.valid_config)
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 保存配置
            config.save_pretrained(tmp_dir)
            
            # 驗證文件存在
            config_path = os.path.join(tmp_dir, "config.json")
            self.assertTrue(os.path.exists(config_path))
            
            # 加載配置
            loaded_config = SAM2Config.from_pretrained(tmp_dir)
            self.assertEqual(loaded_config.seg_type, config.seg_type)
            self.assertEqual(loaded_config.image_size, config.image_size)
            self.assertEqual(loaded_config.seg_config, config.seg_config)

class TestSAM2Config(unittest.TestCase):
    """測試 SAM2 模型配置類"""
    
    def setUp(self):
        """設置測試環境"""
        self.valid_config = {
            "seg_type": "sam2",
            "seg_config": {
                "image_encoder": {
                    "type": "vit",
                    "depth": 12,
                    "embed_dim": 768
                },
                "prompt_encoder": {
                    "embed_dim": 256,
                    "image_embed_size": 1024
                },
                "mask_decoder": {
                    "transformer_dim": 256,
                    "transformer_heads": 8
                }
            }
        }
    
    def test_custom_validation(self):
        """測試 SAM2 特定配置驗證"""
        # 測試缺少必需配置項
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", seg_config={})
        self.assertIn("Missing required key", str(context.exception))
        
        # 測試缺少 image_encoder
        invalid_config = {
            "prompt_encoder": {},
            "mask_decoder": {}
        }
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", seg_config=invalid_config)
        self.assertIn("image_encoder", str(context.exception))
        
        # 測試缺少 prompt_encoder
        invalid_config = {
            "image_encoder": {},
            "mask_decoder": {}
        }
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", seg_config=invalid_config)
        self.assertIn("prompt_encoder", str(context.exception))
        
        # 測試缺少 mask_decoder
        invalid_config = {
            "image_encoder": {},
            "prompt_encoder": {}
        }
        with self.assertRaises(ValueError) as context:
            SAM2Config(seg_type="sam2", seg_config=invalid_config)
        self.assertIn("mask_decoder", str(context.exception))
    
    def test_default_init(self):
        """測試默認配置初始化"""
        config = SAM2Config(seg_type="sam2")
        
        # 驗證基本屬性
        self.assertEqual(config.seg_type, "sam2")
        self.assertEqual(config.model_type, "sam2")
        
        # 驗證默認配置
        self.assertIn("image_encoder", config.seg_config)
        self.assertIn("prompt_encoder", config.seg_config)
        self.assertIn("mask_decoder", config.seg_config)
    
    def test_custom_init(self):
        """測試自定義配置初始化"""
        config = SAM2Config(**self.valid_config)
        
        # 驗證配置
        self.assertEqual(config.seg_type, "sam2")
        self.assertEqual(config.seg_config["image_encoder"]["type"], "vit")
        self.assertEqual(config.seg_config["image_encoder"]["depth"], 12)
        self.assertEqual(config.seg_config["prompt_encoder"]["embed_dim"], 256)
        self.assertEqual(config.seg_config["mask_decoder"]["transformer_dim"], 256)

class TestOtherSegConfig(unittest.TestCase):
    """測試其他分割模型配置類"""
    
    def setUp(self):
        """設置測試環境"""
        self.valid_config = {
            "seg_type": "other_seg",
            "seg_config": {
                "encoder": {
                    "type": "resnet",
                    "depth": 50
                },
                "decoder": {
                    "type": "fpn",
                    "in_channels": [256, 512, 1024, 2048]
                }
            }
        }
    
    def test_custom_validation(self):
        """測試其他分割模型特定配置驗證"""
        # 測試缺少必需配置項
        with self.assertRaises(ValueError) as context:
            OtherSegConfig(seg_type="other_seg", seg_config={})
        self.assertIn("Missing required key", str(context.exception))
        
        # 測試缺少 encoder
        invalid_config = {
            "decoder": {"type": "fpn"}
        }
        with self.assertRaises(ValueError) as context:
            OtherSegConfig(seg_type="other_seg", seg_config=invalid_config)
        self.assertIn("encoder", str(context.exception))
        
        # 測試缺少 decoder
        invalid_config = {
            "encoder": {"type": "resnet"}
        }
        with self.assertRaises(ValueError) as context:
            OtherSegConfig(seg_type="other_seg", seg_config=invalid_config)
        self.assertIn("decoder", str(context.exception))
    
    def test_default_init(self):
        """測試默認配置初始化"""
        config = OtherSegConfig(seg_type="other_seg")
        
        # 驗證基本屬性
        self.assertEqual(config.seg_type, "other_seg")
        self.assertEqual(config.model_type, "other_seg")
        
        # 驗證默認配置
        self.assertIn("encoder", config.seg_config)
        self.assertIn("decoder", config.seg_config)
    
    def test_custom_init(self):
        """測試自定義配置初始化"""
        config = OtherSegConfig(**self.valid_config)
        
        # 驗證配置
        self.assertEqual(config.seg_type, "other_seg")
        self.assertEqual(config.seg_config["encoder"]["type"], "resnet")
        self.assertEqual(config.seg_config["encoder"]["depth"], 50)
        self.assertEqual(config.seg_config["decoder"]["type"], "fpn")
        self.assertEqual(
            config.seg_config["decoder"]["in_channels"],
            [256, 512, 1024, 2048]
        )

class TestSegConfigFactory(unittest.TestCase):
    """測試分割模型配置工廠類"""
    
    def setUp(self):
        """設置測試環境"""
        # 保存原始配置類
        self.original_classes = SegConfigFactory._config_classes.copy()
    
    def tearDown(self):
        """清理測試環境"""
        # 恢復原始配置類
        SegConfigFactory._config_classes.clear()
        SegConfigFactory._config_classes.update(self.original_classes)
    
    def test_create_config(self):
        """測試創建配置實例"""
        # 測試創建 SAM2 配置
        config = SegConfigFactory.create_config("sam2")
        self.assertIsInstance(config, SAM2Config)
        self.assertEqual(config.seg_type, "sam2")
        
        # 測試創建其他分割模型配置
        config = SegConfigFactory.create_config("other_seg")
        self.assertIsInstance(config, OtherSegConfig)
        self.assertEqual(config.seg_type, "other_seg")
    
    def test_register_config_class(self):
        """測試註冊配置類"""
        # 創建測試配置類
        class TestConfig(BaseSegConfig):
            model_type = "test_model"
            def _validate_custom_config(self):
                pass
        
        # 註冊配置類
        SegConfigFactory.register_config("test_model", TestConfig)
        
        # 驗證配置類已註冊
        config = SegConfigFactory.create_config("test_model")
        self.assertIsInstance(config, TestConfig)
        self.assertEqual(config.seg_type, "test_model")
    
    def test_invalid_config_class(self):
        """測試註冊無效配置類"""
        class InvalidConfig:
            pass
        
        with self.assertRaises(TypeError) as context:
            SegConfigFactory.register_config("invalid", InvalidConfig)
        self.assertIn("subclass of BaseSegConfig", str(context.exception))
    
    def test_invalid_model_type(self):
        """測試創建無效模型類型的配置"""
        with self.assertRaises(ValueError) as context:
            SegConfigFactory.create_config("invalid_model")
        self.assertIn("not registered", str(context.exception))

class TestCreateSegConfig(unittest.TestCase):
    """測試分割模型配置工廠函數"""
    
    def test_default_creation(self):
        """測試創建默認配置"""
        # 測試默認配置（SAM2）
        config = create_seg_config()
        self.assertIsInstance(config, SAM2Config)
        self.assertEqual(config.seg_type, "sam2")
        
        # 測試指定默認類型
        config = create_seg_config(seg_type="sam2")
        self.assertIsInstance(config, SAM2Config)
        self.assertEqual(config.seg_type, "sam2")
    
    def test_custom_creation(self):
        """測試創建自定義配置"""
        # 創建自定義 SAM2 配置
        custom_config = {
            "image_encoder": {
                "type": "custom_vit",
                "depth": 24
            }
        }
        config = create_seg_config(
            seg_type="sam2",
            seg_config=custom_config
        )
        self.assertIsInstance(config, SAM2Config)
        self.assertEqual(config.seg_type, "sam2")
        self.assertEqual(
            config.seg_config["image_encoder"]["type"],
            "custom_vit"
        )
        
        # 創建自定義其他分割模型配置
        custom_config = {
            "encoder": {"type": "custom_encoder"},
            "decoder": {"type": "custom_decoder"}
        }
        config = create_seg_config(
            seg_type="other_seg",
            seg_config=custom_config
        )
        self.assertIsInstance(config, OtherSegConfig)
        self.assertEqual(config.seg_type, "other_seg")
        self.assertEqual(
            config.seg_config["encoder"]["type"],
            "custom_encoder"
        )
    
    def test_parameter_passing(self):
        """測試參數傳遞"""
        # 測試傳遞額外參數
        config = create_seg_config(
            seg_type="sam2",
            image_size=2048,
            patch_size=32,
            hidden_size=1024
        )
        self.assertEqual(config.image_size, 2048)
        self.assertEqual(config.patch_size, 32)
        self.assertEqual(config.hidden_size, 1024)
    
    def test_error_handling(self):
        """測試錯誤處理"""
        # 測試無效的模型類型
        with self.assertRaises(ValueError) as context:
            create_seg_config(seg_type="invalid_model")
        self.assertIn("not registered", str(context.exception))
        
        # 測試無效的配置參數
        with self.assertRaises(ValueError) as context:
            create_seg_config(
                seg_type="sam2",
                image_size=-1
            )
        self.assertIn("image_size", str(context.exception))

if __name__ == "__main__":
    unittest.main() 
