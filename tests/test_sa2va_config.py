import unittest
import tempfile
import shutil
import os

from core.configuration.sa2va_config import Sa2VAConfig
from core.configuration.vlm_model_configs import LLaVAConfig, QwenVLConfig
from core.configuration.seg_config import SAM2Config

class TestSa2VAConfig(unittest.TestCase):

    def test_default_initialization(self):
        """測試使用默認參數進行初始化"""
        config = Sa2VAConfig()
        
        # 驗證頂層參數
        self.assertEqual(config.vlm_type, "llava")
        self.assertEqual(config.seg_type, "sam2")
        self.assertEqual(config.model_type, "sa2va")
        
        # 驗證子配置的類型
        self.assertIsInstance(config.vlm_config, LLaVAConfig)
        self.assertIsInstance(config.seg_config, SAM2Config)

    def test_custom_initialization(self):
        """測試使用自定義類型進行初始化"""
        config = Sa2VAConfig(vlm_type="qwen-vl")
        
        self.assertEqual(config.vlm_type, "qwen-vl")
        self.assertIsInstance(config.vlm_config, QwenVLConfig)
        self.assertIsInstance(config.seg_config, SAM2Config) # seg_config 應保持默認

    def test_parameter_override(self):
        """測試覆蓋深層次的配置參數"""
        custom_vlm_params = {
            "llm_config": {
                "hidden_size": 4097,  # 使用一個獨特的值來測試
                "num_attention_heads": 17
            }
        }
        config = Sa2VAConfig(vlm_type="llava", vlm_config=custom_vlm_params)
        
        self.assertEqual(config.vlm_config.llm_config["hidden_size"], 4097)
        self.assertEqual(config.vlm_config.llm_config["num_attention_heads"], 17)

    def test_serialization_deserialization(self):
        """測試配置的保存和加載流程"""
        # 創建一個臨時目錄來保存配置
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 1. 創建一個帶有自定義參數的原始配置
            original_config = Sa2VAConfig(
                vlm_type="qwen-vl",
                seg_type="sam2",
                vlm_config={"llm_config": {"hidden_size": 4096}},
                fusion_strategy="concat"
            )
            
            # 2. 保存配置
            original_config.save_pretrained(temp_dir)
            
            # 檢查 config.json 文件是否已創建
            self.assertTrue(os.path.exists(os.path.join(temp_dir, "config.json")))
            
            # 3. 加載配置
            loaded_config = Sa2VAConfig.from_pretrained(temp_dir)
            
            # 4. 驗證加載的配置
            
            # 驗證頂層參數是否一致
            self.assertEqual(original_config.model_type, loaded_config.model_type)
            self.assertEqual(original_config.vlm_type, loaded_config.vlm_type)
            self.assertEqual(original_config.seg_type, loaded_config.seg_type)
            self.assertEqual(original_config.fusion_strategy, loaded_config.fusion_strategy)
            
            # 驗證子配置的類型是否正確還原
            self.assertIsInstance(loaded_config.vlm_config, QwenVLConfig)
            self.assertIsInstance(loaded_config.seg_config, SAM2Config)
            
            # 驗證深層次的參數是否一致
            self.assertEqual(
                original_config.vlm_config.llm_config["hidden_size"],
                loaded_config.vlm_config.llm_config["hidden_size"]
            )

            # 使用 to_dict 進行完整的字典比較，確保所有內容都一致
            self.assertDictEqual(original_config.to_dict(), loaded_config.to_dict())

        finally:
            # 清理臨時目錄
            shutil.rmtree(temp_dir)

    def test_error_handling_for_invalid_type(self):
        """測試傳入無效的模型類型時是否會拋出錯誤"""
        with self.assertRaises(ValueError):
            Sa2VAConfig(vlm_type="non_existent_model")

if __name__ == '__main__':
    unittest.main() 