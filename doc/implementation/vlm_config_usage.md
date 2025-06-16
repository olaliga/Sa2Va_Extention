# VLM 配置系統使用指南

本文檔提供了 VLM 配置系統的使用示例，包括基本配置、註冊、創建和管理配置實例等操作。

## 1. 基本配置類使用

### 1.1 創建 VLM 配置

```python
from core.configuration import VLMConfigFactory

# 創建 LLaVA 配置
llava_config = VLMConfigFactory.create_vlm_config(
    "llava",
    vision_config={
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16
    },
    llm_config={
        "architectures": ["LlamaForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32
    }
)

# 創建 Qwen-VL 配置
qwen_vl_config = VLMConfigFactory.create_vlm_config(
    "qwen-vl",
    vision_config={
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16
    },
    llm_config={
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32
    }
)

# 創建 Intern-VL 配置
intern_vl_config = VLMConfigFactory.create_vlm_config(
    "intern-vl",
    vision_config={
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "use_flash_attn": True  # Intern-VL 特有的配置
    },
    llm_config={
        "architectures": ["InternLM2ForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "vocab_size": 32000
    }
)

# 創建 Phi-3 配置
phi3_config = VLMConfigFactory.create_vlm_config(
    "phi3",
    vision_config={
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "use_flash_attn": True
    },
    llm_config={
        "architectures": ["PhiForCausalLM"],
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "max_position_embeddings": 2048,
        "rope_scaling": {  # Phi-3 特有的 RoPE 配置
            "type": "linear",
            "factor": 2.0
        }
    }
)
```

### 1.2 使用默認配置

```python
from core.configuration import (
    DEFAULT_LLAVA_LLM_CONFIG,
    DEFAULT_QWEN_VL_LLM_CONFIG,
    DEFAULT_INTERN_VL_LLM_CONFIG,
    DEFAULT_VISION_CONFIG
)

# 使用默認配置創建 LLaVA 配置
llava_config = VLMConfigFactory.create_vlm_config(
    "llava",
    vision_config=DEFAULT_VISION_CONFIG,
    llm_config=DEFAULT_LLAVA_LLM_CONFIG
)

# 使用默認配置創建 Qwen-VL 配置
qwen_vl_config = VLMConfigFactory.create_vlm_config(
    "qwen-vl",
    vision_config=DEFAULT_VISION_CONFIG,
    llm_config=DEFAULT_QWEN_VL_LLM_CONFIG
)

# 使用默認配置創建 Intern-VL 配置
intern_vl_config = VLMConfigFactory.create_vlm_config(
    "intern-vl",
    vision_config={**DEFAULT_VISION_CONFIG, "use_flash_attn": True},
    llm_config=DEFAULT_INTERN_VL_LLM_CONFIG
)

# 使用默認配置創建 Phi-3 配置
phi3_config = VLMConfigFactory.create_vlm_config(
    "phi3",
    vision_config={**DEFAULT_VISION_CONFIG, "use_flash_attn": True},
    llm_config={
        **DEFAULT_INTERN_VL_LLM_CONFIG,
        "architectures": ["PhiForCausalLM"],
        "max_position_embeddings": 2048,
        "rope_scaling": {
            "type": "linear",
            "factor": 2.0
        }
    }
)
```

## 2. 配置驗證

### 2.1 基本驗證

```python
# 配置會自動進行驗證
try:
    config = VLMConfigFactory.create_vlm_config(
        "llava",
        vision_config={},  # 缺少必需的參數
        llm_config=DEFAULT_LLAVA_LLM_CONFIG
    )
except ValueError as e:
    print(f"驗證失敗: {e}")
```

### 2.2 特定模型的驗證

```python
# Phi-3 的 RoPE 配置驗證
try:
    config = VLMConfigFactory.create_vlm_config(
        "phi3",
        vision_config=DEFAULT_VISION_CONFIG,
        llm_config={
            **DEFAULT_INTERN_VL_LLM_CONFIG,
            "architectures": ["PhiForCausalLM"],
            "rope_scaling": {
                "type": "invalid",  # 無效的 RoPE 類型
                "factor": 2.0
            }
        }
    )
except ValueError as e:
    print(f"RoPE 配置驗證失敗: {e}")

# Intern-VL 的 Flash Attention 驗證
try:
    config = VLMConfigFactory.create_vlm_config(
        "intern-vl",
        vision_config=DEFAULT_VISION_CONFIG,  # 缺少 use_flash_attn
        llm_config=DEFAULT_INTERN_VL_LLM_CONFIG
    )
except ValueError as e:
    print(f"Flash Attention 配置驗證失敗: {e}")
```

## 3. 配置序列化

### 3.1 保存配置

```python
# 將配置保存為 JSON 文件
config = VLMConfigFactory.create_vlm_config(
    "llava",
    vision_config=DEFAULT_VISION_CONFIG,
    llm_config=DEFAULT_LLAVA_LLM_CONFIG
)

# 保存到文件
config.save_pretrained("./configs/llava")

# 或者保存為 JSON 字符串
config_json = config.to_json_string()
```

### 3.2 加載配置

```python
# 從文件加載配置
config = VLMConfigFactory.create_config_from_pretrained("./configs/llava")

# 從字典加載配置
config_dict = {
    "model_type": "llava",
    "vision_config": DEFAULT_VISION_CONFIG,
    "llm_config": DEFAULT_LLAVA_LLM_CONFIG
}
config = VLMConfigFactory.create_config_from_dict(config_dict)
```

## 4. 自定義配置

### 4.1 創建自定義配置類

```python
from core.configuration import BaseVLMConfig

class CustomVLMConfig(BaseVLMConfig):
    def __init__(self, vision_config: dict, llm_config: dict, custom_param: int = 100, **kwargs):
        super().__init__(vision_config, llm_config, **kwargs)
        self.model_type = "custom_vlm"
        self.custom_param = custom_param
    
    def _validate_vision_config(self):
        super()._validate_vision_config()
        # 添加自定義驗證邏輯
        if "custom_vision_param" not in self.vision_config:
            raise ValueError("Missing required key 'custom_vision_param' in vision_config")
    
    def _validate_llm_config(self):
        super()._validate_llm_config()
        # 添加自定義驗證邏輯
        if "custom_llm_param" not in self.llm_config:
            raise ValueError("Missing required key 'custom_llm_param' in llm_config")
```

### 4.2 註冊自定義配置

```python
from core.configuration import VLMConfigRegistry

# 註冊自定義配置
registry = VLMConfigRegistry()
registry.register_vlm_config("custom_vlm", CustomVLMConfig)

# 使用自定義配置
config = VLMConfigFactory.create_vlm_config(
    "custom_vlm",
    vision_config={
        **DEFAULT_VISION_CONFIG,
        "custom_vision_param": 200
    },
    llm_config={
        **DEFAULT_LLAVA_LLM_CONFIG,
        "custom_llm_param": 300
    },
    custom_param=400
)
```

## 5. 最佳實踐

### 5.1 配置管理

1. **使用默認配置**
   - 優先使用預定義的默認配置
   - 只在必要時覆蓋默認值
   - 保持配置的一致性

2. **配置驗證**
   - 始終進行配置驗證
   - 提供清晰的錯誤消息
   - 在適當的地方進行參數檢查

3. **配置序列化**
   - 使用 `save_pretrained` 保存配置
   - 使用 `from_pretrained` 加載配置
   - 保持配置文件的版本控制

### 5.2 錯誤處理

1. **驗證錯誤**
   ```python
   try:
       config = VLMConfigFactory.create_vlm_config(...)
   except ValueError as e:
       # 處理驗證錯誤
       logger.error(f"配置驗證失敗: {e}")
   except KeyError as e:
       # 處理缺少鍵的錯誤
       logger.error(f"缺少必需的配置項: {e}")
   ```

2. **類型錯誤**
   ```python
   try:
       config = VLMConfigFactory.create_vlm_config(
           "llava",
           vision_config="invalid",  # 應該是字典
           llm_config=DEFAULT_LLAVA_LLM_CONFIG
       )
   except TypeError as e:
       # 處理類型錯誤
       logger.error(f"配置類型錯誤: {e}")
   ```

### 5.3 性能考慮

1. **配置創建**
   - 重用配置實例
   - 使用默認配置
   - 避免頻繁創建配置

2. **配置驗證**
   - 按照邏輯順序進行驗證
   - 快速失敗
   - 緩存驗證結果

3. **配置序列化**
   - 只在必要時序列化
   - 使用高效的序列化方法
   - 考慮使用緩存

## 6. 常見問題

1. **Q: 如何添加新的模型配置？**
   A: 繼承 `BaseVLMConfig`，實現必要的驗證邏輯，然後註冊到註冊表。

2. **Q: 配置驗證失敗怎麼辦？**
   A: 檢查錯誤信息，確保所有參數符合要求。可以查看配置類的文檔了解具體的參數要求。

3. **Q: 如何處理模型特定的配置？**
   A: 在相應的配置類中實現 `_validate_vision_config` 和 `_validate_llm_config` 方法，添加特定的驗證邏輯。

4. **Q: 配置註冊表是全局的嗎？**
   A: 是的，註冊表使用單例模式，全局只有一個實例。所有模塊共享同一個註冊表。

5. **Q: 如何擴展默認配置？**
   A: 使用 `register_default_configs` 函數註冊新的默認配置，或使用字典合併操作擴展現有配置。 