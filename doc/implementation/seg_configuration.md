# 配置系統實現文檔

## 概述

本文檔詳細說明了分割模型配置系統的實現細節，主要基於 `core/configuration/seg_config.py` 文件。該配置系統提供了一個靈活且可擴展的框架，用於管理不同類型分割模型的配置。

## 核心組件

### 1. 配置註冊表 (SegConfigRegistry)

配置註冊表是一個類級別的單例，用於管理和註冊不同類型分割模型的默認配置。

```python
class SegConfigRegistry:
    _configs: ClassVar[Dict[str, Dict[str, Any]]] = {}
```

#### 主要功能：
- 註冊新的模型配置
- 獲取指定模型類型的默認配置
- 列出所有已註冊的模型類型

#### 使用示例：
```python
# 註冊新的模型配置
SegConfigRegistry.register("model_type", default_config)

# 獲取默認配置
config = SegConfigRegistry.get_default_config("model_type")

# 列出所有模型類型
model_types = SegConfigRegistry.list_models()
```

### 2. 配置基類 (BaseSegConfig)

配置基類是所有分割模型配置的抽象基類，繼承自 `transformers.PretrainedConfig`。

#### 主要特性：
- 支持基本參數驗證
- 支持自定義配置驗證
- 提供序列化和反序列化功能
- 支持配置的保存和加載

#### 關鍵參數：
```python
def __init__(
    self,
    seg_type: str,                    # 模型類型
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
)
```

#### 驗證邏輯：
1. 基本參數驗證 (`_validate_basic_params`)：
   - 驗證 patch_size
   - 驗證 image_size
   - 驗證整除關係
   - 驗證其他基本參數

2. 自定義配置驗證 (`_validate_custom_config`)：
   - 由子類實現
   - 驗證特定模型類型的配置要求

### 3. 配置工廠 (SegConfigFactory)

配置工廠類用於創建和管理不同類型的配置實例。

#### 主要功能：
- 註冊新的配置類
- 創建指定類型的配置實例

#### 使用示例：
```python
# 註冊新的配置類
SegConfigFactory.register_config("model_type", CustomConfig)

# 創建配置實例
config = SegConfigFactory.create_config("model_type", **kwargs)
```

### 4. 工廠函數 (create_seg_config)

提供了一個便捷的工廠函數，用於創建分割模型配置實例。

#### 使用示例：
```python
# 創建默認配置
config = create_seg_config(seg_type="sam2")

# 創建自定義配置
config = create_seg_config(
    seg_type="sam2",
    seg_config={"image_encoder": {...}}
)
```

## 默認配置

系統預設了兩種模型的默認配置：

### 1. SAM2 配置
```python
{
    "image_encoder": {
        "type": "vit",
        "depth": 12,
        "embed_dim": 768,
        # ... 其他參數
    },
    "prompt_encoder": {
        "embed_dim": 256,
        "image_embed_size": 1024,
        # ... 其他參數
    },
    "mask_decoder": {
        "transformer_dim": 256,
        "transformer_heads": 8,
        # ... 其他參數
    }
}
```

### 2. 其他分割模型配置
```python
{
    "encoder": {
        "type": "resnet",
        "depth": 50
    },
    "decoder": {
        "type": "fpn",
        "in_channels": [256, 512, 1024, 2048]
    }
}
```

## 擴展指南

### 添加新的模型配置

1. 創建新的配置類：
```python
class NewModelConfig(BaseSegConfig):
    model_type = "new_model"
    
    def _validate_custom_config(self) -> None:
        # 實現特定的配置驗證邏輯
        pass
```

2. 註冊默認配置：
```python
SegConfigRegistry.register("new_model", {
    # 定義默認配置
})
```

3. 註冊配置類：
```python
SegConfigFactory.register_config("new_model", NewModelConfig)
```

### 最佳實踐

1. 配置驗證：
   - 始終在配置類中實現適當的驗證邏輯
   - 使用清晰的錯誤消息
   - 按照邏輯順序進行驗證

2. 默認值：
   - 為所有參數提供合理的默認值
   - 在文檔中明確說明默認值的含義

3. 類型提示：
   - 使用類型提示提高代碼可讀性
   - 為所有公共方法添加文檔字符串

4. 錯誤處理：
   - 使用具體的異常類型
   - 提供有意義的錯誤消息
   - 在適當的地方進行參數驗證

## 注意事項

1. 配置合併：
   - 用戶配置會覆蓋默認配置
   - 合併時保持配置的層級結構

2. 序列化：
   - 所有配置都可以序列化為 JSON
   - 支持保存和加載配置

3. 兼容性：
   - 確保新添加的配置與現有系統兼容
   - 在更新配置時考慮向後兼容性

## 相關文件

- `core/configuration/seg_config.py`: 配置系統的主要實現
- `tests/test_seg_configuration.py`: 配置系統的單元測試
- `core/configuration/__init__.py`: 配置系統的導出接口 