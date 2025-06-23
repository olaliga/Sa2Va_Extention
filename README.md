# Sa2VA Models Refactor

一個模塊化的視覺語言模型和分割框架。

## 安裝

### 從源碼安裝

```bash
# 克隆倉庫
git clone https://github.com/yourusername/sa2va-models-refactor.git
cd sa2va-models-refactor

# 安裝依賴
pip install -e .
```

### 開發環境設置

```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
.\venv\Scripts\activate  # Windows

# 安裝開發依賴
pip install -e ".[dev]"
```

## 運行測試

```bash
# 運行所有測試
python -m unittest discover tests

# 運行特定測試文件
python -m unittest tests/test_configuration.py
```

## 使用示例

```python
from core.configuration import Sa2VAConfig

# 創建默認配置
config = Sa2VAConfig()

# 創建自定義配置
custom_config = Sa2VAConfig(
    vlm_config=BaseVLMConfig(
        vlm_type="llava",
        vlm_config={
            "vision_tower": "clip-vit-large-patch14",
            "mm_projector": {
                "hidden_size": 1024,
                "output_size": 4096
            }
        }
    ),
    seg_config=BaseSegConfig(
        seg_type="sam2",
        seg_config={
            "image_encoder": {
                "type": "vit",
                "depth": 12,
                "embed_dim": 768
            }
        }
    ),
    fusion_strategy="attention",
    fusion_config={
        "num_heads": 8,
        "dropout": 0.1
    }
)

# 保存配置
config.save_pretrained("path/to/save")

# 加載配置
loaded_config = Sa2VAConfig.from_pretrained("path/to/load")
```

## 開發指南

1. 確保代碼風格符合 PEP 8
2. 添加適當的類型提示
3. 為所有公共 API 添加文檔字符串
4. 編寫單元測試
5. 更新文檔

## 許可證

MIT License
