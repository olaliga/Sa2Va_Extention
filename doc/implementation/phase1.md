# Sa2VA 重构第一阶段实现指南

## 1. 阶段目标

第一阶段的主要目标是实现基础架构，包括：
1. 实现基础配置类
2. 实现基础模型类
3. 确保与 Transformers 的兼容性

## 2. 具体任务

### 2.1 基础配置类实现

1. **创建配置目录**
   ```
   sa2va_models_refactor/
   └── core/
       └── configuration/
           ├── __init__.py
           ├── vlm_config.py
           ├── seg_config.py
           └── sa2va_config.py
   ```

2. **实现 BaseVLMConfig**
   - 继承 `PretrainedConfig`
   - 实现基本配置参数
   - 添加配置验证

3. **实现 BaseSegConfig**
   - 继承 `PretrainedConfig`
   - 实现基本配置参数
   - 添加配置验证

4. **实现 Sa2VAConfig**
   - 继承 `PretrainedConfig`
   - 整合 VLM 和分割模型配置
   - 实现配置序列化

### 2.2 基础模型类实现

1. **创建模型目录**
   ```
   sa2va_models_refactor/
   └── core/
       └── base/
           ├── __init__.py
           ├── base_vlm.py
           ├── base_seg.py
           └── base_fusion.py
   ```

2. **实现 BaseVLM**
   - 继承 `PreTrainedModel`
   - 实现基本接口
   - 添加模型验证

3. **实现 BaseSeg**
   - 继承 `PreTrainedModel`
   - 实现基本接口
   - 添加模型验证

4. **实现 BaseFusion**
   - 继承 `nn.Module`
   - 实现特征融合接口
   - 添加融合策略

### 2.3 Transformers 集成

1. **创建模型注册**
   ```python
   # 在 __init__.py 中注册模型
   from transformers import AutoConfig, AutoModel
   
   AutoConfig.register("sa2va", Sa2VAConfig)
   AutoModel.register(Sa2VAConfig, Sa2VAModel)
   ```

2. **实现模型加载**
   ```python
   # 实现 from_pretrained 方法
   @classmethod
   def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
       config = kwargs.pop("config", None)
       if config is None:
           config = AutoConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)
       return super().from_pretrained(pretrained_model_name_or_path, config=config, *model_args, **kwargs)
   ```

3. **实现模型保存**
   ```python
   # 实现 save_pretrained 方法
   def save_pretrained(self, save_directory, **kwargs):
       config = self.config
       config.save_pretrained(save_directory)
       super().save_pretrained(save_directory, **kwargs)
   ```

## 3. 实现步骤

### 3.1 配置类实现

1. **BaseVLMConfig**
   ```python
   # vlm_config.py
   from transformers import PretrainedConfig
   
   class BaseVLMConfig(PretrainedConfig):
       model_type = "base_vlm"
       
       def __init__(
           self,
           vlm_type: str = "llava",
           vlm_config: dict = None,
           **kwargs
       ):
           super().__init__(**kwargs)
           self.vlm_type = vlm_type
           self.vlm_config = vlm_config or {}
           
       def to_dict(self):
           output = super().to_dict()
           output["vlm_type"] = self.vlm_type
           output["vlm_config"] = self.vlm_config
           return output
   ```

2. **BaseSegConfig**
   ```python
   # seg_config.py
   from transformers import PretrainedConfig
   
   class BaseSegConfig(PretrainedConfig):
       model_type = "base_seg"
       
       def __init__(
           self,
           seg_type: str = "sam2",
           seg_config: dict = None,
           **kwargs
       ):
           super().__init__(**kwargs)
           self.seg_type = seg_type
           self.seg_config = seg_config or {}
           
       def to_dict(self):
           output = super().to_dict()
           output["seg_type"] = self.seg_type
           output["seg_config"] = self.seg_config
           return output
   ```

3. **Sa2VAConfig**
   ```python
   # sa2va_config.py
   from transformers import PretrainedConfig
   
   class Sa2VAConfig(PretrainedConfig):
       model_type = "sa2va"
       
       def __init__(
           self,
           vlm_config: BaseVLMConfig = None,
           seg_config: BaseSegConfig = None,
           fusion_strategy: str = "attention",
           **kwargs
       ):
           super().__init__(**kwargs)
           self.vlm_config = vlm_config or BaseVLMConfig()
           self.seg_config = seg_config or BaseSegConfig()
           self.fusion_strategy = fusion_strategy
           
       def to_dict(self):
           output = super().to_dict()
           output["vlm_config"] = self.vlm_config.to_dict()
           output["seg_config"] = self.seg_config.to_dict()
           output["fusion_strategy"] = self.fusion_strategy
           return output
   ```

### 3.2 模型类实现

1. **BaseVLM**
   ```python
   # base_vlm.py
   from transformers import PreTrainedModel
   from transformers.modeling_outputs import BaseModelOutput
   
   class BaseVLM(PreTrainedModel):
       config_class = BaseVLMConfig
       base_model_prefix = "vlm"
       
       def __init__(self, config):
           super().__init__(config)
           self.config = config
           
       def forward(
           self,
           pixel_values=None,
           input_ids=None,
           attention_mask=None,
           **kwargs
       ) -> BaseModelOutput:
           raise NotImplementedError
           
       def get_visual_features(self):
           raise NotImplementedError
   ```

2. **BaseSeg**
   ```python
   # base_seg.py
   from transformers import PreTrainedModel
   from transformers.modeling_outputs import BaseModelOutput
   
   class BaseSeg(PreTrainedModel):
       config_class = BaseSegConfig
       base_model_prefix = "seg"
       
       def __init__(self, config):
           super().__init__(config)
           self.config = config
           
       def forward(
           self,
           pixel_values=None,
           prompts=None,
           **kwargs
       ) -> BaseModelOutput:
           raise NotImplementedError
           
       def predict_masks(self, image_features, prompts=None):
           raise NotImplementedError
   ```

3. **BaseFusion**
   ```python
   # base_fusion.py
   import torch.nn as nn
   
   class BaseFusion(nn.Module):
       def __init__(self, config):
           super().__init__()
           self.config = config
           
       def forward(self, vlm_features, seg_features):
           raise NotImplementedError
   ```

### 3.3 模型注册

```python
# __init__.py
from transformers import AutoConfig, AutoModel
from .configuration import Sa2VAConfig
from .modeling import Sa2VAModel

AutoConfig.register("sa2va", Sa2VAConfig)
AutoModel.register(Sa2VAConfig, Sa2VAModel)
```

## 4. 测试计划

### 4.1 单元测试

1. **配置测试**
   ```python
   def test_vlm_config():
       config = BaseVLMConfig()
       assert config.vlm_type == "llava"
       assert isinstance(config.vlm_config, dict)
       
   def test_seg_config():
       config = BaseSegConfig()
       assert config.seg_type == "sam2"
       assert isinstance(config.seg_config, dict)
       
   def test_sa2va_config():
       config = Sa2VAConfig()
       assert isinstance(config.vlm_config, BaseVLMConfig)
       assert isinstance(config.seg_config, BaseSegConfig)
   ```

2. **模型测试**
   ```python
   def test_base_vlm():
       config = BaseVLMConfig()
       model = BaseVLM(config)
       assert isinstance(model, PreTrainedModel)
       
   def test_base_seg():
       config = BaseSegConfig()
       model = BaseSeg(config)
       assert isinstance(model, PreTrainedModel)
   ```

### 4.2 集成测试

1. **模型加载测试**
   ```python
   def test_model_loading():
       model = AutoModel.from_pretrained(
           "ByteDance/Sa2VA-1B",
           trust_remote_code=True
       )
       assert isinstance(model, Sa2VAModel)
   ```

2. **模型保存测试**
   ```python
   def test_model_saving():
       model = Sa2VAModel(config)
       model.save_pretrained("test_save")
       assert os.path.exists("test_save/config.json")
       assert os.path.exists("test_save/pytorch_model.bin")
   ```

## 5. 注意事项

### 5.1 实现注意事项

1. **配置类**
   - 确保所有配置参数都有默认值
   - 实现配置验证
   - 支持配置序列化

2. **模型类**
   - 正确继承 Transformers 基类
   - 实现所有必需方法
   - 保持接口一致性

3. **模型注册**
   - 正确注册模型和配置
   - 处理版本兼容性
   - 提供错误处理

### 5.2 测试注意事项

1. **单元测试**
   - 测试所有公共接口
   - 测试边界情况
   - 测试错误处理

2. **集成测试**
   - 测试模型加载
   - 测试模型保存
   - 测试模型推理

### 5.3 文档注意事项

1. **代码文档**
   - 添加详细的文档字符串
   - 说明参数和返回值
   - 提供使用示例

2. **API 文档**
   - 说明所有公共接口
   - 提供使用指南
   - 添加注意事项

## 6. 后续计划

1. **第二阶段准备**
   - 迁移现有 LLaVA 实现
   - 迁移现有 SAM2 实现
   - 实现特征融合策略

2. **性能优化**
   - 优化模型加载
   - 优化内存使用
   - 优化计算效率

3. **功能扩展**
   - 添加新模型支持
   - 添加新功能
   - 完善文档 