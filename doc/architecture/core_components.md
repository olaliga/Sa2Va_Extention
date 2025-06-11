# Sa2VA 核心组件设计

## 1. 基础模型类

### 1.1 BaseVLM (视觉语言模型基类)

```python
class BaseVLM(PreTrainedModel):
    """
    视觉语言模型基类，继承自 Transformers 的 PreTrainedModel
    """
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
        """
        前向传播方法
        
        Args:
            pixel_values: 图像输入
            input_ids: 文本输入 ID
            attention_mask: 注意力掩码
            **kwargs: 其他参数
            
        Returns:
            BaseModelOutput: 包含模型输出的标准格式
        """
        raise NotImplementedError
        
    def get_visual_features(self):
        """
        获取视觉特征
        
        Returns:
            torch.Tensor: 视觉特征
        """
        raise NotImplementedError
```

### 1.2 BaseSeg (分割模型基类)

```python
class BaseSeg(PreTrainedModel):
    """
    分割模型基类，继承自 Transformers 的 PreTrainedModel
    """
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
        """
        前向传播方法
        
        Args:
            pixel_values: 图像输入
            prompts: 分割提示
            **kwargs: 其他参数
            
        Returns:
            BaseModelOutput: 包含模型输出的标准格式
        """
        raise NotImplementedError
        
    def predict_masks(self, image_features, prompts=None):
        """
        预测分割掩码
        
        Args:
            image_features: 图像特征
            prompts: 分割提示
            
        Returns:
            torch.Tensor: 分割掩码
        """
        raise NotImplementedError
```

### 1.3 BaseFusion (特征融合基类)

```python
class BaseFusion(nn.Module):
    """
    特征融合基类
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
    def forward(self, vlm_features, seg_features):
        """
        融合特征
        
        Args:
            vlm_features: VLM 特征
            seg_features: 分割模型特征
            
        Returns:
            torch.Tensor: 融合后的特征
        """
        raise NotImplementedError
```

## 2. 配置类

### 2.1 BaseVLMConfig

```python
class BaseVLMConfig(PretrainedConfig):
    """
    VLM 配置基类
    """
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
```

### 2.2 BaseSegConfig

```python
class BaseSegConfig(PretrainedConfig):
    """
    分割模型配置基类
    """
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
```

### 2.3 Sa2VAConfig

```python
class Sa2VAConfig(PretrainedConfig):
    """
    Sa2VA 主配置类
    """
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
```

## 3. 主模型类

### 3.1 Sa2VAModel

```python
class Sa2VAModel(PreTrainedModel):
    """
    Sa2VA 主模型类
    """
    config_class = Sa2VAConfig
    base_model_prefix = "sa2va"
    
    def __init__(self, config):
        super().__init__(config)
        self.vlm = self._init_vlm()
        self.seg_model = self._init_seg_model()
        self.fusion = self._init_fusion()
        
    def _init_vlm(self):
        """初始化 VLM"""
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            self.config.vlm_config.vlm_type,
            config=self.config.vlm_config.vlm_config
        )
        
    def _init_seg_model(self):
        """初始化分割模型"""
        from transformers import AutoModel
        return AutoModel.from_pretrained(
            self.config.seg_config.seg_type,
            config=self.config.seg_config.seg_config
        )
        
    def forward(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        **kwargs
    ) -> BaseModelOutput:
        """
        前向传播方法
        
        Args:
            pixel_values: 图像输入
            input_ids: 文本输入 ID
            attention_mask: 注意力掩码
            **kwargs: 其他参数
            
        Returns:
            BaseModelOutput: 包含模型输出的标准格式
        """
        # VLM 处理
        vlm_outputs = self.vlm(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs
        )
        
        # 分割模型处理
        seg_outputs = self.seg_model(
            pixel_values=pixel_values,
            **kwargs
        )
        
        # 特征融合
        fused_features = self.fusion(
            vlm_outputs.last_hidden_state,
            seg_outputs.last_hidden_state
        )
        
        return BaseModelOutput(
            last_hidden_state=fused_features,
            hidden_states=vlm_outputs.hidden_states,
            attentions=vlm_outputs.attentions
        )
```

### 3.2 Sa2VAProcessor

```python
class Sa2VAProcessor(ProcessorMixin):
    """
    Sa2VA 处理器类
    """
    attributes = ["image_processor", "tokenizer"]
    
    def __init__(
        self,
        image_processor,
        tokenizer,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
    def __call__(self, images=None, text=None, **kwargs):
        """
        处理输入数据
        
        Args:
            images: 图像输入
            text: 文本输入
            **kwargs: 其他参数
            
        Returns:
            dict: 处理后的输入数据
        """
        if images is not None:
            pixel_values = self.image_processor(images, return_tensors="pt")
        if text is not None:
            text_inputs = self.tokenizer(text, return_tensors="pt")
            
        return {
            "pixel_values": pixel_values,
            **text_inputs
        }
```

## 4. 工厂类

### 4.1 ModelFactory

```python
class ModelFactory:
    """
    模型工厂类，用于创建模型实例
    """
    @staticmethod
    def create_vlm(config):
        """
        创建 VLM 实例
        
        Args:
            config: VLM 配置
            
        Returns:
            BaseVLM: VLM 实例
        """
        vlm_registry = {
            "llava": LLaVAModel,
            "qwen_vl": QwenVLModel,
            "intern_vl": InternVLModel
        }
        return vlm_registry[config.model_type](config)
    
    @staticmethod
    def create_seg_model(config):
        """
        创建分割模型实例
        
        Args:
            config: 分割模型配置
            
        Returns:
            BaseSeg: 分割模型实例
        """
        seg_registry = {
            "sam2": SAM2Model,
            "other_seg": OtherSegModel
        }
        return seg_registry[config.model_type](config)
```

## 5. 关键接口说明

### 5.1 模型接口

1. **VLM 接口**
   - `forward`: 前向传播
   - `get_visual_features`: 获取视觉特征
   - `encode_image`: 图像编码
   - `encode_text`: 文本编码

2. **分割模型接口**
   - `forward`: 前向传播
   - `predict_masks`: 预测分割掩码
   - `encode_image`: 图像编码

3. **特征融合接口**
   - `forward`: 特征融合
   - `get_fusion_config`: 获取融合配置

### 5.2 配置接口

1. **VLM 配置**
   - `vlm_type`: 模型类型
   - `vlm_config`: 模型配置

2. **分割模型配置**
   - `seg_type`: 模型类型
   - `seg_config`: 模型配置

3. **主配置**
   - `vlm_config`: VLM 配置
   - `seg_config`: 分割模型配置
   - `fusion_strategy`: 融合策略

### 5.3 处理器接口

1. **图像处理**
   - `process_image`: 处理图像
   - `process_batch`: 批处理图像

2. **文本处理**
   - `process_text`: 处理文本
   - `process_batch`: 批处理文本

## 6. 实现注意事项

1. **模型实现**
   - 确保继承正确的基类
   - 实现所有必要的接口方法
   - 保持与 Transformers 的兼容性

2. **配置实现**
   - 提供合理的默认值
   - 支持配置序列化
   - 实现配置验证

3. **处理器实现**
   - 处理各种输入格式
   - 提供批处理支持
   - 实现数据预处理和后处理

4. **工厂实现**
   - 维护模型注册表
   - 提供模型创建方法
   - 支持动态注册新模型 