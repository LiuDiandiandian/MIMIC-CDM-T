# 模型后端架构改进说明

## 概述

项目已将模型加载与推理逻辑重构为**可插拔的后端架构**，采用类似 OpenAI API 的统一接口设计，提升了代码的可维护性和扩展性。

## 核心改进

### 1. 新的后端体系（`models/backend.py`）

#### 后端类型
- **OpenAI**: `openai_api_key` 指定时自动选择，支持 GPT-3.5、GPT-4 等
- **Transformers**: 默认后端，支持任意 HF 模型（Llama、Mistral、Falcon 等）
- **ExLlama**: GPTQ 量化模型的高性能推理，`exllama=True` 时启用
- **vLLM**: 超高吞吐推理引擎，使用 `vllm://model_name` 前缀指定
- **Human**: 交互式人工输入，模型名为 `"Human"`

#### 统一配置（`GenerationConfig`）
所有后端共享统一的生成参数：
```python
GenerationConfig(
    temperature=0.01,      # 温度
    top_k=1,              # top-k 采样
    top_p=0.95,           # nucleus 采样
    max_new_tokens=512,   # 生成长度
    do_sample=True,
    repetition_penalty=1.2,
    seed=None,
    ...
)
```

### 2. 简化的 `CustomLLM` 接口

重构前后的对比：

**重构前**：
```python
# 大量 if/elif 分支，混乱的模型加载逻辑
if self.model_name == "Human":
    ...
elif self.openai_api_key:
    ...
elif "GPTQ" in self.model_name and self.exllama:
    ...
elif self.model_name.startswith("vllm://"):
    ...
# ... 更多分支
```

**重构后**：
```python
# 清晰的统一接口
def load_model(self, base_models: str) -> None:
    backend_config = BackendConfig(
        backend_type=self.backend_type,
        model_name=self.model_name,
        openai_api_key=self.openai_api_key,
        # ... 其他配置
    )
    self.backend = BackendFactory.create(backend_config)
    self.backend.load()
```

### 3. 向后兼容性

已通过属性代理保持向后兼容：
```python
@property
def tokenizer(self):
    if self.backend and hasattr(self.backend, 'tokenizer'):
        return self.backend.tokenizer
    return None

@property
def model(self):
    if self.backend and hasattr(self.backend, 'model'):
        return self.backend.model
    return None
```

现有代码无需改动，仍可正常使用 `llm.tokenizer` 等接口。

## 使用示例

### 1. 使用 OpenAI API
```python
from models.models import CustomLLM

llm = CustomLLM(
    model_name="gpt-4",
    openai_api_key="sk-...",
    max_context_length=8192,
)
llm.load_model("")  # 无需 base_models 路径
```

### 2. 使用本地 Transformers 模型
```python
llm = CustomLLM(
    model_name="meta-llama/Llama-2-70B-Chat",
    max_context_length=4096,
)
llm.load_model("/path/to/base_models")
```

### 3. 使用 GPTQ 量化模型（高性能）
```python
llm = CustomLLM(
    model_name="TheBloke/Llama-2-70B-Chat-GPTQ",
    exllama=True,  # 启用 ExLlama 后端
    max_context_length=4096,
)
llm.load_model("/path/to/base_models")
```

### 4. 使用 vLLM 加速
```python
llm = CustomLLM(
    model_name="vllm://meta-llama/Llama-2-70B-Chat",
    max_context_length=4096,
)
llm.load_model("/path/to/base_models")
```

### 5. 交互式人工输入
```python
llm = CustomLLM(
    model_name="Human",
)
llm.load_model("")
```

## 扩展：添加自定义后端

若需支持新的模型后端（如 Anthropic Claude、本地 vLLM 等），只需：

1. 在 `models/backend.py` 中继承 `LLMBackend`：
```python
class MyCustomBackend(LLMBackend):
    def load(self) -> None:
        # 模型加载逻辑
        pass
    
    def generate(self, prompt, stop_sequences, gen_config) -> str:
        # 推理逻辑
        pass
```

2. 注册后端：
```python
BackendFactory.register("my_backend", MyCustomBackend)
```

3. 使用：
```python
llm = CustomLLM(
    model_name="my-model",
)
llm.backend_type = "my_backend"  # 显式指定
llm.load_model(base_models)
```

## 配置文件支持

在 `configs/` 中可直接指定后端参数：

```yaml
# configs/config.yaml
backend_type: "transformers"  # 或 openai, exllama, vllm
exllama: false
load_in_4bit: true
max_context_length: 4096
```

## 迁移指南（如果之前有自定义 CustomLLM）

若你之前扩展或修改过 `CustomLLM` 类：

1. **属性访问**：`llm.tokenizer`、`llm.model`、`llm.generator` 仍可用
2. **方法调用**：`llm.load_model()` 和 `llm._call()` 签名不变
3. **新增选项**：`backend_type` 可显式指定后端类型

基本上无需改动，可直接升级。

## 后端架构图

```
CustomLLM (LangChain LLM interface)
    ↓
BackendFactory (后端工厂)
    ↓
Backend (抽象基类)
    ├── OpenAIBackend
    ├── TransformersBackend
    ├── ExLlamaBackend
    ├── VLLMBackend
    └── HumanBackend
```

## 性能建议

| 场景 | 推荐后端 | 理由 |
|------|---------|------|
| API 调用（GPT-4） | OpenAI | 稳定性、功能完整 |
| 本地推理（无量化） | Transformers | 通用、易于扩展 |
| 本地推理（GPTQ）  | ExLlama | 高性能推理 |
| 超大规模服务 | vLLM | 高吞吐、支持并发 |

## 常见问题

**Q: 与旧代码兼容吗？**  
A: 是的，完全兼容。所有公开接口保持不变。

**Q: 如何指定 4-bit 量化？**  
A: 使用 `load_in_4bit=True`，TransformersBackend 会自动使用 BitsAndBytes。

**Q: 支持 LoRA 微调吗？**  
A: 当前 ExLlamaBackend 预留了接口。其他后端可在 `load()` 中加载 LoRA。

**Q: 能否同时使用多个后端？**  
A: 可以，创建多个 `CustomLLM` 实例，分别指定不同的后端类型。
