# MIMIC 临床决策框架

## 1. 项目简介

该项目用于评估大模型（LLM）在临床决策任务中的表现。主要目标是模拟真实临床环境，让模型根据患者信息进行问诊、分析并给出诊断与处理建议。

本项目包含两个主要任务：
- `run.py`：临床决策过程任务，模型在多个回合中逐步获取信息并给出诊断。
- `run_full_info.py`：全信息诊断任务，所有信息一次性提供，模型直接给出诊断。

## 2. 环境准备

建议使用 Python 3.10 创建虚拟环境。

```powershell
python -m venv venv
venv\Scripts\activate
pip install --no-deps -r requirements.txt
```

如果使用 CUDA 本地模型，请先配置 CUDA_HOME：

```powershell
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
```

## 3. 配置路径

项目配置文件位于 `configs/`。其中主要路径配置在 `configs/paths/example.yaml`：

```yaml
base_mimic: 
base_models: 
lab_test_mapping_path: 
local_logging_dir: 
```

请将其修改为实际路径：
- `base_mimic`：MIMIC 或预处理数据目录
- `base_models`：本地模型缓存目录
- `lab_test_mapping_path`：化验映射文件路径
- `local_logging_dir`：结果保存目录

## 4. 模型配置

模型配置文件位于 `configs/model/`，例如：
- `GPT4.yaml`
- `GPT35Turbo.yaml`
- `WizardLM70B.yaml`
- `Llama3Instruct70B.yaml`

默认模型由 `configs/config.yaml` 的 `defaults` 设置决定。可以在命令行中覆盖：

```powershell
python run.py model=GPT4
```

### OpenAI 模型
如果使用 OpenAI，请在对应模型配置文件或命令行中指定：
- `openai_api_key`
- `model_name` 为 `gpt-4` 或其他可用模型

## 5. 运行示例

### 5.1 运行临床决策任务

```powershell
python run.py pathology=appendicitis model=GPT4
```

### 5.2 运行全信息诊断任务

```powershell
python run_full_info.py pathology=cholecystitis model=GPT4
```

### 5.3 运行本地模型

```powershell
python run.py pathology=pancreatitis model=WizardLM70B
```

## 6. 常用参数说明

| 参数 | 示例 | 说明 |
|------|------|------|
| `pathology` | `appendicitis` | 选择病种 |
| `model` | `GPT4`、`WizardLM70B` | 选择模型 |
| `summarize` | `True` | 是否自动摘要会话 |
| `fewshot` | `False` | 是否启用 few-shot 示例 |
| `include_ref_range` | `False` | 是否显示实验室参考范围 |
| `bin_lab_results` | `False` | 是否将检验结果离散化 |
| `provide_diagnostic_criteria` | `False` | 是否提供诊断标准工具 |
| `only_abnormal_labs` | `False` | 是否仅提供异常化验结果 |
| `seed` | `2023` | 随机种子 |
| `first_patient` | `hadm_id` | 仅从指定患者开始 |

## 7. 输出结果

运行后，结果会保存在 `local_logging_dir` 下的自动生成子目录中，通常包含：
- `*_results.pkl`：模型输出结果
- `*_eval.pkl`：评估结果
- `*.log`：运行日志
