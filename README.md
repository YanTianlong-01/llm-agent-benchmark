# LLM Agent Benchmark

基于 [EvalScope](https://evalscope.readthedocs.io/) 的 **BFCL-v3 / BFCL-v4** 评测脚本，用于评估大语言模型的 Function Calling / Tool Calling 能力。

## 目录

- [简介](#简介)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [完整参数说明](#完整参数说明)
- [评测子集说明](#评测子集说明)
- [测试结果示例](#测试结果示例)
- [常见问题](#常见问题)

---

## 简介

本项目提供了一个封装脚本 `run_bfcl_evalscope.py`，基于 EvalScope 框架运行 [Berkeley Function Calling Leaderboard (BFCL)](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html) v3/v4 评测。

**核心特性：**

- 自动从 OpenAI-compatible API 的 `/v1/models` 接口检测模型名称
- 支持 BFCL-v3 全部 17 个子集（单轮 / 多轮 / 实时 / 相关性判断）
- 支持 BFCL-v4 新增的 Web Search 和 Memory 子集
- 灵活指定子集子集，支持快速测试（`--limit`）和正式评测
- 区分原生 Function Calling 模型与非 FC 模型的评测模式

## 环境准备

### 1. 安装 Python 依赖

```bash
# 推荐 Python 3.11+
pip install -U evalscope
pip install bfcl-eval==2025.10.27.1
pip install soundfile
```

> **注意：** `bfcl-eval` 版本需与 EvalScope 兼容。如遇版本冲突，可尝试：
> ```bash
> pip install bfcl-eval --upgrade
> ```

### 2. 准备模型服务

启动一个 OpenAI-compatible API 服务，例如使用 `llama-cpp-server`：

```bash
llama-server \
  -m /path/to/your-model.gguf \
  --host 127.0.0.1 \
  --port 1235 \
  -c 32000 \
  --ctx-size 131072
```

也可以用 Ollama、LM Studio、vLLM 等任何提供 OpenAI-compatible API 的后端。

## 快速开始

### 快速测试（指定子集 + 限制条数）

```bash
python run_bfcl_evalscope.py \
  --api-url http://127.0.0.1:1235/v1 \
  --api-key EMPTY \
  --subsets parallel_multiple javascript live_parallel_multiple irrelevance
```

这是「基础 4 项」快速评测，覆盖 AST_NON_LIVE、AST_LIVE 和 RELEVANCE 三大类。

### 完整评测（所有子集）

```bash
python run_bfcl_evalscope.py \
  --api-url http://127.0.0.1:1235/v1 \
  --api-key EMPTY \
```

`--limit 0` 表示不限制条数，跑完整数据集。

### 多轮 Agent 评测

```bash
# 基础多轮
python run_bfcl_evalscope.py \
  --api-url http://127.0.0.1:1235/v1 \
  --api-key EMPTY \
  --subsets multi_turn_base

# 长上下文多轮
python run_bfcl_evalscope.py \
  --api-url http://127.0.0.1:1235/v1 \
  --api-key EMPTY \
  --subsets multi_turn_long_context
```

### 使用远端 API

```bash
python run_bfcl_evalscope.py \
  --api-url https://api.deepseek.com/v1 \
  --model-name deepseek-v4-flash \
  --api-key sk-your-key-here \
  --subsets parallel_multiple javascript live_parallel_multiple irrelevance
```

### 非 Function Calling 模型

如果你的模型/API 不支持 OpenAI tools/tool_calls 格式：

```bash
python run_bfcl_evalscope.py \
  --api-url http://127.0.0.1:1235/v1 \
  --not-fc-model
```

## 完整参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model-name` | *(自动检测)* | 手动指定模型名。不填时自动从 `/v1/models` 检测 |
| `--model-tag` | *(同 model-name)* | 结果目录的命名标签 |
| `--api-url` | `http://127.0.0.1:1235/v1` | OpenAI-compatible API 地址 |
| `--api-key` | `EMPTY` | API key。本地服务通常填 `EMPTY` |
| `--output-dir` | `results_bfcl` | 结果输出目录 |
| `--temperature` | `0.0` | 评测温度。Agent/tool calling 对比建议设为 0 |
| `--max-tokens` | `32000` | 最大输出 token 数 |
| `--eval-batch-size` | `1` | 评测批次大小 |
| `--is-fc-model` | `True` | 是否按原生 function calling 模型评测 |
| `--not-fc-model` | `False` | API 不支持 OpenAI tools 时加上此参数 |
| `--subsets` | *(全部 17 个)* | 指定只跑哪些 BFCL-v3 子集 |

## 评测子集说明

BFCL-v3 的 17 个子集分为 4 大类：

### AST_NON_LIVE（经典函数调用）

| 子集 | 测试内容 |
|------|----------|
| `simple` | 单轮单函数调用 |
| `multiple` | 单轮多函数调用 |
| `parallel` | 并行函数调用 |
| `parallel_multiple` | 并行 + 多函数组合 |
| `java` | Java 风格函数调用 |
| `javascript` | JavaScript 风格函数调用 |

### AST_LIVE（实时/动态数据）

| 子集 | 测试内容 |
|------|----------|
| `live_simple` | 实时数据下单函数调用 |
| `live_multiple` | 实时数据下多函数调用 |
| `live_parallel` | 实时数据下并行调用 |
| `live_parallel_multiple` | 实时数据下复杂并行多调用 |

### RELEVANCE（工具调用判断）

| 子集 | 测试内容 |
|------|----------|
| `irrelevance` | 无关请求不应调用工具 |
| `live_relevance` | 相关请求应调用工具 |
| `live_irrelevance` | live 场景下不该调用时保持克制 |

### MULTI_TURN（多轮 Agent）

| 子集 | 测试内容 |
|------|----------|
| `multi_turn_base` | 基础多轮工具调用 |
| `multi_turn_miss_func` | 工具缺失时的处理能力 |
| `multi_turn_miss_param` | 参数缺失时应追问而非编造 |
| `multi_turn_long_context` | 长上下文中的多轮工具调用 |

### 汇总指标

评测结果会按以下维度聚合：

| 汇总项 | 包含内容 |
|--------|----------|
| `NON_LIVE` | simple, multiple, parallel, parallel_multiple, java, javascript |
| `LIVE` | live_simple, live_multiple, live_parallel, live_parallel_multiple |
| `RELEVANCE` | irrelevance, live_relevance, live_irrelevance |
| `MULTI_TURN` | multi_turn_base, multi_turn_miss_func, multi_turn_miss_param, multi_turn_long_context |
| `OVERALL` | 所有子集综合 |

### BFCL-v4 新增子集

BFCL-v4 在 v3 基础上新增：

| 子集 | 测试内容 |
|------|----------|
| `web_search_base` | 基础 Web 搜索 |
| `web_search_no_snippet` | 无摘要的 Web 搜索 |
| `memory_kv` | 键值记忆 |
| `memory_vector` | 向量记忆 |
| `memory_rec_sum` | 摘要记忆 |

使用方法：将 `datasets=['bfcl_v3']` 改为 `datasets=['bfcl_v4']`，并配置 `SERPAPI_API_KEY`。

## 测试结果示例

[完整测试结果网站](https://yantianlong-01.github.io/llm_benchmark/)

以下为部分 Qwen3.6-27B 量化模型的评测结果（基础 4 项，`--limit 50`）：

| 模型 | NON_LIVE | LIVE | RELEVANCE | OVERALL |
|------|----------|------|-----------|---------|
| ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM | 0.865 | 0.8333 | 0.90 | **0.8492** |
| Ununnilium/qwen3.6-27b-IQ4_XS-pure | 0.865 | 0.8333 | 0.90 | **0.8492** |
| unsloth/Qwen3.6-27B-UD-Q3_K_XL | 0.86 | 0.7917 | 0.88 | 0.8258 |
| ManniX-ITA/Qwen3.6-27B-Omnimerge-v4-IQ3_M | 0.84 | 0.8333 | 0.86 | 0.8367 |
| unsloth/Qwen3.6-27B-UD-IQ3_XXS | 0.85 | 0.75 | 0.84 | 0.80 |

> 多轮评测结果（ggufbench/Qwen3.6-27B-4bpw-16GB-VRAM）：
> - `multi_turn_base`: 0.52（耗时 ~1h20m）
> - `multi_turn_long_context`: 0.44（耗时 ~1h29m）

## 常见问题

### Q: 自动检测模型名失败

确保你的 API 服务支持 `/v1/models` 端点。如果不支持，用 `--model-name` 手动指定：

```bash
python run_bfcl_evalscope.py --model-name "my-custom-model"
```

### Q: 评测结果目录在哪？

默认在 `results_bfcl/<模型名>/` 下。可通过 `--output-dir` 修改。

### Q: 多轮评测为什么很慢？

多轮任务需要模型与工具多轮交互，每轮都需要完整的 API 请求。本地 CPU/GPU 推理速度直接影响耗时。建议：
- 确保 `--max-tokens` 足够大（默认 32000）
- `--temperature` 设为 0 保证结果可复现

### Q: 如何切换到 BFCL-v4？

修改脚本中 `datasets=["bfcl_v3"]` 为 `datasets=["bfcl_v4"]`，并更新 `subset_list`。Web Search 子集需要设置 `SERPAPI_API_KEY` 环境变量。

---

## 参考

- [EvalScope BFCL-v3 文档](https://evalscope.readthedocs.io/en/latest/third_party/bfcl_v3.html)
- [Gorilla BFCL-v3 官方博客](https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html)
- [EvalScope GitHub](https://github.com/modelscope/evalscope)
- [BFCL GitHub](https://github.com/ShishirPatil/gorilla)
