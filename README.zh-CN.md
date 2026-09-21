# Yida-Model-14B

[English](README.md) | **中文**

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Yida--Model--14B-ffd21e)](https://huggingface.co/rwang220/Yida-Model-14B)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/badge/repo-public-success)](https://github.com/rwang220/yida_model)

Yida-Model-14B 是在 [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) 上，用内部中文医疗指令数据做 LoRA 监督微调后，再合并得到的全量权重。Hub 上是合并后的 `bfloat16` 权重（**不是** PEFT adapter）。

本模型仅供研究与工程评估。它**不是**医疗器械，不得单独作为临床决策依据。

| 资源 | 链接 |
| --- | --- |
| 权重与 Hub 模型卡 | [huggingface.co/rwang220/Yida-Model-14B](https://huggingface.co/rwang220/Yida-Model-14B) |
| 本仓库（说明、配置、分词器） | [github.com/rwang220/yida_model](https://github.com/rwang220/yida_model) |
| 英文官网 | [`website/index.html`](website/index.html) |
| 基座模型 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |

## 本仓库包含什么

Git **不存放**约 27.5 GiB 的权重分片，权重只在 Hugging Face 上。

| 路径 | 说明 |
| --- | --- |
| `README.md` / `README.zh-CN.md` | GitHub 文档（英文 / 本中文页） |
| `LICENSE` | Apache-2.0（继承自 Qwen3-14B） |
| `config.json` / `generation_config.json` | 模型结构与默认采样 |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` | 分词器 |
| `chat_template.jinja` | Qwen3 思维链 / 工具调用对话模板 |
| `model.safetensors.index.json` | Hub 上六个权重分片的索引 |
| `examples/inference.py` | 从 Hub 加载并生成 |
| `website/index.html` | 英文官网（单文件离线页） |

下载权重：

```bash
huggingface-cli download rwang220/Yida-Model-14B --local-dir ./weights
```

## 模型信息

| 项目 | 内容 |
| --- | --- |
| 开发者 | [rwang220](https://huggingface.co/rwang220) |
| 类型 | 因果语言模型（`Qwen3ForCausalLM`） |
| 参数量 | 14.77B（14,768,307,200） |
| 精度 | bfloat16 |
| 结构 | 40 层，hidden size 5120，GQA 40/8，intermediate size 17408，词表 151936 |
| 上下文 | `max_position_embeddings` 为 40,960；RoPE 为 `default`（本 checkpoint 未开 YaRN） |
| 分词长度 | `model_max_length` 为 131,072；SFT 使用 8,192 |
| 语言 | 中文、英文（训练数据以中文医疗任务为主） |
| 许可 | Apache 2.0 |

`generation_config.json` 默认采样：temperature 0.6，top_p 0.95，top_k 20。

## 使用方法

需要带 Qwen3 支持的较新 `transformers`（本次导出由 transformers 5.5.0 写出）。

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "rwang220/Yida-Model-14B"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="auto",
)

messages = [{"role": "user", "content": "请用通俗语言解释高血压的常见注意事项。"}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)
inputs = tokenizer([text], return_tensors="pt").to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, temperature=0.6, top_p=0.95, top_k=20)
print(tokenizer.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

同一示例见 [`examples/inference.py`](examples/inference.py)。非思维链路径请在 `apply_chat_template` 中设 `enable_thinking=False`。

vLLM：

```bash
vllm serve rwang220/Yida-Model-14B --reasoning-parser qwen3
```

## 训练

使用 [ms-swift](https://github.com/modelscope/ms-swift) 做 LoRA，再合并为全量权重。训练代码不公开。

| 项目 | 内容 |
| --- | --- |
| 基座 | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| 框架 | ms-swift 4.0.2，PEFT 0.18.0，transformers 5.3.0.dev0，torch 2.9.0+cu128 |
| 硬件 | 4 GPU，DDP |
| 实验 | `all_14b_9.4/v0-20260904-061558` |
| 起止时间 | 2026-09-04 06:16 → 2026-09-09 19:02 |
| LoRA | rank 16，alpha 32，dropout 0.05，目标模块 `q/k/v/o/gate/up/down_proj` |
| 优化 | fused AdamW，lr `1e-4`，cosine，warmup ratio 0.05，weight decay 0.1 |
| 日程 | 4 epoch，47,384 step，max length 8192 |
| 批次 | 单卡 1，梯度累积 8（全局 batch 32） |
| 最终 train loss | 0.5922 |
| 最终 eval loss | 0.5709，token acc 0.8124 |
| 最后 / 最佳 ckpt | `checkpoint-47384` |
| 合并 | 6 个 safetensor 分片（`max_shard_size=5GB`） |

### 数据

内部医疗 SFT 混合数据（`merged_scores_v4_three_models_score5` / 前缀优化 CoT）。多数类别上限 10,000 条；`MedSafety` 与 `MedEthics` 为 30,000。验证集 10,000 条 held-out。

任务族（训练注册表中的名称）：CMB-Clin-extended, MedAnalysis, MedCare, MedChartQC, MedDiag, MedDiffer, MedEthics, MedExam, MedExplain, MedHC, MedHG, MedInsureCalc, MedInsureCheck, MedLitQA, MedMC, MedOutcome, MedPHM, MedPathQC, MedPopular, MedPrimary, MedPsychCare, MedPsychQA, MedRecordGen, MedRehab, MedReportQC, MedRxCheck, MedRxPlan, MedSafety, MedSpeQA, MedSummary, MedSynonym, MedTeach, MedTerm, MedTreat, SMDoc。

## Hub 文件

Hugging Face 上除本仓库已镜像的分词器与配置外，还包括：

- `model-00001-of-00006.safetensors` … `model-00006-of-00006.safetensors`
- Hub 模型卡（技术内容与本说明一致）

Hub 总大小约 **27.5 GiB**（约 29.5 GB）。

## 用途与限制

面向中文医疗对话、病历/报告起草及相关 NLP 原型研究。

限制：

- 可能编造指南、剂量、诊断和文献。
- 训练数据为内部数据，此处未完整公开。
- 本说明中的评估仅为训练/验证 loss，不声称公开临床基准成绩。
- 输出可能把思维过程（`<think>…</think>`）和最终答案混在一起。

## 引用

GitHub 侧栏的 “Cite this repository” 来自 [CITATION.cff](CITATION.cff)。

```bibtex
@misc{yida-model-14b,
  title        = {Yida-Model-14B},
  author       = {Wang, Rui},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/rwang220/Yida-Model-14B}},
  note         = {Code and docs: https://github.com/rwang220/yida_model}
}
```

同时请引用 [Qwen3](https://huggingface.co/Qwen/Qwen3-14B)。

## 许可

Apache License 2.0。见 [LICENSE](LICENSE) 与 [NOTICE](NOTICE)。

## 贡献与安全

见 [CONTRIBUTING.md](CONTRIBUTING.md)、[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) 与 [SECURITY.md](SECURITY.md)。
