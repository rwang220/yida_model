# Contributing

## English

This repo mirrors documentation, tokenizer files, and configs for [Yida-Model-14B](https://huggingface.co/rwang220/Yida-Model-14B). Weight shards stay on Hugging Face. Training code is not published.

Useful contributions:

- Fixes to the GitHub or Hub model cards (keep [README.md](README.md) and [README.zh-CN.md](README.zh-CN.md) in sync)
- Clearer inference examples

Please do **not** open PRs that add:

- training scripts, LoRA adapters, or internal run configs
- `*.safetensors` / `*.bin` / `*.gguf`
- tokens, `.env`, API keys
- internal patient data or full training corpora

Workflow: fork, branch, pull request. Use the PR template checklist.

By contributing you agree the contribution is licensed under Apache-2.0, same as this repository.

## 中文

本仓库保存 [Yida-Model-14B](https://huggingface.co/rwang220/Yida-Model-14B) 的说明、分词器和配置。权重分片只在 Hugging Face。训练代码不公开。

欢迎：

- 修正 GitHub / Hub 模型卡（请同步 [README.md](README.md) 与 [README.zh-CN.md](README.zh-CN.md)）
- 更清楚的推理示例

请不要提交：

- 训练脚本、LoRA adapter 或内部运行配置
- `*.safetensors` / `*.bin` / `*.gguf`
- token、`.env`、API key
- 内部患者数据或完整训练集

流程：fork → 分支 → Pull Request，并按 PR 模板自检。

贡献内容按本仓库相同的 Apache-2.0 许可授权。
