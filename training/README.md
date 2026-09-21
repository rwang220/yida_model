# Training snapshot

[English README](../README.md) · [中文说明](../README.zh-CN.md)

This directory is the training-side companion to [rwang220/Yida-Model-14B](https://huggingface.co/rwang220/Yida-Model-14B). The Hub listing is the **merged full-weight** checkpoint. These files record how that checkpoint was trained.

本目录对应 Hub 上的合并全量权重；这里保存的是训练脚本和最终 LoRA 配置。

| File | Source on GPU 220 |
| --- | --- |
| `sft.sh` | `/train/hwj/apps/Qwen3/sft.sh` |
| `args.json` | `all_14b_9.4/v0-20260904-061558/checkpoint-47384/args.json` |
| `adapter_config.json` | same checkpoint (LoRA before merge) |

Run used 4 GPUs (`4,5,6,7`), ms-swift 4.0.2, LoRA rank 16 / alpha 32, 4 epochs, 47,384 steps, then merge into `yida_modelv2`.

Weights (~27.5 GiB) stay on Hugging Face and are gitignored here:

```bash
huggingface-cli download rwang220/Yida-Model-14B --local-dir ./weights
```
