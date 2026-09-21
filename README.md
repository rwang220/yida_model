# Yida-Model-14B

**English** | [中文](README.zh-CN.md)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97_Hugging_Face-Yida--Model--14B-ffd21e)](https://huggingface.co/rwang220/Yida-Model-14B)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![GitHub release](https://img.shields.io/badge/repo-public-success)](https://github.com/rwang220/yida_model)

Yida-Model-14B is a merged full-weight checkpoint of [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) after LoRA supervised fine-tuning on a governed Chinese medical knowledge corpus. Hub files are merged `bfloat16` weights (**not** a PEFT adapter).

This model is for research and engineering evaluation. It is **not** a medical device and must not be used as the sole basis for clinical decisions.

| Resource | Link |
| --- | --- |
| Weights and Hub model card | [huggingface.co/rwang220/Yida-Model-14B](https://huggingface.co/rwang220/Yida-Model-14B) |
| This repository (docs, configs, tokenizer) | [github.com/rwang220/yida_model](https://github.com/rwang220/yida_model) |
| English website | [rwang220.github.io/yida_model](https://rwang220.github.io/yida_model/) |
| Base model | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |

## What this repository contains

Git does not store the ~27.5 GiB weight shards. Those live on Hugging Face.

| Path | Description |
| --- | --- |
| `README.md` / `README.zh-CN.md` | GitHub documentation (this file and the Chinese version) |
| `LICENSE` | Apache-2.0 (inherited from Qwen3-14B) |
| `config.json` / `generation_config.json` | Model architecture and default sampling |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` | Tokenizer |
| `chat_template.jinja` | Qwen3 thinking / tool-calling chat template |
| `model.safetensors.index.json` | Index of the six Hub weight shards |
| `examples/inference.py` | Load from Hub and generate |
| `docs/index.html` | English official website (GitHub Pages source) |

Download weights:

```bash
huggingface-cli download rwang220/Yida-Model-14B --local-dir ./weights
```

## Model details

| Item | Value |
| --- | --- |
| Developed by | [rwang220](https://huggingface.co/rwang220) |
| Model type | Causal LM (`Qwen3ForCausalLM`) |
| Parameters | 14.77B (14,768,307,200) |
| Precision | bfloat16 |
| Architecture | 40 layers, hidden size 5120, GQA 40/8, intermediate size 17408, vocab 151936 |
| Context | `max_position_embeddings` 40,960; RoPE `default` (no YaRN in this checkpoint) |
| Tokenizer length | `model_max_length` 131,072; SFT used 8,192 |
| Languages | Chinese and English (training mix is primarily Chinese medical tasks) |
| License | Apache 2.0 |

Default sampling in `generation_config.json`: temperature 0.6, top_p 0.95, top_k 20.

## How to use

Requires a recent `transformers` with Qwen3 support (this export was written by transformers 5.5.0).

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

The same example is in [`examples/inference.py`](examples/inference.py). Set `enable_thinking=False` in `apply_chat_template` for the non-thinking path.

vLLM:

```bash
vllm serve rwang220/Yida-Model-14B --reasoning-parser qwen3
```

## Training

Fine-tuned with [ms-swift](https://github.com/modelscope/ms-swift) LoRA, then merged into full weights. Training code is not published.

| Item | Value |
| --- | --- |
| Base model | [Qwen/Qwen3-14B](https://huggingface.co/Qwen/Qwen3-14B) |
| Framework | ms-swift 4.0.2, PEFT 0.18.0, transformers 5.3.0.dev0, torch 2.9.0+cu128 |
| Hardware | 4 GPUs, DDP |
| Run | `all_14b_9.4/v0-20260904-061558` |
| Start / end | 2026-09-04 06:16 → 2026-09-09 19:02 |
| LoRA | rank 16, alpha 32, dropout 0.05, targets `q/k/v/o/gate/up/down_proj` |
| Optim | fused AdamW, lr `1e-4`, cosine, warmup ratio 0.05, weight decay 0.1 |
| Schedule | 4 epochs, 47,384 steps, max length 8192 |
| Batch | per-device 1, grad accum 8 (global batch 32) |
| Final train loss | 0.5922 |
| Final eval loss | 0.5709, token acc 0.8124 |
| Last / best ckpt | `checkpoint-47384` |
| Merge | 6 safetensor shards (`max_shard_size=5GB`) |

### Data

A structured corpus built from more than a thousand medical textbooks, clinical guidelines, and reference works, spanning internal medicine, surgery, obstetrics and gynecology, pediatrics, pharmacology, diagnostics, and evidence-based medicine. Fine-tuning data is produced through a governance loop: collaborative generation, three-axis scoring (medical accuracy, expression quality, safety ethics), category balancing, and semantic deduplication.

## Evaluation

The 14B experimental model is tied for 10th on the MedBench public leaderboard for comprehensive Chinese medical evaluation (medical text generation, knowledge QA, clinical reasoning, text comprehension, and safety & compliance). That ranking is a research-stage snapshot recorded in the paper and does not indicate clinical validity.

This checkpoint's training metrics: final train loss 0.5922; eval loss 0.5709, token acc 0.8124.

## Hub files

On Hugging Face, in addition to the tokenizer and configs mirrored here:

- `model-00001-of-00006.safetensors` … `model-00006-of-00006.safetensors`
- Hub model card (same technical content as this README)

Total Hub payload is about **27.5 GiB** (~29.5 GB).

## Intended use and limitations

Intended for research on Chinese medical dialogue, chart/report drafting, and related NLP prototypes.

Limitations:

- Can hallucinate guidelines, doses, diagnoses, and citations.
- Training data is not fully documented here.
- MedBench ranking is a research snapshot and does not indicate clinical validity.
- Outputs may mix thinking traces (`<think>…</think>`) with the final answer.

## Citation

GitHub also exposes this via [CITATION.cff](CITATION.cff) (“Cite this repository”).

```bibtex
@misc{yida-model-14b,
  title        = {Yida-Model-14B},
  author       = {Wang, Rui},
  year         = {2026},
  howpublished = {\url{https://huggingface.co/rwang220/Yida-Model-14B}},
  note         = {Code and docs: https://github.com/rwang220/yida_model}
}
```

Also cite [Qwen3](https://huggingface.co/Qwen/Qwen3-14B).

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md), [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), and [SECURITY.md](SECURITY.md).
