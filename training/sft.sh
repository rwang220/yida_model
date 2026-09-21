#!/bin/bash

set -euo pipefail

# Qwen3 LoRA 训练脚本（支持单卡/多卡）
# 用法：
#   bash sft.sh             # 默认使用 0 号卡（单卡）
#   bash sft.sh 1           # 使用 1 号卡（单卡）
#   bash sft.sh 0,1,2,3     # 使用 0,1,2,3 四卡
#   MODEL_PATH=/data/models/Qwen3.5-27B bash sft.sh 0,1

# 建议使用已验证支持 Qwen3.5/Qwen3 的 swift conda 环境
CONDA_ENV=${CONDA_ENV:-mswift-qwen3}

if ! command -v conda >/dev/null 2>&1; then
    echo "错误: 未找到 conda 命令，请先安装或初始化 conda。"
    exit 1
fi

if ! conda env list | awk '{print $1}' | grep -Fxq "$CONDA_ENV"; then
    echo "错误: conda 环境不存在: $CONDA_ENV"
    echo "可用环境如下:"
    conda env list
    exit 1
fi

run_python() {
    conda run --no-capture-output -n "$CONDA_ENV" python "$@"
}

if ! run_python -c "import swift" >/dev/null 2>&1; then
    echo "错误: 当前 conda 环境缺少 swift 包: $CONDA_ENV"
    echo "请切换到包含 swift 的环境，例如: CONDA_ENV=mswift-qwen3 bash sft.sh 0"
    exit 1
fi

echo "=== 使用 conda 环境: $CONDA_ENV ==="

echo "=== Python 关键包版本 ==="
run_python -c "
import importlib
for n in ['swift', 'transformers', 'peft', 'torch']:
    m = importlib.import_module(n)
    print(f'{n}={getattr(m, \"__version__\", \"n/a\")}')
" || {
    echo "错误: 无法读取 swift/transformers/peft/torch 版本，请检查 Python 环境完整性。"
    exit 1
}
echo "========================"

# 1) 固定数据集配置（与当前流程一致）
DATA_INFO_PATH=/train/yrh_1/dpt/merged_scores_v4_three_models_score5/merged_by_prefix_opt_cot/data_info.json
VAL_DATASET=${VAL_DATASET:-/train/yrh_1/dpt/merged_scores_v4_three_models_score5/merged_by_prefix_opt_cot/val_10000.jsonl}

DATASET_ARGS=$(run_python -c "
import json
from pathlib import Path

LIMIT = 10000
# 这两个类别数据量大，单独限制为 30000
SPECIAL_LIMITS = {'MedSafety': 30000, 'MedEthics': 30000}

def count_samples(dataset_path: str) -> int:
    p = Path(dataset_path)
    suffix = p.suffix.lower()

    # jsonl: one non-empty line is one sample
    if suffix == '.jsonl':
        with p.open('r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())

    # json: prefer top-level list, then common list fields
    if suffix == '.json':
        with p.open('r', encoding='utf-8') as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return len(obj)
        if isinstance(obj, dict):
            for key in ('data', 'train', 'samples', 'items', 'conversations', 'messages'):
                v = obj.get(key)
                if isinstance(v, list):
                    return len(v)

    # fallback: treat as line-based file
    with p.open('r', encoding='utf-8') as f:
        return sum(1 for line in f if line.strip())

data = json.load(open('$DATA_INFO_PATH', 'r', encoding='utf-8'))
dataset_args = []
for item in data:
    dataset_name = item['dataset_name']
    sample_count = count_samples(item['dataset_path'])
    limit = SPECIAL_LIMITS.get(dataset_name, LIMIT)
    use_count = min(limit, sample_count)
    dataset_args.append(f\"{dataset_name}#{use_count}\")
    # dataset_args.append(f\"{dataset_name}\")

print(' '.join(dataset_args))
")

echo "=== 正在使用的数据集采样参数 ==="
echo $DATASET_ARGS
echo "==============================="

MODEL_TYPE=${MODEL_TYPE:-qwen3}
MODEL_PATH=${MODEL_PATH:-/train/models/Qwen3-14B}

GPU_LIST=${1:-0}
if ! [[ "$GPU_LIST" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "错误: GPU 参数格式非法。"
    echo "示例: bash sft.sh 0        (单卡)"
    echo "示例: bash sft.sh 0,1,2,3  (多卡)"
    exit 1
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_LIST"
NUM_GPUS=${#GPU_ARRAY[@]}

NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29502}

if ! [[ "$NNODES" =~ ^[0-9]+$ && "$NODE_RANK" =~ ^[0-9]+$ && "$MASTER_PORT" =~ ^[0-9]+$ ]]; then
    echo "错误: NNODES/NODE_RANK/MASTER_PORT 必须是整数。"
    exit 1
fi

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "=== 本次使用 GPU: $GPU_LIST (单卡) ==="
else
    echo "=== 本次使用 GPU: $GPU_LIST (多卡, nproc_per_node=$NUM_GPUS) ==="
    echo "=== DDP 参数: nnodes=$NNODES, node_rank=$NODE_RANK, master_addr=$MASTER_ADDR, master_port=$MASTER_PORT ==="
fi

# 2) 单卡推荐参数（保守策略，降低微调后与原模型的差距）
# LR=1e-5:       低学习率，避免过拟合/遗忘（原 5e-5 过高）
# NUM_EPOCHS=1:  单轮训练，减少对原始分布的破坏y，原3
# LORA_ALPHA=16: alpha==rank，LoRA缩放系数=1.0（原2.0），抑制更新幅度
# WEIGHT_DECAY:  L2 正则，防止参数偏移过大
# MAX_GRAD_NORM: 梯度裁剪，防止单步更新过大
MAX_LENGTH=${MAX_LENGTH:-8192}
PER_DEVICE_BS=${PER_DEVICE_BS:-1}
GRAD_ACC=${GRAD_ACC:-8}
LORA_RANK=${LORA_RANK:-16}  
LORA_ALPHA=${LORA_ALPHA:-32} #原32
LR=${LR:-1e-4} # 1e-4
NUM_EPOCHS=${NUM_EPOCHS:-4}  #原3
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}  #原0.1
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}   #原1.0
DATASET_NUM_PROC=${DATASET_NUM_PROC:-8}
DATALOADER_NUM_WORKERS=${DATALOADER_NUM_WORKERS:-4}
EVAL_STEPS=${EVAL_STEPS:-500}
PER_DEVICE_EVAL_BS=${PER_DEVICE_EVAL_BS:-1}
LOAD_FROM_CACHE_FILE=${LOAD_FROM_CACHE_FILE:-false}
TARGET_MODULES=${TARGET_MODULES:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}

# 3) 启动训练
export CUDA_VISIBLE_DEVICES=$GPU_LIST
export VLLM_USE_V1=0
export PYTORCH_ALLOC_CONF=expandable_segments:True

read -r -a DATASET_ARR <<< "$DATASET_ARGS"
read -r -a TARGET_MODULES_ARR <<< "$TARGET_MODULES"

SFT_ARGS=(
    --model_type "$MODEL_TYPE"
    --model "$MODEL_PATH"
    --custom_dataset_info "$DATA_INFO_PATH"
    --dataset "${DATASET_ARR[@]}"
    --val_dataset "$VAL_DATASET"
    --output_dir all_14b_9.4
    --lora_rank "$LORA_RANK"
    --lora_alpha "$LORA_ALPHA"
    --torch_dtype bfloat16
    --target_modules "${TARGET_MODULES_ARR[@]}"
    --num_train_epochs "$NUM_EPOCHS"
    --max_length "$MAX_LENGTH"
    --per_device_train_batch_size "$PER_DEVICE_BS"
    --gradient_accumulation_steps "$GRAD_ACC"
    --learning_rate "$LR"
    --weight_decay "$WEIGHT_DECAY"
    --max_grad_norm "$MAX_GRAD_NORM"
    --dataset_num_proc "$DATASET_NUM_PROC"
    --dataloader_num_workers "$DATALOADER_NUM_WORKERS"
    --load_from_cache_file "$LOAD_FROM_CACHE_FILE"
    --gradient_checkpointing true
    --logging_steps 50
    --eval_steps "$EVAL_STEPS"
    --per_device_eval_batch_size "$PER_DEVICE_EVAL_BS"
    --save_steps 1000
    --save_only_model true
    --lr_scheduler_type cosine
    --warmup_ratio 0.05
)

echo "=== 模型配置: model_type=$MODEL_TYPE, model=$MODEL_PATH ==="
echo "=== Data 并行: dataset_num_proc=$DATASET_NUM_PROC, dataloader_num_workers=$DATALOADER_NUM_WORKERS, cache=$LOAD_FROM_CACHE_FILE ==="
echo "=== 训练参数: lr=$LR, epochs=$NUM_EPOCHS, lora_rank=$LORA_RANK, lora_alpha=$LORA_ALPHA, weight_decay=$WEIGHT_DECAY, max_grad_norm=$MAX_GRAD_NORM ==="
echo "=== OOM 安全参数: max_length=$MAX_LENGTH, grad_acc=$GRAD_ACC, target_modules=$TARGET_MODULES ==="

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "=== 启动模式: 单卡 swift.cli.sft ==="
    run_python -m swift.cli.sft "${SFT_ARGS[@]}"
else
    echo "=== 启动模式: 多卡 torch.distributed.run + swift.cli.sft ==="
    run_python -m torch.distributed.run \
        --nproc_per_node "$NUM_GPUS" \
        --nnodes "$NNODES" \
        --node_rank "$NODE_RANK" \
        --master_addr "$MASTER_ADDR" \
        --master_port "$MASTER_PORT" \
        -m swift.cli.sft "${SFT_ARGS[@]}"
fi