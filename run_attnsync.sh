#!/usr/bin/env bash
# Train -> generate 50k (dense, SDE 250, cfg 1.0) -> evaluate, for one attention-sync run.
# Usage: WANDB_API_KEY=... bash run_attnsync.sh <gpus> <exp-name> <heads: mean|per-head> <port> [mask-ratio] [mask-unit] [loss: l1|js|kl] [weight]
set -euo pipefail

GPUS=$1; EXP=$2; HEADS=$3; PORT=$4; MASK=${5:-0.0}; MASK_UNIT=${6:-element}; LOSS=${7:-l1}; WEIGHT=${8:-0.1}
NPROC=$(awk -F, '{print NF}' <<< "$GPUS")
EVAL_GPU=${GPUS%%,*}
OUT=/v/mnt/GH/SiT
SAMPLES=/root/samples
REF=/root/VIRTUAL_imagenet256_labeled.npz
STEPS=400000
CKPT=$OUT/$EXP/checkpoints/$(printf %07d $STEPS).pt

cd "$(dirname "$0")"
echo "[$(date)] train $EXP on GPUs $GPUS (heads=$HEADS, mask=$MASK/$MASK_UNIT, loss=$LOSS, weight=$WEIGHT)"
if [[ ! -f $CKPT ]]; then
  CUDA_VISIBLE_DEVICES=$GPUS accelerate launch \
    --multi_gpu --num_processes "$NPROC" --main_process_port "$PORT" --mixed_precision bf16 \
    -m attnsync.train \
    --model SiT-B/2 --exp-name "$EXP" \
    --data-dir /root/imagenet_256 \
    --output-dir "$OUT" \
    --use-tread-routing \
    --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
    --use-attn-sync --attn-sync-student-blocks 4 --attn-sync-teacher-block 7 \
    --attn-sync-heads "$HEADS" --attn-sync-loss "$LOSS" --attn-sync-weight "$WEIGHT" \
    --attn-sync-mask-ratio "$MASK" --attn-sync-mask-unit "$MASK_UNIT" \
    --batch-size 256 --max-train-steps $STEPS \
    --allow-tf32
fi

echo "[$(date)] generate $EXP"
CUDA_VISIBLE_DEVICES=$GPUS torchrun \
  --standalone --nproc_per_node="$NPROC" \
  -m attnsync.generate \
  --ckpt "$CKPT" \
  --sample-dir "$SAMPLES" \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0

NPZ=$(ls -t "$SAMPLES"/SiT-B-2-"$EXP"-$(printf %07d $STEPS)-*.npz | head -1)
echo "[$(date)] evaluate $NPZ"
CUDA_VISIBLE_DEVICES=$EVAL_GPU /opt/conda/envs/tread-eval/bin/python evaluator.py "$REF" "$NPZ" \
  | tee "$OUT/$EXP/eval_$(printf %07d $STEPS).txt"
echo "[$(date)] done $EXP"
