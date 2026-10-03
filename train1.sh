WANDB_API_KEY=${WANDB_API_KEY} \
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m tread_routesync.train \
  --model SiT-B/2 --exp-name dense_sync4 \
  --data-dir /root/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-tread-routing \
  --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --use-dense-sparse-sync --dense-sparse-sync-tokens routed \
  --dense-sparse-sync-ratio 0.1 --dense-sparse-sync-weight 0.05 \
  --batch-size 256 --max-train-steps 400000 \
  --allow-tf32

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --nproc_per_node=4 \
  -m tread_routesync.generate \
  --ckpt /v/mnt/GH/SiT/dense_sync4/checkpoints/0400000.pt \
  --sample-dir /root/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0

# conda activate tread-eval

# CUDA_VISIBLE_DEVICES=3 python evaluator.py /v/mnt/GH/VIRTUAL_imagenet256_labeled.npz \
#     /root/samples/SiT-B-2-dense_sync-0400000-size-256-vae-ema-cfg-1.0-glow-0.0-ghigh-1.0-seed-0-sde-tread-dense-ratio-0.5-recursive-0-pattern-grouped-depth-emb-0/.npz
