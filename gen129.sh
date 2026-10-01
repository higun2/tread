accelerate launch \
  --num_processes 8 \
  -m attention_alignment_hdiv.train \
  --exp-name v5 \
  --model SiT-B/2 \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --batch-size 256 \
  --mixed-precision bf16 \
  --allow-tf32 \
  --max-train-steps 400000 \
  --encoder-depths 3 5 7 \
  --attn-loss-type kl \
  --kl-coeff 0.1 \
  --enable-hdiv \
  --hdiv-weight 0.1 \
  --hdiv-cka-threshold 0.1 \
  --hdiv-detach-deep


  torchrun --nnodes=1 --nproc_per_node=8 --master_port=10005 \
  -m attention_alignment_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/v5/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0
