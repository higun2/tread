CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch train_b2.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-L/2" \
  --kl-coeff=0.1 \
  --encoder-depths 8 18 \
  --cfg-prob=0.1 \
  --output-dir="/media/data1/GH/SiT" \
  --exp-name="try60-L9" \
  --data-dir=/mnt/GH/imagenet_256
  # --max-train-steps=4000000 \


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=10005 generate2.py \
  --model SiT-L/2 \
  --num-fid-samples 50000 \
  --ckpt /media/data1/GH/SiT/try60-L9/checkpoints/0400000.pt \
  --sample-dir /mnt/GH/samples \
  --path-type=linear \
  --per-proc-batch-size=32 \
  --mode=ode \
  --num-steps=250 \
  --cfg-scale=1.0

