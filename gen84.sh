WANDB_API_KEY=${WANDB_API_KEY} \
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  accelerate launch train_b2.py \
  --report-to="wandb" \
  --allow-tf32 \
  --mixed-precision="fp16" \
  --seed=0 \
  --path-type="linear" \
  --prediction="v" \
  --weighting="uniform" \
  --model="SiT-L/2" \
  --kl-coeff=0.1 \
  --encoder-depths 3 5 16 \
  --cfg-prob=0.1 \
  --output-dir="/mnt/GH/SiT" \
  --exp-name="try60-L7" \
  --data-dir=/mnt/GH/imagenet_256 \
  --resume-step=0150000


torchrun --nnodes=1 --nproc_per_node=8 --master_port=10005 generate2.py \
  --model SiT-L/2 \
  --num-fid-samples 50000 \
  --ckpt /mnt/GH/SiT/try60-L7/checkpoints/0400000.pt \
  --sample-dir /mnt/GH/samples \
  --path-type=linear \
  --per-proc-batch-size=32 \
  --mode=ode \
  --num-steps=250 \
  --cfg-scale=1.0



# WANDB_API_KEY=${WANDB_API_KEY} \
#   CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   accelerate launch train_b2.py \
#   --report-to="wandb" \
#   --allow-tf32 \
#   --mixed-precision="fp16" \
#   --seed=0 \
#   --path-type="linear" \
#   --prediction="v" \
#   --weighting="uniform" \
#   --model="SiT-B/2" \
#   --kl-coeff=0.1 \
#   --encoder-depths 3 4 7 \
#   --cfg-prob=0.1 \
#   --output-dir="/mnt/GH/SiT" \
#   --exp-name="try66" \
#   --data-dir=/mnt/GH/imagenet_256

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
#   torchrun --nnodes=1 --nproc_per_node=4 --master_port=10005 generate2.py \
#   --model SiT-B/2 \
#   --num-fid-samples 50000 \
#   --ckpt /mnt/GH/SiT/try66/checkpoints/0400000.pt \
#   --sample-dir /mnt/GH/samples \
#   --path-type=linear \
#   --per-proc-batch-size=32 \
#   --mode=sde \
#   --num-steps=250 \
#   --cfg-scale=1.0
