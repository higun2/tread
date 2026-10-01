"""Generation entry point for row-H-div checkpoints.

Row/CKA H-div and attention capture are training-only, so inference is exactly
the ordinary SiT path used by ``attention_alignment_hdiv.generate``.
"""

from attention_alignment_hdiv.generate import (
    create_npz_from_sample_folder,
    main,
    parse_args,
)

__all__ = ["create_npz_from_sample_folder", "main", "parse_args"]


if __name__ == "__main__":
    main(parse_args())



'''
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master_port=10004 \
  -m attention_alignment_row_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/n1/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0


CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nnodes=1 --nproc_per_node=4 --master_port=10005 \
  -m attention_alignment_row_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/n2/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0

torchrun --nnodes=1 --nproc_per_node=8 --master_port=10005 \
  -m attention_alignment_row_hdiv.generate \
  --ckpt /v/mnt/GH/SiT/sit_b2_attn_row_l1/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --num-fid-samples 50000 \
  --per-proc-batch-size 32 \
  --mode sde \
  --num-steps 250 \
  --cfg-scale 1.0

'''
