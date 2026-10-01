# SiT-L/2 training step: TREAD vs one random dense block

Measured October 1, 2026 on one NVIDIA RTX 6000 Ada Generation, PyTorch 2.5.1+cu124. The GPU was shared with another training process at about 99% utilization. Source: [`benchmark.py`](../benchmark.py); raw numbers: [`results.json`](results.json).

| Mode | Median ms/step | Range across 3 rounds | Images/s at local batch 32 |
|---|---:|---:|---:|
| TREAD | 314.63 | 313.82–315.16 | 101.71 |
| TREAD + one random dense routed block | 322.31 | 322.25–325.72 | 99.28 |

The selected dense block raised measured step time by **2.44%**. Conditions: SiT-L/2, local batch 32, bf16 autocast, TF32 enabled, routing `[3,21)`, active ratio 0.5, default FP32 routed endpoint, no RouteSync. Each mode used the same initialization. After three warmup steps per mode, three rounds alternated order and timed eight synchronized forward + flow-matching loss + backward + gradient clipping + AdamW steps each.

Inputs and labels stayed on the GPU. The measurement excludes data loading, VAE posterior sampling, DDP communication, EMA, logging, image preview and checkpoint I/O. Shared-GPU contention limits the absolute numbers and may change the relative overhead. An isolated eight-GPU training run is needed for precise end-to-end throughput. No FID or quality result is implied.

Reproduce:

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python -m tread_random_dense.benchmark \
  --device cuda:0 --batch-size 32 --warmup 3 --steps 8 --rounds 3
```
