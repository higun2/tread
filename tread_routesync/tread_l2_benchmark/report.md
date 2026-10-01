# SiT-L/2: dense vs TREAD training-step speed

Measured September 29, 2026 on one NVIDIA RTX 6000 Ada Generation with PyTorch 2.5.1+cu124. The GPU was shared with another active training process during the measurement. Benchmark source: [`benchmark_tread_l2.py`](../benchmark_tread_l2.py); raw measurements: [`results.json`](results.json).

| Mode | Median step time | Range across 3 rounds | Images/s, local batch 32 |
|---|---:|---:|---:|
| Dense (TREAD off) | 466.51 ms | 466.06–474.49 ms | 68.59 |
| TREAD | 331.68 ms | 327.31–334.40 ms | 96.48 |

TREAD is **1.41× faster** for the measured training step, a **28.9% reduction in step time**. Three rounds alternated order: dense → TREAD, TREAD → dense, dense → TREAD. Each round timed eight steps after three warmup steps per mode, with CUDA synchronization around each timed group.

Both modes used SiT-L/2 at latent input 32×32, local batch 32, bf16 autocast, TF32 enabled, the same initialization, flow matching, gradient clipping and AdamW. TREAD routed blocks `[2,21)` with active ratio 0.5 and its default FP32 endpoint. RouteSync, dense–sparse alignment and attention correction were off in both modes.

Inputs and labels stayed on the GPU. This measures forward, loss, backward, clipping and optimizer step. It excludes loading latents, posterior sampling, DDP synchronization, EMA update, logging and checkpoint I/O. Because another process occupied each GPU at nearly 100% utilization, these are **shared-GPU measurements**; do not directly multiply the images/s by eight to predict a clean eight-GPU training run. The relative comparison is more useful than the absolute times, but contention can still affect it.

Reproduce with:

```bash
CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 python -m tread_routesync.benchmark_tread_l2 \
  --device cuda:0 --batch-size 32 --warmup 3 --steps 8 --rounds 3
```
