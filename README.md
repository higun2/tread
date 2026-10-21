# TREAD + AttnSync: aligning attention across depth in token-routed diffusion transformers

This repository trains class-conditional SiT models on ImageNet 256×256 with
[TREAD](https://arxiv.org/abs/2501.04765) token routing and **AttnSync**, an attention
alignment loss between blocks of the same network. It also contains a TREAD + LayerSync
baseline and Sparse Guidance sampling. The code base started from
[REPA](https://github.com/sihyun-yu/REPA).

## Method

AttnSync uses a deeper block as a stopped teacher for the attention maps of an earlier
(student) block:

```text
L = L_FM + λ · mean_{s ∈ students} D( A_s , stopgrad(A_teacher) )
```

- `A` are the attention probabilities `[B, heads, K, K]` of a block. For the student and
  teacher only, they are recomputed in FP32 from the block's own adaLN, qkv and q/k norm.
  The real block forward stays fused SDPA. A test checks that the recomputed maps reproduce
  the attention output.
- `D` is JS, L1 or KL(teacher ‖ student), averaged over query rows. Heads are matched one to
  one (`per-head`) or averaged first (`mean`).
- With TREAD, student and teacher must both be routed blocks. All routed blocks process the
  same active token subset, so query/key rows correspond exactly.
- No parameters are added. Inference is the ordinary dense SiT forward.

## Repository layout

| Path | Contents |
|---|---|
| `attnsync/` | TREAD + AttnSync model, loss, training (`attnsync.train`), generation with CFG / Sparse Guidance (`attnsync.generate`), tests. Options: [`attnsync/README.md`](attnsync/README.md) |
| `tread_layersync/` | TREAD + LayerSync baseline (token-wise cosine between blocks). See [`tread_layersync/README.md`](tread_layersync/README.md) |
| `models/` | SiT backbone |
| `samplers.py` | Euler (ODE) and Euler–Maruyama (SDE) samplers with CFG and guidance interval |
| `evaluator.py` | ADM evaluation suite (FID, sFID, IS, precision, recall) |
| `preprocessing/` | ImageNet → SD-VAE latent preprocessing (from REPA) |
| `run_attnsync.sh` | Train → generate 50K → evaluate launcher for one B/2 run |

`experiments/` (analysis scripts, results, notes) is kept out of version control.

## Setup

```bash
conda create -n tread python=3.11 -y
conda activate tread
pip install -r requirements.txt
```

Prepare ImageNet 256×256 latents with the [preprocessing guide](preprocessing/README.md).
`--data-dir` points to the folder that contains `images/` and `vae-sd/`.

The evaluator uses TensorFlow and the ADM reference batch `VIRTUAL_imagenet256_labeled.npz`
from [guided-diffusion](https://github.com/openai/guided-diffusion/tree/main/evaluations).
It is easiest to run in a separate environment.

## Training

Block indices are zero-based. Global batch 256, 400K steps unless noted.

**TREAD + AttnSync, SiT-B/2** (main B/2 configuration: route `[2, 9)`, student 4 → teacher 7, per-head JS, λ = 1.0)

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m attnsync.train --model SiT-B/2 --exp-name attn_sync_js_w1 \
  --data-dir [DATA_DIR] --output-dir [OUTPUT_DIR] \
  --use-tread-routing --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --use-attn-sync --attn-sync-student-blocks 4 --attn-sync-teacher-block 7 \
  --attn-sync-heads per-head --attn-sync-loss js --attn-sync-weight 1.0 \
  --batch-size 256 --max-train-steps 400000 --allow-tf32
```

**Larger models.** Same flags with the model's route and alignment pair. The runs below used
`--compile --fused-optim`, which only changes speed.

| Model | Route | Student → teacher |
|---|---|---|
| SiT-L/2 | `--tread-start-block 2 --tread-end-block 21` | `--attn-sync-student-blocks 4 --attn-sync-teacher-block 19` |
| SiT-XL/2 | `--tread-start-block 2 --tread-end-block 25` | `--attn-sync-student-blocks 4 --attn-sync-teacher-block 23` |

**Baselines.** TREAD alone: the AttnSync command without the `--use-attn-sync ...` flags.
TREAD + LayerSync:

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m tread_layersync.train --model SiT-B/2 --exp-name tread-layersync-4to7 \
  --data-dir [DATA_DIR] --output-dir [OUTPUT_DIR] \
  --use-tread-routing --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --layersync-student-blocks 4 --layersync-teacher-blocks 7 --layersync-weight 0.1 \
  --batch-size 256 --max-train-steps 400000 --allow-tf32
```

Checkpoints go to `[OUTPUT_DIR]/<exp-name>/checkpoints/`. Resume with `--resume-step`,
`--resume-last` or `--resume-ckpt`.

## Generation and evaluation

Generate 50K samples with dense inference (all tokens through all blocks):

```bash
torchrun --standalone --nproc_per_node=4 -m attnsync.generate \
  --ckpt [OUTPUT_DIR]/attn_sync_js_w1/checkpoints/0400000.pt --sample-dir [SAMPLE_DIR] \
  --tread-eval-mode dense --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

- **CFG with a guidance interval:** `--cfg-scale 1.5 --guidance-low 0 --guidance-high 0.5`.
  Here t = 1 is noise and t = 0 is data, and guidance is applied only for
  `low ≤ t ≤ high`.
- **Sparse Guidance:** `--sg-scale W --sg-strong γs --sg-weak γw` with `--cfg-scale 1.0`.
  γ is the fraction of tokens dropped in each branch. With `--sg-aux-ckpt` an earlier
  checkpoint of the same run is the weak branch (SG_FID). `--sg-strong-end` and
  `--sg-weak-end` give a cosine γ schedule over sampling.

Evaluate the `.npz` written next to the sample folder:

```bash
python evaluator.py VIRTUAL_imagenet256_labeled.npz [SAMPLE_DIR]/<sample-folder>.npz
```

## Results

ImageNet 256×256, FID-50K. Unless noted: 400K steps, batch 256, EMA, SDE 250 steps,
**no guidance (CFG 1.0)**, dense inference, training seed 0.

**SiT-B/2, TREAD route `[2, 9)`, alignment pair 4 → 7**

| Method | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|
| TREAD | 16.85 | 6.69 | 82.03 | 0.66 | **0.57** |
| TREAD + LayerSync (λ 0.1) | 15.41 | 7.14 | 89.40 | 0.68 | 0.56 |
| TREAD + AttnSync (per-head JS, λ 1.0) | **11.04** | **6.16** | **107.87** | **0.72** | 0.55 |

Across the six tested AttnSync settings on this setup (L1 mean or per-head with λ 0.1, KL
with λ 0.1, JS with λ 0.5, 1.0 or 2.0), FID was 11.04–12.82, below TREAD and TREAD + LayerSync
in every case.

**Larger models, TREAD + AttnSync**

| Model | Steps | Student → teacher | FID ↓ | sFID ↓ | IS ↑ | Precision ↑ | Recall ↑ |
|---|---|---|---|---|---|---|---|
| SiT-L/2 | 400K | 4 → 19 | 5.15 | 5.41 | 163.08 | 0.789 | 0.551 |
| SiT-XL/2 | 400K | 4 → 23 | 4.13 | 4.86 | 182.96 | 0.811 | 0.542 |
| SiT-XL/2 | 1M | 4 → 23 | 2.86 | 4.87 | 217.3 | 0.807 | 0.579 |

There are no matched TREAD baselines for L/2 and XL/2 in this repository. These are
achieved numbers, not measured improvements.

**SiT-XL/2 at 1M with guidance**

| Sampler | Guidance | FID ↓ |
|---|---|---|
| SDE 250 | CFG 1.5, interval [0, 0.5] | 2.64 |
| ODE 40 | CFG 1.5, interval [0, 0.5] | 2.68 |
| ODE 40 | SG_FID, aux checkpoint 200K, w 2.0 | 2.54 |
| ODE 40 | SG_FID, aux checkpoint 200K, w 3.0, interval [0, 0.7] | **2.51** |

Guidance gains on this model are small, and strong full-interval CFG increases FID.
All numbers come from single training and sampling runs.

## Acknowledgement

This code builds on [REPA](https://github.com/sihyun-yu/REPA), [SiT](https://github.com/willisma/SiT),
[DiT](https://github.com/facebookresearch/DiT) and [edm2](https://github.com/NVlabs/edm2), and
follows [TREAD](https://arxiv.org/abs/2501.04765) for token routing and
[Guiding Token-Sparse Diffusion Models](https://arxiv.org/abs/2601.01608) for Sparse Guidance.
