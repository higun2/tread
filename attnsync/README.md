# TREAD + Routed Attention Sync

Later routed blocks act as a stopped teacher for the attention maps of earlier routed
blocks:

```text
L = L_FM + lambda * mean_{s in students} D(A_s, sg(A_teacher))
```

The routed interval is `[tread_start_block, tread_end_block)`. Inside it, every block
processes the same active token subset, so query/key rows correspond exactly between
blocks. That is why student and teacher blocks must both be routed blocks. Block indices
are **zero-based**, like `--tread-start-block`.

The package is a copy of `densepush`. DensePush and RouteSync are still available and off
by default. Attention sync adds no parameters and runs only in sparse training forwards,
so evaluation and `attnsync.generate` are unchanged.

## How the maps are computed

The real block forward stays fused SDPA. For student and teacher blocks only, the
attention log-probabilities `[B,H,K,K]` are recomputed in FP32 from the block's own
adaLN, qkv and q/k norm. The TREAD diagonal attention correction is included when it is
enabled. A test checks that these maps reproduce the real attention output. The teacher is
computed under `no_grad` with the autocast cache off (a CUDA regression test covers this).
Gradients from the sync loss reach only the student blocks and the blocks before them.

## Options

| flag | default | meaning |
|---|---|---|
| `--use-attn-sync` | off | enable |
| `--attn-sync-student-blocks` | (required) | e.g. `3 5` |
| `--attn-sync-teacher-block` | (required) | e.g. `7` |
| `--attn-sync-heads` | mean | `mean`: average head probabilities first (head-permutation invariant); `per-head`: match head h to head h |
| `--attn-sync-loss` | js | `l1` (sum per row, in [0,2]), `js` (nats, in [0, log 2]), `kl` = KL(teacher ‖ student) |
| `--attn-sync-weight` | 0.1 | lambda, applied to the mean over student blocks |
| `--attn-sync-ratio` | 1.0 | fraction of the local batch whose maps are computed |

All divergences are averaged over query rows, samples (and heads for `per-head`). Logged
values: `loss/attn_sync` and `attn_sync/<loss>_block<i>` for each student block.

At initialization with SiT-B/2, `[2,9)`, students `3 5` and teacher `7`, the loss values
are about: mean-JS 0.0026, mean-L1 0.12, per-head-KL 0.096. The loss scales differ a lot
between options, so tune lambda for each loss type.

## Usage

```bash
accelerate launch --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m attnsync.train --model SiT-B/2 --exp-name attn_sync \
  --data-dir /root/imagenet_256 --output-dir /v/mnt/GH/SiT \
  --use-tread-routing --tread-start-block 2 --tread-end-block 9 --tread-active-ratio 0.5 \
  --use-attn-sync --attn-sync-student-blocks 3 5 --attn-sync-teacher-block 7 \
  --attn-sync-heads mean --attn-sync-loss js --attn-sync-weight 0.1 \
  --batch-size 256 --max-train-steps 400000 --allow-tf32
```

Tests: `python -m unittest discover -s attnsync/tests -t .`
