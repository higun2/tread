# TREAD + LayerSync (selectable student/teacher blocks)

TREAD training (`attnsync.model` with its sync losses off, forward unchanged) plus a LayerSync loss
between user-chosen blocks:

```text
L = L_FM + w * mean_pairs( -mean_i cos(h_student[i], stopgrad(h_teacher[i])) )
```

Tokenwise negative cosine with a stopped deeper target, no projection head, no
extra parameters (checkpoints keep TREAD's state-dict keys; inference is the
ordinary dense SiT forward, so existing TREAD generation code works).

## Choosing blocks

Block indices are **zero-based block outputs** (SiT-B/2: 0–11). Pairs are formed
position-wise from the two lists; the teacher must be deeper than its student.

| Student / teacher location | Tokens compared |
|---|---|
| both outside the routed interval | all tokens (or `--layersync-tokens active/routed`) |
| one or both inside `[start, end)` | the active tokens that the routed block processed |

Inside the routed interval the same active subset passes every block, so
routed→routed pairs correspond exactly; a prefix/suffix block is gathered at
the active positions. `--layersync-tokens active|routed` selects TREAD's active
or bypass tokens when both blocks see all tokens (e.g. prefix→suffix).

```bash
# routed block 4 learns from routed block 8
--layersync-student-blocks 4 --layersync-teacher-blocks 8 --layersync-weight 0.1
# two pairs: 4->8 and 2->10
--layersync-student-blocks 4 2 --layersync-teacher-blocks 8 10 --layersync-weight 0.1
# prefix block 2 -> suffix block 10, bypass (routed) tokens only
--layersync-student-blocks 2 --layersync-teacher-blocks 10 --layersync-tokens routed
```

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
  --multi_gpu --num_processes 4 --mixed_precision bf16 \
  -m tread_layersync.train \
  --model SiT-B/2 --exp-name tread-layersync-4to8 \
  --data-dir /v/mnt/GH/imagenet_256 --output-dir /v/mnt/GH/SiT \
  --use-tread-routing --tread-start-block 3 --tread-end-block 9 --tread-active-ratio 0.5 \
  --layersync-student-blocks 4 --layersync-teacher-blocks 8 --layersync-weight 0.1 \
  --batch-size 256 --max-train-steps 400000 --allow-tf32
```

W&B logs `loss/fm`, `loss/layersync`, `loss/total` and `layersync/cos_{s}_{t}`
per pair. `--layersync-weight 0` (default) is exactly TREAD. Resume checks the
LayerSync settings.

## Generate

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
  -m tread_temporal_cache.generate --schedule dense \
  --ckpt /v/mnt/GH/SiT/tread-layersync-4to8/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples --num-fid-samples 50000 \
  --per-proc-batch-size 64 --mode sde --num-steps 250 --cfg-scale 1.0
```

## Tests

```bash
python -m unittest tread_layersync.tests.test_layersync -v
```

Capture forward equals TREAD bit-for-bit (same RNG), routed features carry the
active indices, pair alignment for prefix/routed/suffix combinations, weight 0
equals the TREAD loss, the teacher is stop-gradient, and invalid pairs are
rejected.
