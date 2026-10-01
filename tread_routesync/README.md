# TREAD + RouteSync R-P Relational Loss

This package keeps the fixed-subset TREAD execution from `tread_token_routing` and adds one optional training loss. The original package is unchanged. There is no learned router, projector, teacher model, extra Transformer block, or extra model forward.

## Feature locations and index meaning

TREAD uses a half-open routed interval `[tread_start_block, tread_end_block)`. For SiT-B/2 with `[3,9)`:

```text
blocks 0-2:   dense prefix
              h_pre = output of block 2, immediately before token splitting
blocks 3-8:   P (active_indices) is processed; R (bypass_indices) is unchanged
              restore R and P to their original spatial positions
block 9:      first existing dense suffix block
              h_post = output of block 9
blocks 10-11: remaining dense suffix
```

The existing code calls the processed group `active_indices` and the skipped/routed group `bypass_indices`. RouteSync maps them as follows:

```text
processed_idx = active_indices   # P
routed_idx    = bypass_indices   # R
```

The same original spatial indices gather both `h_pre` and `h_post`. RouteSync therefore requires at least one dense suffix block (`tread_end_block < model depth`) and non-empty R and P groups (`0 < tread_active_ratio < 1`). Recursive `AABBCC` and `ABCABC` execution use the same semantic feature locations; `h_post` defaults to the output of the first suffix block; `--routesync-target-blocks` selects later suffix outputs as well.

## Loss

For each state, R and P features are gathered and normalized once along the hidden dimension. The implementation computes only the cross-group cosine matrix. By default, the post relation is stopped:

```text
relation_pre  = normalize(h_pre_R)  @ normalize(h_pre_P).T
relation_post = normalize(h_post_R) @ normalize(h_post_P).T

L_routesync = mean(abs(relation_pre - stopgrad(relation_post)))
L_total     = L_original + routesync_weight * L_routesync
```

RouteSync follows the current AMP/autocast precision, so BF16 training computes its relation matrices under BF16 autocast as well. It does not cast features to FP32 or alter model blocks, optimizer state, TREAD's FP32 endpoint, routing selection, or the flow-matching objective.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m tread_routesync.train \
  --model SiT-B/2 --exp-name tread-routesync-b \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-tread-routing \
  --tread-start-block 3 --tread-end-block 9 \
  --tread-active-ratio 0.5 \
  --tread-recursive --tread-num-groups 3 \
  --tread-recursive-pattern grouped --tread-depth-embedding \
  --use-routesync --routesync-weight 0.1 \
  --batch-size 256 --max-train-steps 400000 \
  --allow-tf32
```

Add `--tread-recursive --tread-num-groups 3 --tread-recursive-pattern grouped` for `A1 A1 B1 B1 C1 C1`, or select `interleaved` for `A1 B1 C1 A1 B1 C1`. Add `--tread-depth-embedding` to learn one embedding per logical routed layer and add it to the timestep/class AdaLN condition before every shared-block call. Its default is off; `--no-tread-depth-embedding` disables it explicitly.

Defaults preserve the baseline: `use_routesync=False`, `routesync_weight=0.0`, and `routesync_debug=False`. `--routesync-debug` validates and prints the shapes and partition once. W&B logs `loss/fm`, `loss/route_sync`, `loss/route_sync_weighted`, and `loss/total`. Training preview grids remain dense.

## Generate

RouteSync is inactive whenever the model is in evaluation mode. No pre/post feature references or relation matrices are created, so generation has no RouteSync cost. Dense and sparse generation retain the existing TREAD behavior:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m tread_routesync.generate \
  --ckpt /v/mnt/GH/SiT/tread-routesync-b/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

Use `--tread-eval-mode sparse --tread-seed 1234` to generate with the trained sparse route and a deterministic subset.

## Tests

```bash
python -m unittest tread_routesync.tests.test_routesync -v
```

The tests cover exact disabled equivalence with the original TREAD package, feature boundary and original-index reuse, R/P partition checks, stopped post gradients, RouteSync gradients into `h_pre`, total-loss backward, recursive execution, BF16 autocast, unchanged parameter/optimizer dtypes, inference inactivity, and two-process DDP.

## Random subset alignment

Add `--routesync-sample-ratio 0.5` to the existing training command:

```bash
--use-routesync --routesync-weight 0.1 --routesync-sample-ratio 0.5
```

The ratio is the fraction **kept separately in each R and P group**, not a mask
ratio or a change to TREAD's active ratio. Default `1.0` exactly preserves full
R-P alignment and consumes no additional random numbers. Valid values are `(0, 1]`.
Each image independently selects `max(1, floor(group_size * ratio))` tokens per
group without replacement on every loss call. The same selected original spatial
indices are reused at pre/post depths. Tokens are gathered before relation
matmuls: with R=P=128, ratio 0.5 computes a 64x64 matrix instead of 128x128.
L1 is averaged over selected pairs, so no extra ratio multiplier is applied.

The post target is always stop-gradient. Backbone routing, the full-token FM loss,
and inference are unchanged. No additional model forward is needed. Sampling uses
PyTorch's device RNG, which is included in the existing training resume state.
The ratio is saved in arguments/checkpoints; resume requires the same ratio.
Older checkpoints without it default to 1.0. Use different experiment names for
ratio ablations. Older post-gradient experiment metadata is ignored: this code
always uses stopped post targets, even when loading those checkpoints.

`--routesync-debug` reports the actual sampled matrix shape on the first loss call.

```bash
python -m unittest tread_routesync.tests.test_subset -v
```

## Random training active ratio

To uniformly choose from `{0.3, 0.5, 0.7}` on every training forward, add:

```bash
--tread-active-ratios 0.3 0.5 0.7
```

This optional list overrides the fixed `--tread-active-ratio` **during training
only**. Without it, the original fixed-ratio path and RNG behavior are preserved.
Rank 0 samples one ratio and broadcasts it to all ranks. All images use that
ratio, with independently sampled token positions per image; the same subset is
used throughout the routed interval. Counts use `round(N * ratio)`, so N=256
produces 77, 128, or 179 active tokens. Choices must be distinct and in `(0, 1)`.
This is one backbone forward, not multiple ratios applied to the same input.
With gradient accumulation, each microbatch forward draws its own ratio; logs
`routing/active_ratio` and `routing/active_fraction` show the last microbatch.

Dense inference is unchanged. Optional sparse evaluation uses the fixed
`--tread-active-ratio` (default 0.5), without sampling a ratio. RouteSync subset
sampling applies independently within the resulting R/P groups. Post targets
remain stopped. The ratio list is saved in args/checkpoints and must match on
resume; old checkpoints without it default to fixed-ratio training. Use a new
experiment name for this ablation. RNG states are saved by the existing trainer.

```bash
--use-tread-routing --tread-active-ratios 0.3 0.5 0.7 \
--use-routesync --routesync-weight 0.1 --routesync-sample-ratio 0.5
```


## Sparse attention diagonal correction

Optional; correction is disabled by default. When enabled, its backend defaults to `sdpa`; select `flex` explicitly if needed. Add to the existing training command:

```bash
--tread-attn-correction \
  --tread-attn-correction-strength 1.0 \
  --tread-attn-correction-backend sdpa
```

Only sparse routed attention receives `strength * log((K-1)/(N-1))` on its
self logits. Dense inference uses the original attention path; evaluate with
`--tread-eval-mode dense`. Sparse evaluation retains the correction.
Strength zero takes the original exact path. No new checkpoint parameters.

`flex` uses compiled FlexAttention (CUDA; PyTorch 2.5.1 tested), including the
FP32 endpoint. `sdpa` uses an additive bias and needs no compilation. The backend
selection controls corrected blocks even when `--fused-attn` is disabled.
Flex uses static-shape compilation on PyTorch 2.5; new token/batch shapes may
recompile. For many changing shapes, prefer `sdpa`.

Correction options can intentionally change when resuming an old checkpoint;
a warning records the change. Existing non-correction resume checks still apply.


## RouteSync target depth

Block indices are **zero-based**, matching `--tread-start-block` and
`--tread-end-block`. The target is the output **after** the chosen suffix block.
For SiT-B/2 with routing `[3,9)`, valid targets are 9, 10 and 11 (the 10th–12th
blocks in one-based counting).

```bash
--use-routesync --routesync-target-blocks 10       # fixed block 10
--use-routesync --routesync-target-blocks 9 10 11 # uniform random choice
```

Omitting the option preserves the original target: block `tread_end_block`.
A singleton consumes no extra RNG. Multiple candidates select one target per
training forward, shared by the local batch and synchronized across DDP ranks.
Only that target is used, not an average of candidate losses. Its relation remains
stop-gradient. This reuses the existing forward; no extra backbone pass or layers.
Inference does not sample targets or compute RouteSync. Selection uses checkpointed
PyTorch RNG and is logged as `routesync/target_block` (last microbatch when using
gradient accumulation). Invalid, duplicate and pre-reintegration indices are rejected.
Resume validates the candidate list; old checkpoints default to `tread_end_block`,
and an explicit singleton containing that block is equivalent.


## Direct feature cosine alignment (LayerSync-style comparison)

```bash
# Existing R-P relation L1 (default)
--use-routesync --routesync-loss-type relational --routesync-weight 0.1
# Direct token feature cosine alignment instead of the R-P relation matrix
--use-routesync --routesync-loss-type feature-cosine --routesync-weight 0.1
```

The feature loss is `-mean_i(cosine(h_pre[i], stopgrad(h_post[i])))`.
It aligns corresponding original spatial positions across both R and P tokens;
there are no cross-token pairs, projection heads, or added backbone forwards.
Cosine aligns feature direction, not feature magnitude. Low-precision feature
cosines are reduced in FP32. The default relational computation stays unchanged.
The total objective is `FM + routesync_weight * selected_alignment_loss`.
Only ONE alignment type is used, not both summed together.

This follows the tokenwise stopped-target idea in [LayerSync Eq. 2](https://openreview.net/pdf?id=4itprlvbRQ).
This implementation uses negative mean cosine, matching the paper's sign and offset. The loss can be negative; this is expected. It is a TREAD comparison variant, not a
reproduction of the paper's complete training recipe or chosen layer pairs.

Student: full tokens immediately before TREAD split. Teacher: existing
`--routesync-target-blocks` selection (default `tread_end_block`, zero-based).
Thus direct auxiliary gradients still go to the prefix, not the teacher branch.
`--routesync-sample-ratio 1` uses all tokens; smaller values independently sample
that fraction of each R/P group (floor, minimum one), with identical pre/post
indices. Sampled tokens have equal weight; R/P means are only diagnostics.

For a controlled comparison, use separate experiment names and keep data,
seed, target blocks, sampling ratio, attention correction and training budget
matched. The two loss scales differ: equal weights are a starting ablation,
not matched gradient strength. Compare FM loss and dense-inference FID;
raw total losses from different alignment objectives are not directly comparable.
Feature runs additionally log `routesync/feature_cosine_mean`, `_r_mean`, `_p_mean`.
Existing checkpoints default to relational; resume validates the loss type to
prevent silently changing the objective. No long training/FID run is launched by
this implementation.


### Dense–sparse endpoint self-distillation

Add to a TREAD training command (can also combine with RouteSync):

```bash
--use-dense-sparse-sync \
--dense-sparse-sync-ratio 0.1 \
--dense-sparse-sync-weight 0.1
```

Disabled by default. On each training forward, select `max(1, floor(local_batch * ratio))`
images uniformly without replacement on each rank. Reuse their detached full-token
prefix features and the same timestep/class-dropout condition. Run only the routed
segment densely with current weights under `no_grad`; no extra prefix, suffix, or
prediction head forward. Align the sparse endpoint's active P tokens to the same
spatial positions of this dense endpoint, before reintegration. For `[3, 9)` this is
block 8's output (zero-based), independent of `--routesync-target-blocks`.

The auxiliary loss is **negative mean cosine**, computed in FP32 over channels,
then averaged over selected images and active tokens. It is added as
`L_total = L_FM + routesync_weight * L_RouteSync + dense_sparse_sync_weight * L_dense_sparse`.
The selected fraction does not additionally multiply the loss. Teacher has no gradient;
the student loss trains the routed segment and its upstream prefix. Existing RouteSync
remains independently configurable. Evaluation adds no teacher or auxiliary loss;
zero weight skips the teacher and sampling. No new model parameters/state keys.
The teacher preserves the student's endpoint precision and recursive logical depths;
attention correction automatically skips full-token teacher attention. Autocast's
weight cache is disabled inside the teacher to preserve subsequent student gradients.

Logs: `loss/dense_sparse_sync`, `loss/dense_sparse_sync_weighted`,
`dense_sparse_sync/cosine`, `dense_sparse_sync/samples_per_rank`, and
`dense_sparse_sync/actual_fraction`. At global batch 256 on 4 ranks, ratio 0.1 selects
6 of 64 images per rank (9.375%). Resume checks these training settings; old checkpoints
default to disabled. This is a training option, not a measured FID improvement.
