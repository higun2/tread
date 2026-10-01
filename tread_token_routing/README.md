# Fixed-Subset TREAD-Style Routing

This folder provides a Core-free sparse SiT experiment. At the entrance to a configurable routed block range, every sample independently selects a uniform random fixed-capacity subset of image tokens. The default active ratio is 0.5. The same selected subset passes through every routed block, while the complement follows an identity bypass and is restored to its original positions at the routed exit.

For a 256-token SiT-B/2 sequence and `[3,9)`:

```text
blocks 0-2:   all 256 tokens
blocks 3-8:   the same random 128 active tokens
blocks 9-11:  all 256 tokens
```

There is no persistent Core, relay token group, learned router, score, auxiliary loss, or token merging. By default every routed Transformer block retains its own parameters; optional recursive block sharing is described below. Positional embeddings are added before selection, and class/timestep conditioning remains dense adaLN conditioning. The active and bypass streams are gathered once, carried separately, and restored once without an in-place hidden-state update. The bypass stream remains connected to autograd.

## Train

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m tread_token_routing.train \
  --model SiT-B/2 --exp-name tread-half-b \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-tread-routing \
  --tread-start-block 3 --tread-end-block 9 \
  --tread-active-ratio 0.5 \
  --tread-recursive --tread-num-groups 3 \
  --tread-recursive-pattern grouped \
  --batch-size 256 --max-train-steps 400000 \
  --allow-tf32
```

Training always uses sparse routing when `--use-tread-routing` is enabled. The active subset is independently resampled per sample and forward. `--tread-active-ratio` is fixed capacity: with ratio 0.5, exactly half of the tokens are active in each routed block.

### Recursive routed blocks

`--tread-recursive` retains one independent SiTBlock per group. With `[3,9)` and three groups, `--tread-recursive-pattern` selects either execution order:

```text
non-recursive: A1 B1 C1 D1 E1 F1
grouped:       A1 A1 B1 B1 C1 C1
interleaved:   A1 B1 C1 A1 B1 C1
```

The same randomly selected 50% subset passes through all six executions. A1, B1, and C1 have different parameters; each is called twice. Use `--tread-recursive-pattern grouped` for `AABBCC` or `--tread-recursive-pattern interleaved` for `ABCABC`. Dense inference repeats the selected order on all tokens, while sparse inference repeats it on the selected subset. Recursive topology and pattern are saved in the checkpoint and restored automatically during generation. Omit `--tread-recursive` to use six independent routed blocks.

`--tread-fp32-endpoint` is enabled by default. Under BF16 autocast only block `tread_end_block - 1` runs in FP32 during training, then its output returns to the incoming hidden dtype. Parameters and optimizer state are not converted. Disable this ablation with `--no-tread-fp32-endpoint`.

The EMA preview at step 1 and every `--sampling-steps` always uses dense execution and is logged as one W&B image grid. Individual preview images are not saved.

## Generate

Sparse inference restores the training ratio and range from the checkpoint. `--tread-seed` makes the subset reproducible without changing global PyTorch RNG state:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m tread_token_routing.generate \
  --ckpt /v/mnt/GH/SiT/tread-half-b/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --tread-eval-mode sparse --tread-seed 1234 \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

Dense inference sends all tokens through all blocks:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --standalone --nproc_per_node=8 \
  -m tread_token_routing.generate \
  --ckpt /v/mnt/GH/SiT/tread-half-b/checkpoints/0400000.pt \
  --sample-dir /v/mnt/GH/samples \
  --tread-eval-mode dense \
  --num-fid-samples 50000 --per-proc-batch-size 32 \
  --mode sde --num-steps 250 --cfg-scale 1.0
```

Inference uses the repository's existing FP32 generation behavior. The training-only endpoint stabilization is inactive in both sparse and dense inference.

## Tests

```bash
python -m unittest tread_token_routing.tests.test_tread -v
```

The tests cover exact disabled-baseline equivalence, exact 50/50 disjoint partitioning, fixed subset reuse, grouped `A1 A1 B1 B1 C1 C1` and interleaved `A1 B1 C1 A1 B1 C1` recursive execution, strict recursive checkpoint loading, equivalence to an out-of-place full-state scatter reference, local seeded RNG, sparse/dense inference, bypass autograd, BF16/FP32 endpoint precision, optimizer dtype preservation, dense W&B preview policy, DDP, and CUDA fused/non-fused backward.


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
