# TREAD with one temporary dense routed block

This separate package keeps the original TREAD / RouteSync training and generation code, adding one optional training-time operation. On every training forward, choose **one logical block** from the routed interval `[start_block, end_block)` (zero-based, end exclusive). Immediately before that block, restore active and bypass tokens to original spatial order; execute this block with all tokens; then split its output using the **same token mask** and continue TREAD routing. The bypass tokens therefore participate in exactly one routed block and keep their updated features until the ordinary final merge. No additional transformer block call or loss is introduced.

The choice is **one block for the whole global batch**: rank 0 draws it and broadcasts the index to all GPU ranks. This keeps one uniform sequence length per block within each rank, avoids processing separate batch slices with different block schedules, and balances work across ranks. Token masks remain independent per image/rank. The default candidate set includes every block in the routed interval, uniformly. Inference and evaluation do not sample or execute this temporary dense step; dense inference remains the original full-token network.

## Run

Change only the training module and add `--tread-random-dense` to an existing TREAD command:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --multi_gpu --num_processes 8 --mixed_precision bf16 \
  -m tread_random_dense.train \
  --model SiT-L/2 --exp-name tread-random-dense-l \
  --data-dir /v/mnt/GH/imagenet_256 \
  --output-dir /v/mnt/GH/SiT \
  --use-tread-routing --tread-start-block 3 --tread-end-block 21 \
  --tread-active-ratio 0.5 --tread-random-dense \
  --batch-size 256 --max-train-steps 400000 --allow-tf32
```

The random dense block defaults to any of blocks **3–20** in this example. Limit the candidates if desired: `--tread-random-dense-blocks 7 11 15`. A single candidate fixes that block and does not draw an additional random number. The feature also works with `--tread-recursive`, grouped/interleaved weight sharing, depth embeddings, and optional RouteSync. With recursive sharing, the selection is one **logical call** of the reused block; other calls to the same weights stay sparse.

The feature is off by default. With it off, model parameters, checkpoint keys, random draws, training results, and logging match `tread_routesync`. The existing `loss/fm`, `loss/total`, `loss/route_sync`, optimizer, preview and checkpoint logs are retained. When enabled, `routing/random_dense_block` records the selected zero-based block each optimizer step. `args.json` and checkpoints record the feature and candidate set; resume checks that these settings match. New checkpoints load through `python -m tread_random_dense.generate`, and old TREAD checkpoints are strict-load compatible when this feature is disabled.

This raises compute for one block per step and changes how often bypass tokens receive gradients. A shared-GPU SiT-L/2 synthetic training-step benchmark measured **314.63 ms** for TREAD and **322.31 ms** with the random dense block (+2.44%); see [benchmark report](benchmark_results/report.md) for conditions and limitations. Whether it improves final dense FID needs a training comparison at matched steps and GPU time.

## Verify

```bash
python -m unittest tread_random_dense.tests.test_random_dense -v
```

Seven functional tests passed, including grouped recursive depth embeddings, exact mask reuse, and two-rank BF16 Gloo backward. A two-GPU NCCL BF16 smoke test also confirmed that both ranks select the same block and gradients remain finite.
