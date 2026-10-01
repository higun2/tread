import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from dataset import CustomDataset
from models.sit_b1 import SiT_models


def sample_posterior(moments, latents_scale=1.0, latents_bias=0.0):
    mean, std = torch.chunk(moments, 2, dim=1)
    z = mean + std * torch.randn_like(mean)
    z = z * latents_scale + latents_bias
    return z


def parse_depths(depths, n_blocks):
    unique_depths = sorted(set(depths))
    if not unique_depths:
        raise ValueError("Provide at least one depth via --encoder-depths.")
    if min(unique_depths) < 1:
        raise ValueError("Depth values are 1-based and must be >= 1.")

    indices = [d - 1 for d in unique_depths]
    for idx in indices:
        if idx < 0 or idx >= n_blocks:
            raise ValueError(f"Depth {idx + 1} is out of range. Valid range: 1..{n_blocks}")
    return unique_depths, indices


def create_capture_hook():
    def hook_fn(module, model_input, _output):
        x = model_input[0]
        bsz, n_tokens, channels = x.shape
        qkv = module.qkv(x).reshape(
            bsz, n_tokens, 3, module.num_heads, channels // module.num_heads
        ).permute(2, 0, 3, 1, 4)
        q, k, _v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * module.scale
        attn = attn.softmax(dim=-1)
        module.saved_attn = attn

    return hook_fn


def attn_to_map(attn):
    # attn shape: [B, heads, N, N]
    # Average over heads to get the full NxN attention matrix
    attn_matrix = attn.mean(dim=1)[0]  # [N, N]
    # print(f"Attention matrix shape: {attn_matrix.shape}")

    heatmap = attn_matrix.detach().float().cpu().numpy()
    heatmap = heatmap - heatmap.min()
    denom = float(heatmap.max() - heatmap.min())
    if denom > 1e-12:
        heatmap = heatmap / denom
    else:
        heatmap = np.zeros_like(heatmap)
    return heatmap


def attn_batch_to_matrices(attn):
    # attn shape: [B, heads, N, N] -> [B, N, N] (heads averaged)
    return attn.mean(dim=1).detach().float().cpu().numpy()


def normalize_heatmap(heatmap):
    heatmap = heatmap - heatmap.min()
    denom = float(heatmap.max() - heatmap.min())
    if denom > 1e-12:
        return heatmap / denom
    return np.zeros_like(heatmap)


def zero_upper_triangle(attn_matrix, include_diagonal=True):
    if attn_matrix.ndim != 2 or attn_matrix.shape[0] != attn_matrix.shape[1]:
        raise ValueError(
            f"Expected square 2D matrix, but got shape {attn_matrix.shape}"
        )
    masked = attn_matrix.copy()
    k = 0 if include_diagonal else 1
    masked[np.triu_indices(masked.shape[0], k=k)] = 0.0
    return masked


def compute_attention_metrics(attn_matrix, eps=1e-12):
    if attn_matrix.ndim != 2 or attn_matrix.shape[0] != attn_matrix.shape[1]:
        raise ValueError(
            f"Expected square 2D matrix, but got shape {attn_matrix.shape}"
        )

    n = attn_matrix.shape[0]
    if n <= 1:
        return 0.0, 0.0

    attn = np.clip(attn_matrix.astype(np.float64), 0.0, None)

    # Distance: average |i-j| weighted by attention over all query-key pairs.
    idx = np.arange(n, dtype=np.float64)
    distance_matrix = np.abs(idx[:, None] - idx[None, :]) / float(n - 1)
    total_weight = float(attn.sum())
    if total_weight > eps:
        attn_distance = float((attn * distance_matrix).sum() / total_weight)
    else:
        attn_distance = 0.0

    # Entropy: mean row-wise Shannon entropy, normalized by log(n).
    row_sums = attn.sum(axis=1, keepdims=True)
    valid_rows = (row_sums[:, 0] > eps)
    if not np.any(valid_rows):
        attn_entropy = 0.0
    else:
        row_probs = np.zeros_like(attn)
        row_probs[valid_rows] = attn[valid_rows] / row_sums[valid_rows]
        row_entropy = -(row_probs * np.log(np.clip(row_probs, eps, 1.0))).sum(axis=1)
        attn_entropy = float((row_entropy[valid_rows] / math.log(n)).mean())

    return attn_distance, attn_entropy


def row_normalize(attn_matrix, eps=1e-12):
    attn = np.clip(attn_matrix.astype(np.float64), 0.0, None)
    row_sums = attn.sum(axis=1, keepdims=True)
    probs = np.zeros_like(attn)
    valid_rows = row_sums[:, 0] > eps
    probs[valid_rows] = attn[valid_rows] / row_sums[valid_rows]
    return probs, valid_rows


def compute_pair_metrics(source_attn, target_attn, eps=1e-12):
    source_prob, source_valid = row_normalize(source_attn, eps=eps)
    target_prob, target_valid = row_normalize(target_attn, eps=eps)
    valid_rows = source_valid & target_valid
    if not np.any(valid_rows):
        return {
            "kl_to_target": 0.0,
            "reverse_kl": 0.0,
            "js_divergence": 0.0,
            "l1": 0.0,
            "l2": 0.0,
        }

    p = source_prob[valid_rows]
    q = target_prob[valid_rows]
    log_p = np.log(np.clip(p, eps, 1.0))
    log_q = np.log(np.clip(q, eps, 1.0))
    kl_to_target = float((q * (log_q - log_p)).sum(axis=1).mean())
    reverse_kl = float((p * (log_p - log_q)).sum(axis=1).mean())

    m = 0.5 * (p + q)
    log_m = np.log(np.clip(m, eps, 1.0))
    js = 0.5 * (q * (log_q - log_m)).sum(axis=1)
    js += 0.5 * (p * (log_p - log_m)).sum(axis=1)

    return {
        "kl_to_target": kl_to_target,
        "reverse_kl": reverse_kl,
        "js_divergence": float(js.mean()),
        "l1": float(np.abs(p - q).sum(axis=1).mean()),
        "l2": float(np.square(p - q).sum(axis=1).mean()),
    }


def save_metrics(metrics_by_depth, pair_metrics, output_dir):
    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    pair_by_depth = {item["depth"]: item for item in pair_metrics}
    fieldnames = [
        "depth",
        "distance",
        "entropy",
        "target_depth",
        "kl_to_target",
        "reverse_kl",
        "js_divergence",
        "l1",
        "l2",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for depth, attn_distance, attn_entropy in metrics_by_depth:
            row = {
                "depth": depth,
                "distance": attn_distance,
                "entropy": attn_entropy,
                "target_depth": "",
                "kl_to_target": "",
                "reverse_kl": "",
                "js_divergence": "",
                "l1": "",
                "l2": "",
            }
            if depth in pair_by_depth:
                pair_row = pair_by_depth[depth]
                row.update({k: pair_row[k] for k in fieldnames if k in pair_row})
            writer.writerow(row)

    payload = {
        "per_depth": [
            {"depth": depth, "distance": dist, "entropy": ent}
            for depth, dist, ent in metrics_by_depth
        ],
        "pair_to_target": pair_metrics,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return csv_path, json_path


def save_heatmap(heatmap, output_path, upscale=16):
    # Use a light-to-red colormap to emphasize strong attention on a pale background.
    cmap = cm.get_cmap('Reds')
    colored = cmap(heatmap)  # Apply colormap to get RGBA
    rgb = (colored[:, :, :3] * 255).astype(np.uint8)  # Extract RGB and convert to 0-255

    pil = Image.fromarray(rgb, mode="RGB")
    pil = pil.resize((pil.width * upscale, pil.height * upscale), resample=Image.NEAREST)
    pil.save(output_path)


def save_heatmap_grid(heatmaps, output_path, cols=4):
    if not heatmaps:
        return

    cols = max(1, int(cols))
    n_items = len(heatmaps)
    rows = math.ceil(n_items / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.0, rows * 3.0))
    if isinstance(axes, np.ndarray):
        axes = axes.reshape(rows, cols)
    else:
        axes = np.array([[axes]])

    for i, item in enumerate(heatmaps):
        if len(item) == 4:
            depth, heatmap, attn_distance, attn_entropy = item
            title = f"Depth {depth}\nDist {attn_distance:.3f} | Ent {attn_entropy:.3f}"
        else:
            depth, heatmap = item
            title = f"Depth {depth}"
        r = i // cols
        c = i % cols
        ax = axes[r, c]
        ax.imshow(heatmap, cmap="Reds", interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_items, rows * cols):
        r = i // cols
        c = i % cols
        axes[r, c].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def load_label_from_dataset_json(data_dir, relative_feature_path):
    labels_json = Path(data_dir) / "vae-sd" / "dataset.json"
    with open(labels_json, "r", encoding="utf-8") as f:
        labels = dict(json.load(f)["labels"])

    key = relative_feature_path.replace("\\", "/")
    if key not in labels:
        raise KeyError(f"Label not found for feature path '{key}' in {labels_json}")
    return int(labels[key])


def load_single_moments(args):
    if args.feature_path is not None:
        feature_path = Path(args.feature_path)
        moments = np.load(feature_path)
        moments = torch.from_numpy(moments)

        if args.label is not None:
            label = int(args.label)
        elif args.data_dir is not None:
            features_root = Path(args.data_dir) / "vae-sd"
            rel = str(feature_path.relative_to(features_root)).replace("\\", "/")
            label = load_label_from_dataset_json(args.data_dir, rel)
        else:
            label = 0
        source_name = str(feature_path)
    else:
        if args.data_dir is None:
            raise ValueError("Either --feature-path or --data-dir must be provided.")

        dataset = CustomDataset(args.data_dir, num_classes=args.num_classes)
        moments, label = dataset[args.index]
        label = int(label.item())
        source_name = f"dataset_index_{args.index}"

    return moments, label, source_name


def main(args):
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    latent_size = args.resolution // 8
    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        use_cfg=(args.cfg_prob > 0),
        class_dropout_prob=args.cfg_prob,
        fused_attn=args.fused_attn,
        qk_norm=args.qk_norm,
    ).to(device)
    model.eval()

    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    state_key = "ema" if args.use_ema else "model"
    if state_key not in ckpt:
        raise KeyError(f"Checkpoint does not contain '{state_key}' weights.")
    model.load_state_dict(ckpt[state_key], strict=True)

    depth_list, attn_indices = parse_depths(args.encoder_depths, len(model.blocks))

    hooks = []
    target_attn_modules = []
    for idx in attn_indices:
        attn_module = model.blocks[idx].attn
        attn_module.saved_attn = None
        hooks.append(attn_module.register_forward_hook(create_capture_hook()))
        target_attn_modules.append(attn_module)

    latents_scale = torch.tensor([0.18215] * 4, device=device).view(1, 4, 1, 1)
    latents_bias = torch.zeros((1, 4, 1, 1), device=device)

    attn_sums = [None] * len(target_attn_modules)
    sample_count = 0

    def validate_and_prepare_moments(batch_moments):
        if batch_moments.dim() == 5 and batch_moments.shape[1] == 1:
            batch_moments = batch_moments.squeeze(1)
        if batch_moments.dim() == 4:
            return batch_moments
        raise ValueError(
            f"Expected moments shape [B,C,H,W] or [B,1,C,H,W], but got {tuple(batch_moments.shape)}"
        )

    def run_batch(batch_moments, batch_labels):
        nonlocal sample_count
        batch_moments = validate_and_prepare_moments(batch_moments)
        x = sample_posterior(
            batch_moments.to(device),
            latents_scale=latents_scale,
            latents_bias=latents_bias,
        )

        if args.label is not None:
            y = torch.full((x.shape[0],), int(args.label), device=device, dtype=torch.long)
        else:
            y = batch_labels.to(device=device, dtype=torch.long)

        t = torch.full((x.shape[0],), float(args.timestep), device=device, dtype=torch.float32)

        for attn_module in target_attn_modules:
            attn_module.saved_attn = None

        with torch.no_grad():
            _ = model(x, t, y)

        for i, attn_module in enumerate(target_attn_modules):
            if attn_module.saved_attn is None:
                raise RuntimeError(
                    f"Attention for depth {depth_list[i]} was not captured. Check model forward path and hooks."
                )
            batch_maps = attn_batch_to_matrices(attn_module.saved_attn)
            batch_sum = batch_maps.sum(axis=0)
            if attn_sums[i] is None:
                attn_sums[i] = batch_sum
            else:
                attn_sums[i] += batch_sum
        sample_count += x.shape[0]

    if args.feature_path is not None:
        moments, label, source_name = load_single_moments(args)
        if moments.dim() == 3:
            moments = moments.unsqueeze(0)
        elif moments.dim() == 4 and moments.shape[0] == 1:
            pass
        else:
            raise ValueError(
                f"Expected moments shape [C,H,W] or [1,C,H,W], but got {tuple(moments.shape)}"
            )
        run_batch(moments, torch.tensor([label], dtype=torch.long))
        source_name = f"single_feature:{source_name}"
    else:
        if args.data_dir is None:
            raise ValueError("Either --feature-path or --data-dir must be provided.")
        dataset = CustomDataset(args.data_dir, num_classes=args.num_classes)
        if len(dataset) == 0:
            raise ValueError("Dataset is empty.")

        target_samples = min(args.avg_samples, len(dataset))
        loader = DataLoader(
            dataset,
            batch_size=max(1, args.avg_batch_size),
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
        )

        consumed = 0
        for batch_moments, batch_labels in loader:
            remaining = target_samples - consumed
            if remaining <= 0:
                break
            if batch_moments.shape[0] > remaining:
                batch_moments = batch_moments[:remaining]
                batch_labels = batch_labels[:remaining]
            run_batch(batch_moments, batch_labels)
            consumed += batch_moments.shape[0]

        source_name = f"dataset_avg:{target_samples}_samples_from_{args.data_dir}"

    if sample_count == 0:
        raise RuntimeError("No samples were processed for attention map generation.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    saved_files = []
    metrics_by_depth = []
    avg_attn_by_depth = {}
    collected_heatmaps = []
    for depth, attn_sum in zip(depth_list, attn_sums):
        if attn_sum is None:
            raise RuntimeError(f"Attention sum for depth {depth} is empty.")
        avg_attn = attn_sum / float(sample_count)
        avg_attn_by_depth[depth] = avg_attn
        attn_distance, attn_entropy = compute_attention_metrics(avg_attn)
        metrics_by_depth.append((depth, attn_distance, attn_entropy))
        if args.zero_upper_triangle:
            avg_attn = zero_upper_triangle(avg_attn, include_diagonal=args.zero_diagonal)
        heatmap = normalize_heatmap(avg_attn)
        out_path = output_dir / f"attn_depth_{depth:02d}.png"
        save_heatmap(heatmap, out_path, upscale=args.upscale)
        saved_files.append(out_path)
        collected_heatmaps.append((depth, heatmap, attn_distance, attn_entropy))

    grid_path = None
    if args.save_grid:
        grid_path = output_dir / "attn_depth_grid.png"
        save_heatmap_grid(collected_heatmaps, grid_path, cols=args.grid_cols)

    pair_metrics = []
    if args.reference_depth is not None:
        if args.reference_depth not in avg_attn_by_depth:
            raise ValueError(
                f"--reference-depth {args.reference_depth} was not captured. "
                "Include it in --encoder-depths."
            )
        compare_depths = args.compare_depths or [
            depth for depth in depth_list if depth != args.reference_depth
        ]
        for depth in compare_depths:
            if depth not in avg_attn_by_depth:
                raise ValueError(
                    f"--compare-depths contains {depth}, but it was not captured. "
                    "Include it in --encoder-depths."
                )
            pair = compute_pair_metrics(
                avg_attn_by_depth[depth],
                avg_attn_by_depth[args.reference_depth],
            )
            pair_metrics.append(
                {
                    "depth": depth,
                    "target_depth": args.reference_depth,
                    **pair,
                }
            )

    metrics_csv_path, metrics_json_path = save_metrics(metrics_by_depth, pair_metrics, output_dir)

    for hook in hooks:
        hook.remove()

    print(f"Loaded source: {source_name}")
    print(f"Checkpoint: {args.ckpt_path}")
    print(f"Used weights: {state_key}")
    print(f"Samples averaged: {sample_count}, Timestep: {args.timestep}")
    if args.zero_upper_triangle:
        diag_text = "including diagonal" if args.zero_diagonal else "excluding diagonal"
        print(f"Mask applied: upper triangle zeroed ({diag_text})")
    if args.label is not None:
        print(f"Label override: {int(args.label)}")
    print("Saved attention maps:")
    for path in saved_files:
        print(f"- {path}")
    if grid_path is not None:
        print(f"Combined grid: {grid_path}")
    print(f"Metrics CSV: {metrics_csv_path}")
    print(f"Metrics JSON: {metrics_json_path}")
    print("Per-depth metrics (computed on unmasked average attention):")
    for depth, attn_distance, attn_entropy in metrics_by_depth:
        print(f"- depth {depth}: distance={attn_distance:.6f}, entropy={attn_entropy:.6f}")
    if pair_metrics:
        print(f"Pair metrics to reference depth {args.reference_depth}:")
        for item in pair_metrics:
            print(
                f"- depth {item['depth']} -> {item['target_depth']}: "
                f"KL={item['kl_to_target']:.6f}, JS={item['js_divergence']:.6f}, "
                f"L1={item['l1']:.6f}, L2={item['l2']:.6f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot attention maps from selected SiT blocks.")

    parser.add_argument("--ckpt-path", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--model", type=str, default="SiT-B/2", help="Model key in SiT_models")
    parser.add_argument("--resolution", type=int, default=256, choices=[256, 512])
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument(
        "--encoder-depths",
        type=int,
        nargs="+",
        required=True,
        help="1-based block depths, e.g. --encoder-depths 3 5 7",
    )

    parser.add_argument("--data-dir", type=str, default=None, help="Dataset root that contains vae-sd/")
    parser.add_argument("--index", type=int, default=0, help="Sample index when --data-dir is used")
    parser.add_argument("--feature-path", type=str, default=None, help="Direct path to one .npy moment file")
    parser.add_argument("--avg-samples", type=int, default=5000, help="Number of dataset samples to average when --feature-path is not set")
    parser.add_argument("--avg-batch-size", type=int, default=16, help="Batch size for averaging mode")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers for averaging mode")

    parser.add_argument("--label", type=int, default=None, help="Override label")
    parser.add_argument("--timestep", type=float, default=0.7, help="Diffusion timestep used in forward")
    parser.add_argument("--reference-depth", type=int, default=None, help="Target block depth for pairwise attention metrics")
    parser.add_argument("--compare-depths", type=int, nargs="*", default=None, help="Source block depths compared to --reference-depth")

    parser.add_argument("--cfg-prob", type=float, default=0.1)
    parser.add_argument("--fused-attn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--qk-norm", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use-ema", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--output-dir", type=str, default="attention_plots")
    parser.add_argument("--upscale", type=int, default=16)
    parser.add_argument(
        "--zero-upper-triangle",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Zero upper-triangle values before normalization/plotting",
    )
    parser.add_argument(
        "--zero-diagonal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="When zeroing upper triangle, also zero the diagonal",
    )
    parser.add_argument("--save-grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid-cols", type=int, default=4, help="Number of columns in combined grid image")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None, help="cuda, cpu, or leave empty for auto")

    cli_args = parser.parse_args()
    main(cli_args)


'''
CUDA_VISIBLE_DEVICES=4 python plot_block_attention.py \
  --ckpt-path /mnt/GH/SiT/try0/checkpoints/0400000.pt \
  --model "SiT-B/2" \
  --encoder-depths 1 2 3 4 5 6 7 8 9 10 11\
  --grid-cols 4 \
  --timestep 0.0 \
  --output-dir attention_plots_a \
  --data-dir /mnt/GH/imagenet_256 \
  --feature-path /mnt/GH/imagenet_256/vae-sd/00001/img-mean-std-00001001.npy \

CUDA_VISIBLE_DEVICES=5 python plot_block_attention.py \
  --ckpt-path /mnt/GH/SiT/try30/checkpoints/0400000.pt \
  --model "SiT-B/2" \
  --encoder-depths 1 2 3 4 5 6 7 8 9 10 11\
  --grid-cols 4 \
  --timestep 0.0 \
  --output-dir attention_plots_b \
  --data-dir /mnt/GH/imagenet_256 \
  --feature-path /mnt/GH/imagenet_256/vae-sd/00001/img-mean-std-00001001.npy \
'''


'''
CUDA_VISIBLE_DEVICES=4 python plot_block_attention.py \
  --ckpt-path /mnt/GH/SiT/try30-XL/checkpoints/0400000.pt \
  --model "SiT-XL/2" \
  --encoder-depths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 \
  --grid-cols 4 \
  --timestep 0.5 \
  --output-dir attention_plots_a \
  --data-dir /mnt/GH/imagenet_256 \
  --feature-path /mnt/GH/imagenet_256/vae-sd/00001/img-mean-std-00001001.npy \

CUDA_VISIBLE_DEVICES=5 python plot_block_attention.py \
  --ckpt-path /mnt/GH/SiT/try30-XL/checkpoints/sit_0400000.pt \
  --model "SiT-XL/2" \
  --encoder-depths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 \
  --grid-cols 4 \
  --timestep 0.5 \
  --output-dir attention_plots_b \
  --data-dir /mnt/GH/imagenet_256 \
  --feature-path /mnt/GH/imagenet_256/vae-sd/00001/img-mean-std-00001001.npy \


CUDA_VISIBLE_DEVICES=6 /home/compu/miniconda3/envs/sit/bin/python plot_block_attention.py --ckpt-path /mnt/GH/SiT/try30-XL/checkpoints/0400000.pt --model SiT-XL/2 --encoder-depths 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 --grid-cols 4 --timestep 0.3 --output-dir attention_analysis_1000/sync_t0.30 --data-dir /mnt/GH/imagenet_256 --avg-samples 1000 --avg-batch-size 4 --num-workers 2 --reference-depth 18
'''
