"""Single-GPU SiT-L/2 training-step comparison for dense vs TREAD.

GPU-resident synthetic ImageNet latents; no data loading, DDP, EMA or checkpoint I/O.
"""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path

import torch

from .loss import FlowMatchingLoss
from .model import build_tread_sit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--steps', type=int, default=8)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--output', default='tread_routesync/tread_l2_benchmark/results.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.set_device(args.device)
    device = torch.device(args.device)
    torch.manual_seed(2026)
    images = torch.randn(args.batch_size, 4, 32, 32, device=device)
    labels = torch.randint(0, 1000, (args.batch_size,), device=device)
    models = {}
    optimizers = {}
    criterion = FlowMatchingLoss('v', 'linear', 'uniform')
    for mode in ('dense', 'tread'):
        # Same seed gives corresponding parameters the same initialization.
        torch.manual_seed(1234)
        model = build_tread_sit(
            'SiT-L/2', input_size=32, num_classes=1000,
            class_dropout_prob=.1, use_cfg=True,
            qk_norm=False, fused_attn=True, path_type='linear',
            use_tread_routing=(mode == 'tread'),
            tread_start_block=2, tread_end_block=21,
            tread_active_ratio=.5, tread_fp32_endpoint=True,
        ).to(device).train()
        models[mode] = model
        optimizers[mode] = torch.optim.AdamW(
            model.parameters(), lr=1e-4, betas=(.9, .999),
            weight_decay=0., eps=1e-8,
        )
    result = {
        'gpu': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'batch_size': args.batch_size,
        'config': {'model': 'SiT-L/2', 'bf16': True, 'tf32': True,
                   'start_block': 2, 'end_block': 21, 'active_ratio': .5,
                   'fp32_endpoint': True, 'routesync': False,
                   'dense_sparse_sync': False},
        'scope': 'Forward, flow-matching loss, backward, grad clipping and AdamW step; GPU-resident synthetic latent and labels. Excludes DataLoader, VAE sampling, DDP, EMA, scheduler, logging and checkpoint I/O.',
        'shared_gpu': True,
        'rounds': [],
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    def step(mode):
        model = models[mode]
        opt = optimizers[mode]
        opt.zero_grad(set_to_none=True)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = criterion(model, images, {'y': labels})['total']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        opt.step()

    def time_steps(mode, count):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(count):
            step(mode)
        torch.cuda.synchronize(device)
        return (time.perf_counter() - start) * 1000 / count

    print('GPU:', result['gpu'], 'batch:', args.batch_size, flush=True)
    for mode in ('dense', 'tread'):
        print('warmup', mode, time_steps(mode, args.warmup), 'ms/step', flush=True)
    for round_id in range(args.rounds):
        order = ('dense', 'tread') if round_id % 2 == 0 else ('tread', 'dense')
        for mode in order:
            torch.cuda.reset_peak_memory_stats(device)
            before = torch.cuda.memory_allocated(device)
            ms = time_steps(mode, args.steps)
            row = {'round': round_id, 'mode': mode, 'ms_per_step': ms,
                   'images_per_second': args.batch_size * 1000 / ms,
                   'incremental_peak_mib': (torch.cuda.max_memory_allocated(device) - before) / 2**20}
            result['rounds'].append(row)
            print(row, flush=True)
            out.write_text(json.dumps(result, indent=2))
    summary = {}
    for mode in ('dense', 'tread'):
        values = [r['ms_per_step'] for r in result['rounds'] if r['mode'] == mode]
        summary[mode] = {'median_ms_per_step': statistics.median(values),
                         'range_ms': [min(values), max(values)],
                         'median_images_per_second': args.batch_size * 1000 / statistics.median(values)}
    summary['speedup_tread_over_dense'] = summary['dense']['median_ms_per_step'] / summary['tread']['median_ms_per_step']
    summary['time_reduction_percent'] = (1 - 1 / summary['speedup_tread_over_dense']) * 100
    result['summary'] = summary
    out.write_text(json.dumps(result, indent=2))
    print('summary', summary, flush=True)
    del models, optimizers
    gc.collect()


if __name__ == '__main__':
    main()
