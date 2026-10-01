"""Read-only TREAD attention audit: frozen-QKV and actual routed trajectories."""
import argparse
import csv
import json
import math
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tread_token_routing.config import TREAD_DEFAULTS
from tread_token_routing.model import build_tread_sit, random_subset_indices

MODES = ['dense', 'local_sparse', 'local_corrected', 'full_sparse', 'full_corrected']
METRICS = ['self_mass', 'output_mse', 'output_mae', 'output_cosine', 'output_relative_rmse']


def take(x, indices):
    return x.gather(2, indices[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[-1]))


def attend(q, k, v, scale, correction=0.):
    scores = (q * scale) @ k.transpose(-1, -2)
    if correction:
        scores.diagonal(dim1=-2, dim2=-1).add_(correction)
    a = scores.softmax(-1)
    return a.diagonal(dim1=-2, dim2=-1), a @ v


def measure(diag, output, reference):
    delta = output - reference
    mse = delta.square().mean((-2, -1))
    return torch.stack([
        diag.mean(-1), mse, delta.abs().mean((-2, -1)),
        F.cosine_similarity(output, reference, dim=-1).mean(-1),
        (mse / reference.square().mean((-2, -1)).clamp_min(1e-20)).sqrt(),
    ], -1).cpu().numpy()


def write_csv(path, rows):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class Audit:
    def __init__(self, model):
        self.model = model
        self.mode = 'dense'
        self.cache = {}
        self.records = {}
        self.indices = None
        self.correction = 0.
        self.originals = {}
        for b in range(model.tread_start_block, model.tread_end_block):
            module = model.blocks[b].attn
            self.originals[b] = module.forward
            def forward(module, x, attn_mask=None, block=b):
                assert attn_mask is None
                batch, n, _ = x.shape
                q, k, v = module.qkv(x).reshape(batch, n, 3, module.num_heads, module.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
                q, k = module.q_norm(q), module.k_norm(k)
                correction = self.correction if self.mode == 'full_corrected' else 0.
                diag, output = attend(q, k, v, module.scale, correction)
                if self.mode == 'dense':
                    self.cache[block] = (q, k, v, diag, output)
                else:
                    reference = take(self.cache[block][4], self.indices)
                    self.records[block] = measure(diag, output, reference)
                result = output.transpose(1, 2).reshape(batch, n, module.attn_dim)
                return module.proj_drop(module.proj(module.norm(result)))
            module.forward = types.MethodType(forward, module)


@torch.inference_mode()
def main(args):
    start = time.time()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    saved = ckpt['args']
    saved = saved if isinstance(saved, dict) else vars(saved)
    assert saved['path_type'] == 'linear' and saved['prediction'] == 'v'
    config = {k: saved.get(k, v) for k, v in TREAD_DEFAULTS.items()}
    assert config['use_tread_routing'] and not config['tread_recursive']
    config.update(tread_eval_mode='sparse', tread_seed=None, tread_debug=False)
    model = build_tread_sit(saved['model'], input_size=saved['resolution']//8,
        num_classes=saved['num_classes'], class_dropout_prob=saved.get('cfg_prob', .1),
        use_cfg=saved.get('cfg_prob', .1)>0, qk_norm=saved.get('qk_norm', False),
        fused_attn=saved.get('fused_attn', True), path_type='linear', **config)
    model.load_state_dict(ckpt[args.state_key], strict=True)
    model.to(device).eval()
    del ckpt
    root = Path(args.data_dir or saved['data_dir']) / 'vae-sd'
    entries = json.loads((root/'dataset.json').read_text())['labels']
    selected_ids = np.random.default_rng(args.seed).choice(len(entries), args.samples, replace=False)
    selected = [entries[int(i)] for i in selected_ids]
    write_csv(out/'sample_manifest.csv', [dict(sample_id=n, dataset_index=int(i), file=e[0], label=e[1]) for n, (i,e) in enumerate(zip(selected_ids, selected))])
    side = saved['resolution']//8
    moments = torch.stack([torch.from_numpy(np.load(root/name)).reshape(8,side,side) for name,_ in selected]).float()
    mean, std = moments.chunk(2, 1)
    rng = torch.Generator().manual_seed(args.seed+1)
    clean = (mean + std*torch.randn(mean.shape, generator=rng))*.18215
    noise = torch.randn(clean.shape, generator=rng)
    labels = torch.tensor([y for _,y in selected], dtype=torch.long)
    layers = list(range(model.tread_start_block, model.tread_end_block))
    heads = model.blocks[0].attn.num_heads
    n = model.x_embedder.num_patches
    count = round(n*model.tread_active_ratio)
    correction = math.log((count-1)/(n-1))
    shape = (args.samples, len(args.timesteps), args.route_repeats, len(layers), len(MODES), heads, len(METRICS))
    values = np.full(shape, np.nan, dtype=np.float32)
    routing = np.zeros((args.samples, args.route_repeats, count), dtype=np.int64)
    final_rows = []
    audit = None
    checks = {}
    metadata = dict(checkpoint=str(Path(args.checkpoint).resolve()), checkpoint_args=saved,
        analysis_args=vars(args), state_key=args.state_key, device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        torch_version=torch.__version__, precision='FP32; TF32 disabled; routed attention explicit softmax',
        layers_zero_based=layers, modes=MODES, metrics=METRICS, array_axes=['image','timestep','route_repeat','layer','mode','head','metric'],
        correction=correction, tokens=n, active_tokens=count,
        protocol='Training-data latent audit, not held-out FID. t=0 clean, t=1 noise; EMA; conditional CFG=1. Same images, posterior samples, noise, and route masks across modes/t. Local modes use each dense layer\'s frozen QKV. Full modes run actual sparse networks; corrected mode changes all routed attention diagonals. Compare only matching active queries. Output=per-head AV before output projection. All inference only; no weight updates.')
    (out/'metadata.json').write_text(json.dumps(metadata, indent=2))
    for offset in range(0, args.samples, args.batch_size):
        end = min(offset+args.batch_size, args.samples)
        x0, eps, y = clean[offset:end].to(device), noise[offset:end].to(device), labels[offset:end].to(device)
        for ti, t in enumerate(args.timesteps):
            xt = (1-t)*x0+t*eps
            tt = torch.full((len(x0),), t, device=device)
            if audit is None:
                native_dense = model(xt,tt,y,tread_eval_mode='dense')[0]
                model.tread_seed = args.seed+10000+offset*args.route_repeats
                native_sparse = model(xt,tt,y,tread_eval_mode='sparse')[0]
                audit = Audit(model)
                audit.correction = correction
            audit.mode = 'dense'
            dense = model(xt,tt,y,tread_eval_mode='dense')[0]
            if offset == 0 and ti == 0:
                checks['dense_instrumentation_max_abs'] = (native_dense-dense).abs().max().item()
                checks['dense_instrumentation_rmse'] = (native_dense-dense).square().mean().sqrt().item()
                assert torch.allclose(native_dense, dense, atol=3e-4, rtol=3e-4)
            for repeat in range(args.route_repeats):
                seed = args.seed+10000+offset*args.route_repeats+repeat
                model.tread_seed = seed
                gen = torch.Generator(device=device).manual_seed(seed)
                indices, _ = random_subset_indices(len(x0), n, count, device, gen)
                audit.indices = indices
                routing[offset:end,repeat] = indices.cpu().numpy()
                for li,b in enumerate(layers):
                    q,k,v,diag,output = audit.cache[b]
                    ref = take(output, indices)
                    dense_diag = diag.gather(2,indices[:,None].expand(-1,heads,-1))
                    values[offset:end,ti,repeat,li,0] = measure(dense_diag,ref,ref)
                    qs,ks,vs = [take(z,indices) for z in (q,k,v)]
                    for mi, corr in ((1,0.),(2,correction)):
                        ds, os = attend(qs,ks,vs,model.blocks[b].attn.scale,corr)
                        values[offset:end,ti,repeat,li,mi] = measure(ds,os,ref)
                predictions = {'dense':dense}
                for mi in (3,4):
                    audit.mode = MODES[mi]
                    pred, _, info = model(xt,tt,y,tread_eval_mode='sparse',return_routing_info=True)
                    assert torch.equal(info['active_indices'], indices)
                    predictions[MODES[mi]] = pred
                    for li,b in enumerate(layers):
                        values[offset:end,ti,repeat,li,mi] = audit.records[b]
                    # First routed block must agree with fixed-QKV control.
                    discrepancy = np.max(np.abs(values[offset:end,ti,repeat,0,mi]-values[offset:end,ti,repeat,0,mi-2]))
                    checks['first_layer_local_full_max_metric_difference'] = max(float(discrepancy), checks.get('first_layer_local_full_max_metric_difference',0.))
                    assert discrepancy < 3e-4, discrepancy
                    if offset == 0 and ti == 0 and repeat == 0 and mi == 3:
                        checks['sparse_instrumentation_max_abs'] = (native_sparse-pred).abs().max().item()
                        assert torch.allclose(native_sparse,pred,atol=3e-4,rtol=3e-4)
                for mode,pred in predictions.items():
                    fm = (pred-(eps-x0)).square().mean((1,2,3)).cpu().numpy()
                    mse = (pred-dense).square().mean((1,2,3)).cpu().numpy()
                    for i in range(len(x0)):
                        final_rows.append(dict(sample_id=offset+i,t=t,route_repeat=repeat,mode=mode,fm_mse=float(fm[i]),dense_prediction_mse=float(mse[i])))
        np.savez_compressed(out/'measurements.npz',values=values,active_indices=routing,timesteps=np.array(args.timesteps),layers=np.array(layers),modes=np.array(MODES),metrics=np.array(METRICS))
        write_csv(out/'final_velocity.csv', final_rows)
        metadata.update(completed_samples=end,elapsed_seconds=time.time()-start,validation=checks)
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
        print(f'{end}/{args.samples} images; elapsed={time.time()-start:.1f}s; checks={checks}',flush=True)
    assert np.isfinite(values).all()
    print('Complete',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--data-dir')
    p.add_argument('--state-key',choices=['ema','model'],default='ema')
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--samples',type=int,default=128)
    p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--route-repeats',type=int,default=2)
    p.add_argument('--seed',type=int,default=20260923)
    p.add_argument('--timesteps',type=float,nargs='+',default=[.05,.15,.3,.5,.7,.85,.95])
    main(p.parse_args())
