"""Single-forward sparse velocity comparisons; no dense teacher or model updates."""
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tread_routesync.config import TREAD_DEFAULTS
from tread_routesync.model import build_tread_sit


def patchify(x, patch):
    b, c, h, w = x.shape
    return x.reshape(b, c, h // patch, patch, w // patch, patch).permute(0, 2, 4, 3, 5, 1).reshape(b, -1, patch * patch * c)


def pair_geometry(side, patch, device):
    a, b = torch.triu_indices(side * side, side * side, offset=1, device=device)
    distance = ((a // side - b // side).float().square() + (a % side - b % side).float().square()).sqrt()
    masks = {'all_pairs': torch.ones_like(a, dtype=torch.bool), 'adjacent_patches': distance == 1,
             'distance_1_to_2': distance <= 2, 'distance_2_to_6': (distance > 2) & (distance <= 6), 'distance_gt_6': distance > 6}
    # Actual touching latent pixels across horizontal/vertical PATCH boundaries.
    width = side * patch
    pi, pj, ti, tj = [], [], [], []
    for row in range(side):
        for col in range(side):
            token = row * side + col
            if col + 1 < side:
                for offset in range(patch):
                    left = (row * patch + offset) * width + (col + 1) * patch - 1
                    pi.append(left); pj.append(left + 1); ti.append(token); tj.append(token + 1)
            if row + 1 < side:
                for offset in range(patch):
                    top = ((row + 1) * patch - 1) * width + col * patch + offset
                    pi.append(top); pj.append(top + width); ti.append(token); tj.append(token + side)
    boundary = [torch.tensor(v, device=device) for v in (pi, pj, ti, tj)]
    return a, b, masks, boundary


def metric_vectors(left, right):
    diff = left - right
    valid = (left.norm(dim=-1) > 1e-8) & (right.norm(dim=-1) > 1e-8)
    cosine = (F.normalize(left, dim=-1) * F.normalize(right, dim=-1)).sum(-1)
    return {'l1': diff.abs().mean(-1), 'squared_l2_mean': diff.square().mean(-1), 'cosine': cosine}, valid


def measure(v, target, info, patch, geometry, meta):
    v = v.float()
    batch, channels, height, width = v.shape
    roles = torch.zeros(batch, (height // patch) ** 2, dtype=torch.long, device=v.device)
    roles.scatter_(1, info['active_indices'], 1)  # R=0, P=1
    fields = {'prediction': v}
    if target is not None:
        fields.update(target=target.float(), error=v-target.float())
    a, b, scopes, boundary = geometry
    rows = []
    for field, values in fields.items():
        tokens = patchify(values, patch)
        for kind in ('patch', 'boundary'):
            if kind == 'patch':
                left, right = tokens[:, a], tokens[:, b]
                pair_role = roles[:, a] + roles[:, b]
                current_scopes = scopes
            else:
                pi, pj, ti, tj = boundary
                pixels = values.flatten(2).transpose(1, 2)
                left, right = pixels[:, pi], pixels[:, pj]
                pair_role = roles[:, ti] + roles[:, tj]
                current_scopes = {'boundary_pixels': torch.ones_like(pi, dtype=torch.bool)}
            metrics, valid = metric_vectors(left, right)
            for scope, spatial_mask in current_scopes.items():
                for role, name in enumerate(('R-R', 'R-P', 'P-P')):
                    select = (pair_role == role) & spatial_mask[None]
                    count = select.sum(1)
                    record = {'pair_count': count, 'cosine_valid_pairs': (select & valid).sum(1)}
                    for metric, value in metrics.items():
                        mask = select & valid if metric == 'cosine' else select
                        denom = mask.sum(1)
                        record[metric] = torch.where(denom > 0, (value * mask).sum(1) / denom.clamp_min(1), float('nan'))
                    record = {k: value.cpu().tolist() for k, value in record.items()}
                    for n in range(batch):
                        row = dict(meta[n], field=field, scope=scope, pair_type=name)
                        row.update({key: value[n] for key, value in record.items()})
                        row['rmse'] = math.sqrt(row['squared_l2_mean'])
                        rows.append(row)
    group_rows = []
    if target is not None:
        errors = patchify(v-target.float(), patch)
        for role, name in ((0, 'R'), (1, 'P')):
            mask = roles == role
            for metric, values in (('mse', errors.square().mean(-1)), ('mae', errors.abs().mean(-1))):
                vals = ((values * mask).sum(1) / mask.sum(1)).cpu().tolist()
                group_rows.extend(dict(meta[n], group=name, metric=metric, value=vals[n]) for n in range(batch))
    return rows, group_rows


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


@torch.inference_mode()
def main(args):
    start = time.time()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    saved = checkpoint['args']; saved = saved if isinstance(saved, dict) else vars(saved)
    if saved.get('path_type', 'linear') != 'linear' or saved.get('prediction', 'v') != 'v':
        raise ValueError('This diagnostic currently requires linear velocity prediction')
    config = {k: saved.get(k, default) for k, default in TREAD_DEFAULTS.items()}
    config.update(tread_eval_mode='sparse', tread_seed=None, tread_debug=False)
    if not config['use_tread_routing']:
        raise ValueError('Checkpoint must use TREAD routing')
    model = build_tread_sit(saved['model'], input_size=saved['resolution']//8,
        num_classes=saved['num_classes'], class_dropout_prob=saved.get('cfg_prob', .1),
        use_cfg=saved.get('cfg_prob', .1)>0, qk_norm=saved.get('qk_norm',False),
        fused_attn=saved.get('fused_attn',True), path_type='linear', **config)
    model.load_state_dict(checkpoint[args.state_key], strict=True)
    model = model.to(device).eval()
    del checkpoint
    patch = int(saved['model'].split('/')[1]); latent_size = saved['resolution']//8
    geometry = pair_geometry(latent_size//patch, patch, device)
    data_root = Path(args.data_dir or saved['data_dir']) / 'vae-sd'
    with (data_root/'dataset.json').open() as f:
        entries = json.load(f)['labels']
    rng = np.random.default_rng(args.seed)
    indices = rng.choice(len(entries), size=args.samples, replace=False)
    selected = [entries[int(i)] for i in indices]
    manifest = [{'sample_id': n, 'dataset_index': int(index), 'file': entry[0], 'label': entry[1]} for n, (index,entry) in enumerate(zip(indices, selected))]
    write_csv(out/'sample_manifest.csv', manifest)
    moments = torch.stack([torch.from_numpy(np.load(data_root/name)).reshape(8,latent_size,latent_size) for name,_ in selected]).float()
    mean, std = moments.chunk(2,1)
    generator = torch.Generator().manual_seed(args.seed+1)
    clean = (mean + std * torch.randn(mean.shape, generator=generator)) * .18215
    noise = torch.randn(clean.shape, generator=generator)
    labels = torch.tensor([label for _,label in selected], dtype=torch.long)
    metadata = {'checkpoint': str(Path(args.checkpoint).resolve()), 'checkpoint_args': saved,
        'analysis_args': vars(args), 'torch_version': torch.__version__, 'device_name': torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu',
        'protocol': 'EMA by default; sparse eval, conditional CFG=1, no dense passes. Data: sampled training latents, posterior sampling as training, x_t=(1-t)x0+t*noise, target=noise-x0. Same samples/noise across t and route repeats. Routing seed fixed per batch/repeat across t. Rollout: Euler ODE, fixed partition throughout each trajectory. No VAE decoding.',
        't_convention': '0=clean, 1=noise', 'confidence_unit': 'image; route repeats averaged before paired bootstrap',
        'precision': args.precision}
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    rows, groups = [], []
    def forward(x, t, y):
        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=args.precision=='bf16'):
            return model(x, torch.full((len(x),),float(t),device=device), y,
                         tread_eval_mode='sparse',return_routing_info=True)
    for offset in range(0,args.samples,args.batch_size):
        x0=clean[offset:offset+args.batch_size].to(device); eps=noise[offset:offset+args.batch_size].to(device)
        y=labels[offset:offset+args.batch_size].to(device)
        for route in range(args.route_repeats):
            model.tread_seed=args.seed+10000+offset*args.route_repeats+route
            for t in args.timesteps:
                pred,_,info=forward((1-t)*x0+t*eps,t,y)
                meta=[{'source':'noised_data','sample_id':offset+n,'route_repeat':route,'t':t} for n in range(len(x0))]
                new, grp=measure(pred,eps-x0,info,patch,geometry,meta); rows.extend(new); groups.extend(grp)
        print(f'data {offset+len(x0)}/{args.samples}, elapsed {time.time()-start:.1f}s',flush=True)
    write_csv(out/'per_image_pairs.csv', rows); write_csv(out/'per_image_fm.csv', groups)
    for offset in range(0,args.rollout_samples,args.batch_size):
        count=min(args.batch_size,args.rollout_samples-offset)
        x=torch.randn(count,4,latent_size,latent_size,generator=generator).to(device)
        y=torch.randint(saved['num_classes'],(count,),generator=generator).to(device)
        model.tread_seed=args.seed+200000+offset
        capture={0, args.rollout_steps//10, args.rollout_steps//4, args.rollout_steps//2, 3*args.rollout_steps//4,args.rollout_steps-1}
        for step in range(args.rollout_steps):
            t=1-step/args.rollout_steps
            pred,_,info=forward(x,t,y)
            if step in capture:
                meta=[{'source':'generated_trajectory','sample_id':offset+n,'route_repeat':0,'t':round(t,8)} for n in range(count)]
                new,_=measure(pred,None,info,patch,geometry,meta); rows.extend(new)
            x=x-pred.float()/args.rollout_steps
        if not torch.isfinite(x).all():
            raise RuntimeError('Nonfinite rollout')
        print(f'rollout {offset+count}/{args.rollout_samples}, elapsed {time.time()-start:.1f}s',flush=True)
    write_csv(out/'per_image_pairs.csv', rows)
    metadata['elapsed_seconds']=time.time()-start
    (out/'metadata.json').write_text(json.dumps(metadata,indent=2))
    print('Complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',required=True); p.add_argument('--output',required=True)
    p.add_argument('--data-dir'); p.add_argument('--state-key',default='ema',choices=['ema','model'])
    p.add_argument('--device',default='cuda:0'); p.add_argument('--precision',choices=['bf16','fp32'],default='bf16')
    p.add_argument('--samples',type=int,default=256); p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--route-repeats',type=int,default=2); p.add_argument('--seed',type=int,default=20260922)
    p.add_argument('--timesteps',type=float,nargs='+',default=[.05,.15,.3,.5,.7,.85,.95])
    p.add_argument('--rollout-samples',type=int,default=32); p.add_argument('--rollout-steps',type=int,default=50)
    main(p.parse_args())
