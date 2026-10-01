"""GPU-resident SiT-L/2 step timing: TREAD vs one random dense routed block."""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from .loss import FlowMatchingLoss
from .model import build_tread_sit


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--warmup',type=int,default=3)
    parser.add_argument('--steps',type=int,default=8)
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--output',default='tread_random_dense/benchmark_results/results.json')
    args=parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    device=torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(2026)
    images=torch.randn(args.batch_size,4,32,32,device=device)
    labels=torch.randint(0,1000,(args.batch_size,),device=device)
    modes=('tread','random_dense')
    models={};optimizers={}
    for mode in modes:
        torch.manual_seed(1234)
        model=build_tread_sit('SiT-L/2',input_size=32,num_classes=1000,
            class_dropout_prob=.1,use_cfg=True,qk_norm=False,fused_attn=True,
            path_type='linear',use_tread_routing=True,
            tread_start_block=3,tread_end_block=21,tread_active_ratio=.5,
            tread_fp32_endpoint=True,tread_random_dense=mode=='random_dense',
        ).to(device).train()
        models[mode]=model
        optimizers[mode]=torch.optim.AdamW(model.parameters(),lr=1e-4,
            betas=(.9,.999),weight_decay=0.,eps=1e-8)
    loss=FlowMatchingLoss('v','linear','uniform')
    def step(mode):
        model=models[mode];optimizer=optimizers[mode]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            value=loss(model,images,{'y':labels})['total']
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
    def timed(mode,count):
        torch.cuda.synchronize(device)
        start=time.perf_counter()
        for _ in range(count):step(mode)
        torch.cuda.synchronize(device)
        return (time.perf_counter()-start)*1000/count
    result={'gpu':torch.cuda.get_device_name(device),'torch':torch.__version__,
        'batch_size':args.batch_size,'warmup':args.warmup,'steps_per_round':args.steps,
        'rounds_requested':args.rounds,'tread_route':[3,21],
        'active_ratio':.5,'bf16':True,'tf32':True,'shared_gpu':True,
        'scope':'flow-matching forward+backward, gradient clipping, AdamW step; no data loader, DDP, EMA, checkpoint, preview or logging',
        'measurements':[]}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    for mode in modes:
        print('warmup',mode,round(timed(mode,args.warmup),2),flush=True)
    for i in range(args.rounds):
        order=modes if i%2==0 else tuple(reversed(modes))
        for mode in order:
            ms=timed(mode,args.steps)
            row={'round':i,'mode':mode,'ms_per_step':ms,
                'images_per_second':args.batch_size*1000/ms}
            result['measurements'].append(row)
            output.write_text(json.dumps(result,indent=2))
            print(row,flush=True)
    medians={mode:statistics.median(x['ms_per_step'] for x in result['measurements'] if x['mode']==mode) for mode in modes}
    result['summary']={'median_ms_per_step':medians,
        'random_dense_overhead_percent':100*(medians['random_dense']/medians['tread']-1)}
    output.write_text(json.dumps(result,indent=2))
    print('summary',result['summary'],flush=True)


if __name__=='__main__':main()
