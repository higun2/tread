"""Synthetic in-memory training benchmark; never saves model weights."""
import argparse
import gc
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .attention_correction import corrected_attention
from .model import build_tread_sit
from .loss import FlowMatchingLoss


def timed(fn, steps):
    torch.cuda.synchronize()
    start=time.perf_counter()
    for _ in range(steps):fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-start)*1000/steps


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(8)
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result=dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),args=vars(args),
        caveat='Shared GPU: other jobs active; alternating repeated measurements, not isolated throughput. Synthetic GPU-resident data, no data loader/DDP/checkpointing/sampling/EMA update.',micro=[],training=[],correctness=[])
    def save(): (out/'results.json').write_text(json.dumps(result,indent=2))
    bias=math.log(127/255)
    def op(q,k,v,mode):
        return F.scaled_dot_product_attention(q,k,v) if mode=='baseline' else corrected_attention(q,k,v,bias,mode)
    # Validate forward and all Q/K/V gradients against a float32 explicit reference.
    for dtype in ([] if args.skip_micro else [torch.bfloat16,torch.float32]):
        qkv=[torch.randn(2,12,128,64,device=args.device,dtype=dtype,requires_grad=True) for _ in range(3)]
        ref=[x.detach().float().requires_grad_() for x in qkv]
        scores=ref[0]@ref[1].transpose(-1,-2)/8
        scores=scores+torch.eye(128,device=args.device)*bias
        target=scores.softmax(-1)@ref[2]
        probe=torch.randn_like(target)
        refgrads=torch.autograd.grad((target*probe).sum(),ref)
        for mode in ['sdpa','flex']:
            t0=time.perf_counter()
            actual=op(*qkv,mode)
            grads=torch.autograd.grad((actual.float()*probe).sum(),qkv)
            torch.cuda.synchronize()
            errors=[float((g.float()-r).norm()/r.norm()) for g,r in zip(grads,refgrads)]
            error=float((actual.float()-target).norm()/target.norm())
            assert error < (.02 if dtype==torch.bfloat16 else 1e-4),(mode,dtype,error)
            assert max(errors) < (.025 if dtype==torch.bfloat16 else 1e-4),(mode,dtype,errors)
            result['correctness'].append(dict(mode=mode,dtype=str(dtype),forward_relative_l2=error,gradient_relative_l2=errors,first_compile_and_check_seconds=time.perf_counter()-t0))
            print('correctness',result['correctness'][-1],flush=True);save()
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    torch._dynamo.reset()  # Separate tiny verification shapes from timing shapes.
    # Real QKV projection layout, including the FP32 endpoint used in this trainer.
    for dtype in ([] if args.skip_micro else [torch.bfloat16,torch.float32]):
        base=torch.randn(args.batch_size,128,3,12,64,device=args.device,dtype=dtype)
        inputs=[x.detach().requires_grad_() for x in base.permute(2,0,3,1,4).unbind(0)]
        def make(mode,backward):
            def step():
                for x in inputs:x.grad=None
                with torch.set_grad_enabled(backward):
                    y=op(*inputs,mode)
                    if backward:y.float().square().mean().backward()
            return step
        for backward in [False,True]:
            funcs={m:make(m,backward) for m in ['baseline','sdpa','flex']}
            for fn in funcs.values():timed(fn,args.warmup)
            for round_id in range(args.rounds):
                order=list(funcs);random.Random(round_id).shuffle(order)
                for mode in order:
                    ms=timed(funcs[mode],args.micro_steps)
                    row=dict(mode=mode,dtype=str(dtype),backward=backward,round=round_id,ms=ms)
                    result['micro'].append(row);print('micro',row,flush=True);save()
        for mode in ['baseline','sdpa']:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as p:
                make(mode,True)()
            result.setdefault('sdpa_operators',[]).append(dict(mode=mode,dtype=str(dtype),operators=[e.key for e in p.key_averages() if 'attention' in e.key]));save()
    torch._dynamo.reset()  # Separate microbenchmark graph variants from model graphs.
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)['model']
    models={};opts={}
    for mode in ['baseline','sdpa','flex']:
        model=build_tread_sit('SiT-B/2',input_size=32,num_classes=1000,class_dropout_prob=.1,
            use_cfg=True,qk_norm=False,fused_attn=True,path_type='linear',use_tread_routing=True,
            tread_start_block=3,tread_end_block=9,tread_active_ratio=.5,tread_fp32_endpoint=True,
            use_routesync=args.routesync,routesync_weight=.1,
            tread_attn_correction=mode!='baseline',tread_attn_correction_backend=mode if mode!='baseline' else 'flex').to(args.device).train()
        model.load_state_dict(state,strict=True)
        models[mode]=model
        opts[mode]=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=0.)
    criterion=FlowMatchingLoss('v','linear','uniform',use_routesync=args.routesync,routesync_weight=.1)
    x=torch.randn(args.batch_size,4,32,32,device=args.device)
    labels=torch.randint(0,1000,(args.batch_size,),device=args.device)
    def make_step(mode):
        model=models[mode];opt=opts[mode]
        def step():
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                losses=criterion(model,x,dict(y=labels))
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        return step
    funcs={m:make_step(m) for m in models}
    for mode,fn in funcs.items():
        secs=timed(fn,args.warmup)*args.warmup/1000
        result.setdefault('warmup_seconds',{})[mode]=secs
        print('full warmup',mode,secs,flush=True);save()
    for round_id in range(args.rounds):
        order=list(funcs);random.Random(round_id).shuffle(order)
        for mode in order:
            torch.cuda.reset_peak_memory_stats()
            allocated=torch.cuda.memory_allocated()
            ms=timed(funcs[mode],args.steps)
            row=dict(mode=mode,round=round_id,ms=ms,images_per_second=args.batch_size*1000/ms,
                incremental_peak_mb=(torch.cuda.max_memory_allocated()-allocated)/2**20)
            result['training'].append(row);print('training',row,flush=True);save()
    print('Complete',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',default='/v/mnt/GH/SiT/tread-v/checkpoints/0400000.pt')
    p.add_argument('--output',default='tread_routesync/correction_benchmark')
    p.add_argument('--device',default='cuda:0');p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--warmup',type=int,default=5);p.add_argument('--steps',type=int,default=15)
    p.add_argument('--micro-steps',type=int,default=50);p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--routesync',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--skip-micro',action='store_true',help='Only run full training-step timing')
    main(p.parse_args())
