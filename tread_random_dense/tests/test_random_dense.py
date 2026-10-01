"""Temporary dense routed block, exact topology and DDP synchronization."""
import argparse
import tempfile
from pathlib import Path
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from tread_random_dense.config import add_tread_args, tread_kwargs
from tread_random_dense.loss import FlowMatchingLoss
from tread_random_dense.model import SiT, gather_tokens, restore_token_order
from tread_routesync.model import SiT as OriginalSiT


def make(enabled=True, candidates=None, recursive=False, **extra):
    model = SiT(input_size=8,patch_size=2,hidden_size=32,decoder_hidden_size=32,
                depth=6,num_heads=4,num_classes=10,class_dropout_prob=0,
                qk_norm=False,fused_attn=True,use_tread_routing=True,
                tread_start_block=1,tread_end_block=4,tread_active_ratio=.5,
                tread_recursive=recursive,tread_num_groups=1,
                tread_random_dense=enabled,tread_random_dense_blocks=candidates,
                **extra)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module,'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight,std=.05)
        torch.nn.init.normal_(model.final_layer.linear.weight,std=.05)
    return model


def manual(model, x, t, y, active_idx, bypass_idx, selected):
    x=model.x_embedder(x)+model.pos_embed
    c=model.t_embedder(t)+model.y_embedder(y,model.training)
    prefix=model.prefix_blocks if model.tread_recursive else model.blocks[:model.tread_start_block]
    suffix=model.suffix_blocks if model.tread_recursive else model.blocks[model.tread_end_block:]
    for block in prefix:x=block(x,c)
    active=gather_tokens(x,active_idx);bypass=gather_tokens(x,bypass_idx)
    permutation=torch.cat((active_idx,bypass_idx),1)
    for i in range(model.tread_start_block,model.tread_end_block):
        offset=i-model.tread_start_block
        block=model._recursive_block_for_offset(offset) if model.tread_recursive else model.blocks[i]
        condition=model._condition_for_logical_depth(c,offset)
        if i==selected:
            full=restore_token_order(torch.cat((active,bypass),1),permutation)
            full=model._run_routed_block(block,full,condition,endpoint=i==model.tread_end_block-1)
            active=gather_tokens(full,active_idx);bypass=gather_tokens(full,bypass_idx)
        else:
            active=model._run_routed_block(block,active,condition,endpoint=i==model.tread_end_block-1)
    x=restore_token_order(torch.cat((active,bypass),1),permutation)
    for block in suffix:x=block(x,c)
    return model.unpatchify(model.final_layer(x,c))


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    try:
        for recursive in (False,True):
            model=DistributedDataParallel(make(recursive=recursive))
            criterion=FlowMatchingLoss()
            choices=[]
            for step in range(4):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    result=criterion(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,2])})
                choice=model.module.last_tread_random_dense_block
                other=[None,None]
                dist.all_gather_object(other,choice)
                assert other==[choice,choice]
                choices.append(choice)
                result['total'].backward()
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.requires_grad)
            assert all(1<=c<4 for c in choices)
    finally:dist.destroy_process_group()


class RandomDenseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_each_block_full_once_then_original_mask_restored(self):
        torch.manual_seed(3)
        x=torch.randn(2,4,8,8);t=torch.rand(2);y=torch.tensor([1,2])
        for recursive in (False,True):
            for selected in (1,2,3):
                model=make(candidates=[selected],recursive=recursive)
                routed=(model.recurrent_group_blocks if recursive else model.blocks[1:4])
                calls=[]
                hooks=[b.register_forward_hook(lambda m,a,o:calls.append(a[0].shape[1])) for b in routed]
                state=torch.get_rng_state()
                output,features,info=model(x,t,y,return_routing_info=True)
                for hook in hooks:hook.remove()
                self.assertEqual(info['random_dense_block'],selected)
                self.assertEqual(model.last_tread_random_dense_block,selected)
                self.assertEqual(calls,[16 if i==selected else 8 for i in (1,2,3)])
                self.assertEqual(info['block_input_token_counts'][1:4],calls)
                self.assertIsNone(features)
                # Reuse the actual per-image mask, not an independently sampled one.
                torch.set_rng_state(state)
                reference=manual(model,x,t,y,info['active_indices'],info['bypass_indices'],selected)
                torch.testing.assert_close(output,reference,rtol=0,atol=0)

    def test_disabled_and_eval_are_exact_original(self):
        original=OriginalSiT(input_size=8,patch_size=2,hidden_size=32,
            decoder_hidden_size=32,depth=6,num_heads=4,num_classes=10,
            class_dropout_prob=0,qk_norm=False,fused_attn=True,
            use_tread_routing=True,tread_start_block=1,tread_end_block=4,
            tread_active_ratio=.5)
        off=make(enabled=False);original.load_state_dict(off.state_dict(),strict=True)
        on=make(enabled=True);on.load_state_dict(off.state_dict(),strict=True)
        x=torch.randn(2,4,8,8);t=torch.rand(2);y=torch.tensor([1,2])
        state=torch.get_rng_state();a,_=original(x,t,y);after=torch.get_rng_state()
        torch.set_rng_state(state);b,_=off(x,t,y)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(after,torch.get_rng_state()))
        self.assertEqual(set(on.state_dict()),set(off.state_dict()))
        original.eval();on.eval()
        for mode in ('dense','sparse'):
            torch.set_rng_state(state);a,_=original(x,t,y,tread_eval_mode=mode)
            torch.set_rng_state(state);b,_=on(x,t,y,tread_eval_mode=mode)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertIsNone(on.last_tread_random_dense_block)

    def test_single_candidate_no_selection_rng_and_validation(self):
        parser=argparse.ArgumentParser();add_tread_args(parser)
        self.assertFalse(parser.parse_args([]).tread_random_dense)
        args=parser.parse_args(['--tread-random-dense','--tread-random-dense-blocks','2','3'])
        self.assertEqual(tread_kwargs(args)['tread_random_dense_blocks'],[2,3])
        model=make(candidates=[2]);state=torch.get_rng_state()
        self.assertEqual(model._random_dense_block_for_forward(torch.device('cpu')),2)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        for choices in ([],[0],[4],[1,1]):
            with self.assertRaises(ValueError):make(candidates=choices)
        with self.assertRaises(ValueError):make(enabled=False,candidates=[2])
        with self.assertRaises(ValueError):SiT(input_size=8,patch_size=2,hidden_size=32,
            decoder_hidden_size=32,depth=6,num_heads=4,num_classes=10,
            use_tread_routing=True,tread_start_block=1,tread_end_block=4,
            tread_active_ratio=1.,tread_random_dense=True,qk_norm=False,fused_attn=True)
        with self.assertRaises(ValueError):SiT(input_size=8,patch_size=2,hidden_size=32,
            decoder_hidden_size=32,depth=6,num_heads=4,num_classes=10,
            use_tread_routing=False,tread_random_dense=True,qk_norm=False,fused_attn=True)

    def test_all_candidates_sampled_and_reproducible(self):
        model=make()
        torch.manual_seed(206)
        first=[model._random_dense_block_for_forward(torch.device('cpu')) for _ in range(300)]
        torch.manual_seed(206)
        second=[model._random_dense_block_for_forward(torch.device('cpu')) for _ in range(300)]
        self.assertEqual(first,second)
        self.assertEqual(set(first),{1,2,3})
        self.assertTrue(all(70<first.count(i)<130 for i in (1,2,3)))

    def test_route_sync_and_bf16(self):
        model=make(use_routesync=True,routesync_weight=.1)
        criterion=FlowMatchingLoss(use_routesync=True,routesync_weight=.1)
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=criterion(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,2])})
        torch.testing.assert_close(result['total'],result['fm']+.1*result['route_sync'])
        result['total'].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters() if p.requires_grad))

    def test_recursive_grouped_depth_embedding(self):
        model=SiT(input_size=8,patch_size=2,hidden_size=32,
            decoder_hidden_size=32,depth=8,num_heads=4,num_classes=10,
            class_dropout_prob=0,qk_norm=False,fused_attn=True,
            use_tread_routing=True,tread_start_block=1,tread_end_block=5,
            tread_active_ratio=.5,tread_recursive=True,tread_num_groups=2,
            tread_recursive_pattern='grouped',tread_depth_embedding=True,
            tread_random_dense=True,tread_random_dense_blocks=[2])
        with torch.autocast('cpu',dtype=torch.bfloat16):
            result=FlowMatchingLoss()(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,2])})
        result['total'].backward()
        self.assertEqual(model.last_tread_random_dense_block,2)
        self.assertIsNotNone(model.tread_depth_embeddings.grad)
        self.assertTrue(torch.isfinite(model.tread_depth_embeddings.grad).all())
        self.assertEqual(len(model.recurrent_group_blocks),2)

    def test_two_rank_bf16(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'store'),),nprocs=2,start_method='spawn',join=True)


if __name__=='__main__':unittest.main()
