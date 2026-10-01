"""Direct feature alignment formula, masking, teacher gradients and DDP."""
import argparse
import importlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

PACKAGES=('tread_routesync','tread_routesync2')


def make(package, recursive=False):
    cls=importlib.import_module(package+'.model').SiT
    model=cls(input_size=8,patch_size=2,hidden_size=32,decoder_hidden_size=32,
        depth=7,num_heads=4,num_classes=10,class_dropout_prob=0,
        qk_norm=False,fused_attn=True,use_tread_routing=True,
        tread_start_block=1,tread_end_block=4,tread_recursive=recursive,
        tread_num_groups=1,use_routesync=True,routesync_target_blocks=[4,5,6],
        routesync_loss_type='feature-cosine',tread_attn_correction=True,
        tread_attn_correction_backend='sdpa')
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module,'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight,std=.05)
        torch.nn.init.normal_(model.final_layer.linear.weight,std=.05)
    return model


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    try:
        for package in PACKAGES:
            model=DistributedDataParallel(make(package))
            loss=importlib.import_module(package+'.loss').FlowMatchingLoss(
                use_routesync=True,routesync_loss_type='feature-cosine',
                routesync_weight=.1,routesync_sample_ratio=.5)
            torch.manual_seed(204+rank)
            for _ in range(3):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    result=loss(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,2])})
                result['total'].backward()
                assert torch.isfinite(result['total'])
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
    finally:
        dist.destroy_process_group()


class FeatureCosineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def test_exact_loss_and_gradients_unequal_groups(self):
        torch.manual_seed(26)
        a=torch.randn(2,7,5,dtype=torch.float64,requires_grad=True)
        b=torch.randn_like(a,requires_grad=True)
        r=torch.tensor([[5,1]]).expand(2,-1)
        p=torch.tensor([[6,0,4,2,3]]).expand(2,-1)
        expected=-torch.nn.functional.cosine_similarity(a,b.detach(),dim=-1).mean()
        for package in PACKAGES:
            f=importlib.import_module(package+'.loss').routesync_feature_cosine_loss
            state=torch.get_rng_state()
            with patch('torch.bmm',side_effect=AssertionError('No relation matrix allowed')):
                actual,stats=f(a,b,r,p,debug=True)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            torch.testing.assert_close(actual,expected)
            grad,teacher=torch.autograd.grad(actual,(a,b),retain_graph=True,allow_unused=True)
            ref,=torch.autograd.grad(expected,(a,),retain_graph=True)
            torch.testing.assert_close(grad,ref)
            self.assertIsNone(teacher)
            self.assertEqual(stats['sampled_r_tokens'],2)

    def test_subset_correspondence_and_unselected_zero_gradients(self):
        for package in PACKAGES:
            mod=importlib.import_module(package+'.loss')
            torch.manual_seed(47)
            a=torch.randn(2,9,4,dtype=torch.float64,requires_grad=True)
            b=torch.randn_like(a,requires_grad=True)
            r=torch.tensor([[7,1,3]]).expand(2,-1)
            p=torch.tensor([[0,2,4,5,6,8]]).expand(2,-1)
            state=torch.get_rng_state()
            chosen=torch.cat([mod.sample_group_indices(r,.5),mod.sample_group_indices(p,.5)],1)
            torch.set_rng_state(state)
            actual,stats=mod.routesync_feature_cosine_loss(a,b,r,p,sample_ratio=.5)
            ar=mod.gather_tokens(a,chosen);br=mod.gather_tokens(b.detach(),chosen)
            expected=-torch.nn.functional.cosine_similarity(ar,br,dim=-1).mean()
            torch.testing.assert_close(actual,expected)
            grad,teacher=torch.autograd.grad(actual,(a,b),allow_unused=True)
            selected=torch.zeros(2,9,dtype=torch.bool).scatter_(1,chosen,True)
            self.assertEqual(grad[~selected].abs().sum().item(),0)
            self.assertIsNone(teacher)
            self.assertEqual((stats['sampled_r_tokens'],stats['sampled_p_tokens']),(1,3))

    def test_direction_sign_and_zero_features(self):
        for package in PACKAGES:
            f=importlib.import_module(package+'.loss').routesync_feature_cosine_loss
            a=torch.randn(2,4,8,requires_grad=True)
            r=torch.tensor([[2,0]]).expand(2,-1);p=torch.tensor([[3,1]]).expand(2,-1)
            for factor,expected in [(3.,-1.),(-2.,1.)]:
                loss,_=f(a,a.detach()*factor,r,p)
                self.assertAlmostEqual(loss.item(),expected,places=6)
            zero=torch.zeros_like(a,requires_grad=True)
            loss,_=f(zero,zero,r,p);loss.backward()
            self.assertTrue(torch.isfinite(zero.grad).all())

    def test_mixed_precision_integration_and_prefix_only_auxiliary_gradient(self):
        for package in PACKAGES:
            for recursive in [False,True]:
                model=make(package,recursive)
                fn=importlib.import_module(package+'.loss').FlowMatchingLoss(
                    use_routesync=True,routesync_loss_type='feature-cosine',routesync_weight=.2,
                    routesync_sample_ratio=.5,routesync_debug=True)
                calls=[];hook=model.register_forward_hook(lambda *args:calls.append(1))
                with torch.autocast('cpu',dtype=torch.bfloat16):
                    result=fn(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,2])})
                hook.remove();self.assertEqual(len(calls),1)
                torch.testing.assert_close(result['total'],result['fm']+.2*result['route_sync'])
                self.assertEqual(result['route_sync'].dtype,torch.float32)
                post=list(model.suffix_blocks.parameters())+list(model.recurrent_group_blocks.parameters()) if recursive else list(model.blocks[1:].parameters())
                aux=torch.autograd.grad(result['route_sync'],post,retain_graph=True,allow_unused=True)
                self.assertTrue(all(g is None for g in aux))
                result['total'].backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_cli_defaults_and_invalid_mode(self):
        for package in PACKAGES:
            cfg=importlib.import_module(package+'.config')
            parser=argparse.ArgumentParser();cfg.add_routesync_args(parser)
            self.assertEqual(parser.parse_args([]).routesync_loss_type,'relational')
            args=parser.parse_args(['--routesync-loss-type','feature-cosine'])
            self.assertEqual(cfg.tread_kwargs(args)['routesync_loss_type'],'feature-cosine')
            cls=importlib.import_module(package+'.loss').FlowMatchingLoss
            with self.assertRaises(ValueError):cls(routesync_loss_type='unknown')

    def test_two_rank_bf16_backward(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'store'),),nprocs=2,start_method='spawn',join=True)


if __name__=='__main__':unittest.main()
