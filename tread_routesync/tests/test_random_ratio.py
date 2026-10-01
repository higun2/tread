"""Random-ratio training, inference isolation, resume RNG and rank synchronization."""
import argparse
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from ..config import add_tread_args, tread_kwargs
from ..loss import FlowMatchingLoss
from ..model import SiT


def make_model(**overrides):
    kw = dict(input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=7, num_heads=4, mlp_ratio=2, num_classes=10, class_dropout_prob=0,
        qk_norm=False, fused_attn=False, use_tread_routing=True,
        tread_start_block=1, tread_end_block=5, tread_active_ratio=.5,
        tread_active_ratios=[.3,.5,.7], use_routesync=True,
        tread_num_groups=2)
    kw.update(overrides)
    model=SiT(**kw)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m,'adaLN_modulation'):
                nn.init.normal_(m.adaLN_modulation[-1].weight,std=.02)
                nn.init.normal_(m.adaLN_modulation[-1].bias,std=.02)
        nn.init.normal_(model.final_layer.linear.weight,std=.02)
    return model


def ddp_worker(rank,store):
    torch.set_num_threads(1)
    os.environ.setdefault('GLOO_SOCKET_IFNAME','lo')
    dist.init_process_group('gloo',init_method=f'file://{store}',rank=rank,world_size=2)
    try:
        torch.manual_seed(12)
        model=make_model(tread_recursive=True).train()
        wrapped=DistributedDataParallel(model)
        criterion=FlowMatchingLoss(use_routesync=True,routesync_weight=.1,routesync_sample_ratio=.5)
        torch.manual_seed(200+rank)
        for _ in range(4):
            wrapped.zero_grad(set_to_none=True)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                result=criterion(wrapped,torch.randn(2,4,8,8),{'y':torch.tensor([1,3])})
            result['total'].backward()
            value=torch.tensor(model.last_tread_active_ratio)
            gathered=[torch.zeros_like(value) for _ in range(2)]
            dist.all_gather(gathered,value)
            assert torch.equal(gathered[0],gathered[1])
            for name,p in wrapped.named_parameters():
                if p.requires_grad:
                    assert p.grad is not None and torch.isfinite(p.grad).all(),name
    finally:
        dist.destroy_process_group()


class RandomRatioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads=torch.get_num_threads();torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):torch.set_num_threads(cls.threads)

    def test_sampling_rng_replay_and_disabled_path(self):
        model=make_model().train()
        state=torch.get_rng_state()
        first=[model._active_ratio_for_forward('cpu') for _ in range(120)]
        torch.set_rng_state(state)
        second=[model._active_ratio_for_forward('cpu') for _ in range(120)]
        self.assertEqual(first,second)
        self.assertEqual(set(first),{.3,.5,.7})
        model.tread_active_ratios=None
        state=torch.get_rng_state()
        self.assertEqual(model._active_ratio_for_forward('cpu'),.5)
        self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_ratio_shapes_subset_loss_and_recursive_bf16_backward(self):
        for recursive in (False,True):
            model=make_model(tread_recursive=recursive).train()
            for ratio,count in ((.3,5),(.5,8),(.7,11)):
                model.zero_grad(set_to_none=True)
                criterion=FlowMatchingLoss(use_routesync=True,routesync_weight=.1,routesync_sample_ratio=.5)
                with patch.object(model,'_active_ratio_for_forward',return_value=ratio) as choose:
                    with torch.autocast('cpu',dtype=torch.bfloat16):
                        result=criterion(model,torch.randn(2,4,8,8),{'y':torch.tensor([1,3])})
                choose.assert_called_once()
                self.assertEqual(model.last_tread_active_ratio,ratio)
                self.assertEqual(model.last_tread_active_fraction,count/16)
                self.assertEqual(result['sampled_p_tokens'],max(1,count//2))
                self.assertEqual(result['sampled_r_tokens'],max(1,(16-count)//2))
                result['total'].backward()
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        self.assertIsNotNone(p.grad,name)
                        self.assertTrue(torch.isfinite(p.grad).all(),name)

    def test_eval_never_samples_ratio_and_matches_fixed_model(self):
        model=make_model(tread_seed=42).eval()
        x=torch.randn(2,4,8,8);t=torch.tensor([.2,.7]);y=torch.tensor([1,3])
        for mode in ('sparse','dense'):
            model.tread_active_ratios=(.3,.5,.7)
            state=torch.get_rng_state()
            with torch.no_grad():
                random_config=model(x,t,y,tread_eval_mode=mode,return_routing_info=True)
            self.assertTrue(torch.equal(state,torch.get_rng_state()))
            model.tread_active_ratios=None
            with torch.no_grad():fixed=model(x,t,y,tread_eval_mode=mode,return_routing_info=True)
            torch.testing.assert_close(random_config[0],fixed[0],rtol=0,atol=0)
            if mode=='sparse':self.assertEqual(random_config[2]['active_indices'].shape,(2,8))

    def test_config_and_validation(self):
        parser=argparse.ArgumentParser();add_tread_args(parser)
        args=parser.parse_args(['--tread-active-ratios','0.3','0.5','0.7'])
        self.assertEqual(args.tread_active_ratios,[.3,.5,.7])
        self.assertIsNone(tread_kwargs(SimpleNamespace())['tread_active_ratios'])
        for invalid in ([],[0],[1],[-.2],[float('nan')],[float('inf')],[.5,.5]):
            with self.assertRaises(ValueError):make_model(tread_active_ratios=invalid)

    def test_two_rank_ratio_sync_and_backward(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.start_processes(ddp_worker,args=(str(Path(directory)/'store'),),
                nprocs=2,join=True,start_method='spawn')


if __name__=='__main__':unittest.main()
