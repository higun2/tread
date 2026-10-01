"""Target identity, stopped gradients, RNG compatibility and real DDP sampling."""
import argparse
import importlib
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

PACKAGES=('tread_routesync','tread_routesync2')


def make(package, targets=None, recursive=False):
    cls=importlib.import_module(package+'.model').SiT
    model=cls(input_size=8,patch_size=2,hidden_size=32,decoder_hidden_size=32,
        depth=7,num_heads=4,num_classes=10,class_dropout_prob=0,
        qk_norm=False,fused_attn=True,use_tread_routing=True,
        tread_start_block=1,tread_end_block=4,tread_recursive=recursive,
        tread_num_groups=1,use_routesync=True,routesync_target_blocks=targets)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m,'adaLN_modulation'):
                torch.nn.init.normal_(m.adaLN_modulation[-1].weight,std=.05)
        torch.nn.init.normal_(model.final_layer.linear.weight,std=.05)
    return model


def inputs():
    return torch.randn(2,4,8,8),torch.tensor([.2,.7]),torch.tensor([1,2])


def worker(rank,store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2)
    try:
        for package in PACKAGES:
            model=DDP(make(package,[4,5,6]))
            torch.manual_seed(903+rank)
            criterion=importlib.import_module(package+'.loss').FlowMatchingLoss(use_routesync=True,routesync_weight=.1)
            x,_,y=inputs()
            seen=set()
            for _ in range(8):
                model.zero_grad(set_to_none=True)
                result=criterion(model,x,{'y':y})
                result['total'].backward()
                chosen=model.module.last_routesync_target_block
                seen.add(chosen)
                local=torch.tensor([chosen]); gathered=[torch.empty_like(local) for _ in range(2)]
                dist.all_gather(gathered,local)
                assert gathered[0].item()==gathered[1].item()
                assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            assert len(seen)>1
    finally:
        dist.destroy_process_group()


class TargetBlockTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(2)

    def test_cli_validation(self):
        for package in PACKAGES:
            c=importlib.import_module(package+'.config')
            parser=argparse.ArgumentParser();c.add_routesync_args(parser)
            self.assertEqual(parser.parse_args(['--routesync-target-blocks','4','5','6']).routesync_target_blocks,[4,5,6])
            self.assertIsNone(parser.parse_args([]).routesync_target_blocks)
            for bad in [[],[3],[7],[4,4],[4.5]]:
                with self.assertRaises(ValueError):make(package,bad)

    def test_exact_target_and_stop_gradient(self):
        for package in PACKAGES:
            relation=importlib.import_module(package+'.loss').routesync_relational_loss
            for recursive in [False,True]:
                for target in [4,5,6]:
                    model=make(package,[target],recursive)
                    block=model.suffix_blocks[target-4] if recursive else model.blocks[target]
                    captured=[]
                    handle=block.register_forward_hook(lambda module,args,result:captured.append(result))
                    _,features=model(*inputs())
                    handle.remove()
                    self.assertIs(features['h_post'],captured[0])
                    self.assertEqual(features['target_block'],target)
                    hpre,hpost=features['h_pre'],features['h_post']
                    loss,_=relation(hpre,hpost,features['routed_idx'],features['processed_idx'])
                    grads=torch.autograd.grad(loss,(hpre,hpost),allow_unused=True)
                    self.assertGreater(grads[0].abs().sum().item(),0)
                    self.assertIsNone(grads[1])

    def test_default_singleton_exact_rng_and_checkpoint_compatibility(self):
        for package in PACKAGES:
            base=make(package);explicit=make(package,[4])
            explicit.load_state_dict(base.state_dict(),strict=True)
            args=inputs();rng=torch.get_rng_state()
            a,af=base(*args);end=torch.get_rng_state()
            torch.set_rng_state(rng)
            b,bf=explicit(*args)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            torch.testing.assert_close(af['h_post'],bf['h_post'],rtol=0,atol=0)
            self.assertTrue(torch.equal(end,torch.get_rng_state()))

    def test_random_targets_reproducible_and_uniform(self):
        for package in PACKAGES:
            model=make(package,[4,5,6])
            torch.manual_seed(71);state=torch.get_rng_state()
            selected=[model._routesync_target_for_forward(torch.device('cpu')) for _ in range(900)]
            torch.set_rng_state(state)
            self.assertEqual(selected,[model._routesync_target_for_forward(torch.device('cpu')) for _ in range(900)])
            for block in [4,5,6]:self.assertTrue(240<selected.count(block)<360)
            # Each sampled feature must actually be from the announced layer.
            outputs={}
            handles=[model.blocks[i].register_forward_hook(lambda m,a,o,i=i:outputs.__setitem__(i,o)) for i in [4,5,6]]
            for _ in range(12):
                _,f=model(*inputs())
                self.assertIs(f['h_post'],outputs[f['target_block']])
            for h in handles:h.remove()

    def test_eval_does_not_select_targets(self):
        for package in PACKAGES:
            model=make(package,[4,5,6]).eval();model.tread_seed=5
            args=inputs()
            for mode in ['dense','sparse']:
                state=torch.get_rng_state()
                with torch.no_grad():_,features=model(*args,tread_eval_mode=mode)
                self.assertIsNone(features)
                self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_two_rank_selection_and_backward(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'store'),),nprocs=2,start_method='spawn',join=True)


if __name__=='__main__':unittest.main()
