"""Dense subset teacher: correspondence, no-grad, segment cost and DDP."""
import argparse
import importlib
import itertools
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

PACKAGES = ('tread_routesync', 'tread_routesync2')


def make(package, recursive=False, enabled=True, weight=.1, routesync=False):
    model = importlib.import_module(package+'.model').SiT(
        input_size=8, patch_size=2, hidden_size=32, decoder_hidden_size=32,
        depth=5, num_heads=4, num_classes=10, class_dropout_prob=.2,
        qk_norm=False, fused_attn=True, use_tread_routing=True,
        tread_start_block=1, tread_end_block=3, tread_recursive=recursive,
        tread_num_groups=1, use_routesync=routesync,
        use_dense_sparse_sync=enabled, dense_sparse_sync_weight=weight,
        tread_attn_correction=True)
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, 'adaLN_modulation'):
                torch.nn.init.normal_(module.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(model.final_layer.linear.weight, std=.1)
    return model


def worker(rank, store):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://'+store, rank=rank, world_size=2)
    try:
        for package, recursive in itertools.product(PACKAGES, (False, True)):
            model = DistributedDataParallel(make(package, recursive=recursive, routesync=True))
            criterion = importlib.import_module(package+'.loss').FlowMatchingLoss(
                use_routesync=True, routesync_weight=.1,
                use_dense_sparse_sync=True, dense_sparse_sync_weight=.1)
            for _ in range(2):
                model.zero_grad(set_to_none=True)
                with torch.autocast('cpu', dtype=torch.bfloat16):
                    result = criterion(model, torch.randn(3,4,8,8), {'y':torch.tensor([1,2,3])})
                result['total'].backward()
                assert torch.isfinite(result['total'])
                assert all(p.grad is not None and torch.isfinite(p.grad).all()
                           for p in model.parameters() if p.requires_grad)
    finally:
        dist.destroy_process_group()


class DenseSparseSyncTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_teacher_correspondence_cost_and_gradient(self):
        for package in PACKAGES:
            for recursive in (False, True):
                model = make(package, recursive)
                calls=[]; prefix=[]
                blocks = list(model.recurrent_group_blocks) if recursive else list(model.blocks[1:3])
                hooks = [b.register_forward_hook(lambda m,a,o: calls.append((a[0].shape, torch.is_grad_enabled()))) for b in blocks]
                first = model.prefix_blocks[0] if recursive else model.blocks[0]
                hooks.append(first.register_forward_hook(lambda m,a,o: prefix.append((o.detach(), a[1].detach()))))
                _, features, info = model(torch.randn(20,4,8,8),torch.rand(20),torch.arange(20)%10,return_routing_info=True)
                for h in hooks: h.remove()
                f=features['dense_sparse_sync']
                self.assertEqual(len(prefix),1)
                self.assertEqual(len(calls),4)
                self.assertEqual([c[0][:2] for c in calls],[(2,16),(2,16),(20,8),(20,8)])
                self.assertEqual([c[1] for c in calls],[False,False,True,True])
                self.assertFalse(f['teacher'].requires_grad)
                self.assertEqual(f['sample_indices'].unique().numel(),2)
                # Independently recompute exact dense target from saved prefix/CFG condition.
                with torch.no_grad():
                    h=prefix[0][0][f['sample_indices']]; c=prefix[0][1][f['sample_indices']]
                    for i in range(2):
                        b=blocks[0] if recursive else blocks[i]
                        h=model._run_routed_block(b,h,c,endpoint=i==1)
                    indices=info['active_indices'][f['sample_indices']]
                    expected=importlib.import_module(package+'.model').gather_tokens(h,indices)
                torch.testing.assert_close(f['teacher'],expected)
                loss=importlib.import_module(package+'.loss').dense_sparse_cosine_loss(f['student'],f['teacher'])
                loss.backward()
                self.assertGreater(sum(p.grad.abs().sum().item() for b in blocks for p in b.parameters() if p.grad is not None),0)
                suffix=model.suffix_blocks if recursive else model.blocks[3:]
                self.assertTrue(all(p.grad is None for p in suffix.parameters()))

    def test_formula_and_teacher_stop(self):
        for package in PACKAGES:
            s=torch.randn(2,5,8,requires_grad=True);t=torch.randn_like(s,requires_grad=True)
            fn=importlib.import_module(package+'.loss').dense_sparse_cosine_loss
            loss=fn(s,t)
            torch.testing.assert_close(loss,-torch.nn.functional.cosine_similarity(s,t.detach(),dim=-1).mean())
            loss.backward();self.assertIsNone(t.grad)
            self.assertAlmostEqual(fn(s,3*s).item(),-1,places=6)

    def test_disabled_zero_weight_and_eval_exact(self):
        for package in PACKAGES:
            base=make(package,enabled=False);enabled=make(package);zero=make(package,weight=0)
            enabled.load_state_dict(base.state_dict());zero.load_state_dict(base.state_dict())
            x=torch.randn(3,4,8,8);t=torch.rand(3);y=torch.tensor([1,2,3])
            state=torch.get_rng_state()
            a,_=base(x,t,y);end=torch.get_rng_state()
            torch.set_rng_state(state);b,f=zero(x,t,y)
            torch.testing.assert_close(a,b,rtol=0,atol=0)
            self.assertTrue(torch.equal(end,torch.get_rng_state()));self.assertIsNone(f)
            base.eval();enabled.eval()
            for mode in ('dense','sparse'):
                torch.set_rng_state(state);a,_=base(x,t,y,tread_eval_mode=mode)
                torch.set_rng_state(state);b,f=enabled(x,t,y,tread_eval_mode=mode)
                torch.testing.assert_close(a,b,rtol=0,atol=0);self.assertIsNone(f)

    def test_combined_bf16_loss_and_cli(self):
        for package in PACKAGES:
            cfg=importlib.import_module(package+'.config');p=argparse.ArgumentParser();cfg.add_routesync_args(p)
            args=p.parse_args(['--use-dense-sparse-sync'])
            self.assertEqual(args.dense_sparse_sync_ratio,.1)
            self.assertTrue(cfg.tread_kwargs(args)['use_dense_sparse_sync'])
            criterion=importlib.import_module(package+'.loss').FlowMatchingLoss(
                use_routesync=True,routesync_weight=.2,use_dense_sparse_sync=True,dense_sparse_sync_weight=.3)
            model=make(package,routesync=True,weight=.3)
            with torch.autocast('cpu',dtype=torch.bfloat16):
                result=criterion(model,torch.randn(3,4,8,8),{'y':torch.tensor([1,2,3])})
            torch.testing.assert_close(result['total'],result['fm']+.2*result['route_sync']+.3*result['dense_sparse_sync'])
            self.assertEqual(result['dense_sparse_sync_samples'],1)
            self.assertEqual(result['dense_sparse_sync'].dtype,torch.float32)
            result['total'].backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad))

    def test_two_rank_bf16(self):
        with tempfile.TemporaryDirectory() as folder:
            mp.start_processes(worker,args=(str(Path(folder)/'store'),),nprocs=2,start_method='spawn',join=True)


if __name__=='__main__': unittest.main()
