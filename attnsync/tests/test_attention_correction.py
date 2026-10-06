import copy
import unittest

import torch

from attnsync.attention_correction import corrected_attention
from attnsync.model import SiT


class AttentionCorrectionTests(unittest.TestCase):
    def test_formula_and_gradients(self):
        torch.manual_seed(5)
        tensors=[torch.randn(2,3,8,4,dtype=torch.float64,requires_grad=True) for _ in range(3)]
        q,k,v=tensors
        bias=-.7
        actual=corrected_attention(q,k,v,bias,'sdpa')
        scores=q@k.transpose(-1,-2)/2
        scores=scores+torch.eye(8,dtype=q.dtype)*bias
        expected=scores.softmax(-1)@v
        torch.testing.assert_close(actual,expected)
        probe=torch.randn_like(actual)
        ga=torch.autograd.grad((actual*probe).sum(),tensors,retain_graph=True)
        ge=torch.autograd.grad((expected*probe).sum(),tensors)
        for a,e in zip(ga,ge):torch.testing.assert_close(a,e)

    def test_model_dense_identity_checkpoint_and_sparse_change(self):
        torch.set_num_threads(2)
        for package in ['attnsync']:
            import importlib
            cls=importlib.import_module(package+'.model').SiT
            for recursive in [False,True]:
                kw=dict(input_size=8,patch_size=2,hidden_size=32,decoder_hidden_size=32,
                    depth=5,num_heads=4,num_classes=10,class_dropout_prob=0,
                    qk_norm=False,fused_attn=True,use_tread_routing=True,
                    tread_start_block=1,tread_end_block=3,tread_recursive=recursive,
                    tread_num_groups=1,tread_seed=9,tread_attn_correction_backend='sdpa')
                base=cls(**kw).eval()
                with torch.no_grad():
                    for m in base.modules():
                        if hasattr(m,'adaLN_modulation'):
                            torch.nn.init.normal_(m.adaLN_modulation[-1].weight,std=.1)
                    torch.nn.init.normal_(base.final_layer.linear.weight,std=.1)
                corr=cls(**kw,tread_attn_correction=True).eval()
                corr.load_state_dict(base.state_dict(),strict=True)
                zero=cls(**kw,tread_attn_correction=True,tread_attn_correction_strength=0).eval()
                zero.load_state_dict(base.state_dict(),strict=True)
                x=torch.randn(2,4,8,8);t=torch.tensor([.2,.7]);y=torch.tensor([1,2])
                with torch.no_grad():
                    bd=base(x,t,y,tread_eval_mode='dense')[0]
                    cd=corr(x,t,y,tread_eval_mode='dense')[0]
                    torch.testing.assert_close(bd,cd,rtol=0,atol=0)
                    bs=base(x,t,y)[0];cs=corr(x,t,y)[0]
                    self.assertGreater((bs-cs).abs().max().item(),1e-7)
                    torch.testing.assert_close(bs,zero(x,t,y)[0],rtol=0,atol=0)
                    torch.testing.assert_close(cs,copy.deepcopy(corr)(x,t,y)[0],rtol=0,atol=0)
                corr.train()
                corr(x,t,y)[0].square().mean().backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in corr.parameters() if p.requires_grad))

    def test_one_token_and_dense_no_bias(self):
        q,k,v=[torch.randn(2,3,1,4) for _ in range(3)]
        torch.testing.assert_close(corrected_attention(q,k,v,-.7,'sdpa'),v)


if __name__=='__main__':unittest.main()
