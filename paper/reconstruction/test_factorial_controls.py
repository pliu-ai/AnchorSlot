import copy, json, os, sys
from pathlib import Path
import numpy as np
import pytest
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.network_structured_conditional import StructuredConditionalUNet
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic_group_head import attach_group_head, group_head_loss, build_group_lut
sys.path.insert(0, str(Path(__file__).parent))
import decode_factorial_controls as dec

def test_no_head_has_same_structured_forward_and_zero_auxiliary_gradient():
    torch.manual_seed(10)
    backbone=PlainConvUNet(1,3,(8,16,32),nn.Conv3d,3,(1,2,2),(1,1,1),9,(1,1),False,nn.InstanceNorm3d,{'eps':1e-5,'affine':True},None,None,nn.LeakyReLU,{'inplace':True},deep_supervision=True)
    base=StructuredConditionalUNet(backbone,13,9)
    headed=attach_group_head(copy.deepcopy(base))
    assert not any('group_head' in k for k in base.state_dict())
    x=torch.randn(2,1,8,16,16); ids=torch.tensor([0,3]); target=torch.randint(0,32,(2,1,8,16,16))
    out=base(x,ids); pack=headed(x,ids,return_group_logits=True)
    for a,b in zip(out,pack['logits']): torch.testing.assert_close(a,b)
    loss=sum(t.square().mean() for t in out)
    other=sum(t.square().mean() for t in pack['logits'])+0*group_head_loss(pack['group_logits'],target,None,build_group_lut())
    loss.backward(); other.backward()
    bparams=dict(headed.named_parameters())
    for k,p in base.named_parameters():
        if p.grad is None: assert bparams[k].grad is None
        else: torch.testing.assert_close(p.grad,bparams[k].grad)
    assert headed.group_head.weight.grad.count_nonzero()==0

def test_head_free_dump_does_not_read_coarse_channels():
    meta=json.loads(dec.dl.DEFAULT_DUMP_META.read_text()); compact=[c for c in meta if c['kind']!='group_head']
    arr=np.random.default_rng(2).random((len(compact),2,3,4)).astype(np.float32)
    s,_=dec.as_scores(arr,compact,0,None)
    np.testing.assert_array_equal(s,dec.dl.legacy_scores(arr,compact))

def test_dense_weights_and_invalid_variants():
    for w in (0,.25,.5,.75,1):
        assert dec.parse(f'G{w}')==(w,None)
        assert dec.parse(f'HG{w}a3')==(w,3.)
    for v in ('G1.1','HG0.5a0.5','HG1','G0.5a3'):
        with pytest.raises(ValueError): dec.parse(v)
