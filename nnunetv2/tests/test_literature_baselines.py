import json
import unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from literature_baselines.models import TASKS, UniSegCellMap, CoNeMOSCellMap
from literature_baselines.losses import task_loss, binary_loss, miso_loss, remap_task, deep_loss
from literature_baselines.prediction import merge_candidates, restore_grid


def backbone(channels):
    return PlainConvUNet(1, 3, [4, 8, 16], nn.Conv3d, [3]*3, [1, 2, 2], [2]*3,
                        channels, [2, 2], True, nn.InstanceNorm3d, {'affine': True},
                        None, None, nn.LeakyReLU, {'inplace': True}, True)


class BaselineTests(unittest.TestCase):
    def test_label_partition_and_context_remap(self):
        self.assertEqual(sorted(c for g in TASKS for c in g), list(range(1, 32)))
        target = torch.arange(-1, 32).view(1, 1, 1, 1, 33)
        mapped, valid, active = remap_task(target, torch.tensor([13]), torch.ones(1,32,dtype=torch.bool))
        self.assertFalse(valid[0,0,0,0])
        for slot, c in enumerate((1,2,3,27),1): self.assertEqual(mapped[0,0,0,c+1], slot)
        self.assertEqual(mapped[0,0,0,5], 0)  # mitochondrial membrane is context-task complement
        self.assertEqual(int(active.sum()), 5)

    def test_uniseg_inactive_channels_and_invalid_voxels_have_no_gradient(self):
        z = torch.randn(1,5,2,2,2,requires_grad=True)
        y = torch.tensor([7,8,0,-1,4,27,7,0]).view(1,1,2,2,2)
        annotation = torch.ones(1,32,dtype=torch.bool)
        loss = task_loss(z,y,torch.tensor([1]),annotation)
        changed = z.detach().clone(); changed[:,3:]=1000
        self.assertTrue(torch.allclose(loss.detach(),task_loss(changed,y,torch.tensor([1]),annotation)))
        loss.backward()
        self.assertEqual(float(z.grad[:,3:].abs().sum()), 0)
        self.assertEqual(float(z.grad[0,:,0,1,1].abs().sum()), 0)

    def test_partial_annotation_and_padding_masking(self):
        annotation = torch.ones(1,32,dtype=torch.bool); annotation[:,[0,31]]=False
        y = torch.tensor([1,0,-1,31,4,2,1,3]).view(1,1,2,2,2)
        for method,channels in [('binary',1),('miso',32)]:
            z=torch.randn(1,channels,2,2,2,requires_grad=True)
            loss=binary_loss(z,y,torch.tensor([0]),annotation) if method=='binary' else miso_loss(z,y,annotation)
            loss.backward()
            flat=z.grad.flatten(2)
            self.assertEqual(float(flat[:,:,[1,2,3]].abs().sum()),0)
            if method=='miso': self.assertEqual(float(z.grad[:,[0,31]].abs().sum()),0)

    def test_all_ignored_is_finite_zero(self):
        y=torch.full((1,1,2,2,2),-1)
        annotation=torch.ones(1,32,dtype=torch.bool)
        for f,c in [(lambda z:task_loss(z,y,torch.tensor([0]),annotation),5),
                    (lambda z:binary_loss(z,y,torch.tensor([0]),annotation),1),
                    (lambda z:miso_loss(z,y,annotation),32)]:
            z=torch.randn(1,c,2,2,2,requires_grad=True); loss=f(z); loss.backward()
            self.assertEqual(float(loss),0); self.assertEqual(float(z.grad.abs().sum()),0)

    def test_network_gradients_and_uniseg_cache_equivalence(self):
        torch.manual_seed(12)
        for method,net in [('uniseg',UniSegCellMap(backbone(5))),('conemos',CoNeMOSCellMap(backbone(1)))]:
            x=torch.randn(2,1,16,16,16); q=torch.tensor([0,13 if method=='uniseg' else 30])
            outputs=net(x,q)
            target=torch.randint(0,32,(2,1,16,16,16))
            targets=[target,torch.nn.functional.interpolate(target.float(),size=8,mode='nearest').long()]
            loss=deep_loss(method,outputs,targets,q,torch.ones(2,32,dtype=torch.bool));loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertFalse([n for n,p in net.named_parameters() if p.grad is None])
            if method=='uniseg':
                net.eval()
                with torch.no_grad():
                    a=net.decode(net.encode(x),q); b=net(x,q)
                self.assertTrue(all(torch.allclose(u,v) for u,v in zip(a,b)))
                self.assertGreater(float(net.universal_prompt.grad.abs().sum()),0)
            else:
                self.assertEqual(len(net.feature_films),10)
                for layer in net.feature_films:
                    self.assertGreater(float(layer.affine.weight.grad.abs().sum()),0)

    def test_all_labels_can_be_reconstructed_with_deterministic_ties(self):
        best=np.zeros(31,np.float32); seg=np.zeros(31,np.uint8)
        for labels in TASKS:
            p=np.zeros((len(labels),31),np.float32)
            for i,c in enumerate(labels): p[i,c-1]=.9
            merge_candidates(best,seg,p,labels,.1)
        np.testing.assert_array_equal(seg,np.arange(1,32))
        best=np.zeros(2,np.float32);seg=np.zeros(2,np.uint8)
        merge_candidates(best,seg,np.array([[.7,.4]]),(31,),.5)
        merge_candidates(best,seg,np.array([[.7,.4]]),(1,),.5)
        np.testing.assert_array_equal(seg,[1,0])

    def test_original_geometry_restoration(self):
        from types import SimpleNamespace
        prop=dict(shape_before_cropping=[4,5,6],bbox_used_for_cropping=[[1,3],[2,4],[0,3]])
        restored=restore_grid(np.ones((2,2,3),np.uint8),prop,SimpleNamespace(transpose_backward=[2,0,1]))
        self.assertEqual(restored.shape,(6,4,5)); self.assertEqual(int(restored.sum()),12)

    def test_split_and_repeat_sampler(self):
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiterature import LiteratureTrainer
        meta=json.loads((Path(__file__).parent/'data.json').read_text())
        self.assertEqual(len(meta['train_cases']),202); self.assertEqual(len(meta['validation_cases']),16)
        self.assertFalse(set(meta['train_cases']) & set(meta['validation_cases']))
        trainer=object.__new__(LiteratureTrainer)
        trainer.train_cases=set(meta['train_cases']);trainer.present_ids=meta['present_ids']
        trainer.frequencies={int(k):v for k,v in meta['annotated_crop_frequency'].items()}
        p=trainer.repeat_probabilities(meta['train_cases'])
        self.assertAlmostEqual(float(p.sum()),1);self.assertTrue((p>0).all())
        self.assertIsNone(trainer.repeat_probabilities(meta['validation_cases']))


if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main(verbosity=2)
