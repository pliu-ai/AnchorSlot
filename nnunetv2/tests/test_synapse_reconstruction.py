from __future__ import annotations

import contextlib
import copy
import importlib.util
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_synapse import (
    DYNAMIC_GROUP_SPECS, remap_original_to_structured,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.synapse_consistency import (
    ConsistentSynapseLoss, reconstruct_consistent,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.synapse_reconstruction import (
    masked_probabilities, reconstruct_scores, original_label_loss,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_synapse_reconstruction import (
    nnUNetTrainerSynapseReconstructionScreen,
)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_scores_match_frozen_inference_and_mask_gradient(dtype):
    logits = {g: torch.randn(2, 9, 2, 3, 4, dtype=dtype, requires_grad=True) for g in range(5)}
    score = reconstruct_scores({g: masked_probabilities(x, g) for g, x in logits.items()})
    _, expected = reconstruct_consistent({g: x.float() for g, x in logits.items()})
    torch.testing.assert_close(score, expected)
    target = torch.randint(0, 14, (2, 1, 2, 3, 4))
    original_label_loss(score, target).backward()
    for g, x in logits.items():
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert torch.count_nonzero(x.grad) > 0
        assert torch.count_nonzero(x.grad[:, 5+DYNAMIC_GROUP_SPECS[g].num_slots:8]) == 0


def test_perfect_reconstruction_every_label_and_padding():
    labels = torch.arange(14).reshape(1, 1, 1, 2, 7)
    probabilities = {}
    for g in range(5):
        target, _, _ = remap_original_to_structured(labels, torch.tensor([g]))
        logits = F.one_hot(target[:, 0].long(), 9).movedim(-1, 1).float()*30
        probabilities[g] = masked_probabilities(logits, g)
    scores = reconstruct_scores(probabilities)
    assert torch.equal(scores.argmax(1, keepdim=True), labels)
    assert original_label_loss(scores, labels).abs() < 1e-5
    labels[..., 0] = -1
    changed = scores.clone()
    changed[..., 0] = torch.rand_like(changed[..., 0])
    torch.testing.assert_close(original_label_loss(scores, labels), original_label_loss(changed, labels))


def test_nll_is_normalized_scores_not_softmax_of_probabilities():
    scores = torch.arange(1, 15).float().reshape(1, 14, 1, 1, 1).requires_grad_()
    target = torch.full((1, 1, 1, 1, 1), 13)
    p = scores / scores.sum(1, keepdim=True)
    gt = F.one_hot(target[:, 0], 14).movedim(-1, 1)
    dice = ((2*p*gt + 1e-5)/(p+gt+1e-5))[:, 1:].mean()
    torch.testing.assert_close(original_label_loss(scores, target), -p[:, 13].log().mean()+1-dice)
    torch.testing.assert_close(original_label_loss(scores, target), original_label_loss(scores*7, target))


def test_all_invalid_has_zero_finite_gradients():
    scores = torch.rand(2, 14, 2, 3, 4, requires_grad=True)
    loss = original_label_loss(scores, torch.full((2, 1, 2, 3, 4), -1))
    assert loss == 0
    loss.backward()
    assert torch.count_nonzero(scores.grad) == 0


@pytest.mark.parametrize('bad', [0, 4, 6])
def test_reconstruction_refuses_missing_or_extra_groups(bad):
    with pytest.raises(ValueError):
        reconstruct_scores({g: torch.rand(1,9,1,1,1) for g in range(bad)})


def test_unknown_label_and_shape_fail_closed():
    scores = torch.rand(1,14,2,2,2)
    with pytest.raises(ValueError): original_label_loss(scores, torch.full((1,1,2,2,2), 14))
    with pytest.raises(ValueError): original_label_loss(scores, torch.zeros(1,1,1,1,1))
    p = {g: torch.rand(1,9,1,1,1) for g in range(5)}
    p[0] = torch.rand(2,9,1,1,1)
    with pytest.raises(ValueError): reconstruct_scores(p)


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Conv3d(1, 3, 1)
        self.decoder = torch.nn.Conv3d(3, 9, 1)
        self.condition = torch.nn.Embedding(5, 3)
    def encode(self, x): return [self.encoder(x)]
    def forward(self, x, groups): return self.decode_from_skips(self.encode(x), groups)
    def decode_from_skips(self, features, groups):
        x = features[0]+self.condition(groups)[:,:,None,None,None]
        return [self.decoder(x)]


def toy_trainer(weight):
    t = nnUNetTrainerSynapseReconstructionScreen.__new__(nnUNetTrainerSynapseReconstructionScreen)
    t.network = TinyNetwork()
    t.num_dynamic_groups = 5
    t.reconstruction_weight = weight
    t._unwrap_network = lambda: t.network
    t._autocast_context = contextlib.nullcontext
    t._output_to_float = lambda xs: [x.float() for x in xs]
    t._remap_target_for_group = lambda target, ids: remap_original_to_structured(target, ids)
    loss = ConsistentSynapseLoss()
    t._compute_structured_loss = lambda outputs, target, valid, slots: loss(outputs[0],target,valid,slots)
    return t


@pytest.mark.parametrize('weight', [0., .25])
def test_checkpointed_decoder_loss_and_all_parameter_gradients_equal_eager(weight):
    torch.manual_seed(42)
    t = toy_trainer(weight)
    data = torch.randn(2,1,2,3,4)
    target = torch.randint(0,14,(2,1,2,3,4))
    loss, structured, reconstruction = t.joint_loss(data, target, False)
    torch.testing.assert_close(loss, structured+weight*reconstruction)
    loss.backward()
    gradients = {k:p.grad.clone() for k,p in t.network.named_parameters()}
    t.network.zero_grad(set_to_none=True)
    actual, _, _ = t.joint_loss(data, target, True)
    torch.testing.assert_close(loss,actual)
    actual.backward()
    for k,p in t.network.named_parameters():
        torch.testing.assert_close(p.grad,gradients[k])
        assert torch.isfinite(p.grad).all() and torch.count_nonzero(p.grad)>0
    assert torch.all(t.network.condition.weight.grad.abs().sum(1)>0)


def test_zero_weight_is_exact_query_average():
    t = toy_trainer(0.)
    data=torch.randn(2,1,2,3,4); target=torch.randint(0,14,(2,1,2,3,4))
    actual, structured, _=t.joint_loss(data,target,False)
    torch.testing.assert_close(actual,structured,rtol=0,atol=0)
    independent=[]
    for g in range(5):
        ids=torch.full((2,),g,dtype=torch.long)
        y,v,a=remap_original_to_structured(target,ids)
        independent.append(t._compute_structured_loss(t.network.decode_from_skips(t.network.encode(data),ids),y,v,a))
    torch.testing.assert_close(actual,torch.stack(independent).mean())


@pytest.mark.parametrize('arm', ['control','no_suppression','lr003','all_query_control','reconstruction'])
def test_query_schedule_is_global_and_resume_stable(arm):
    t=toy_trainer(0.); t.screen_arm=arm; t.query_every=4; t.num_iterations_per_epoch=250
    t.current_epoch=50
    for i in range(8):
        t._screen_iteration=i
        assert t.use_all_queries()==(arm in ('all_query_control','reconstruction') and (12500+i)%4==0)


def test_arm_configs_change_only_intended_parameters():
    path=Path(__file__).resolve().parents[2]/'scripts/run_synapse_fold0_followup.py'
    spec=importlib.util.spec_from_file_location('followup_runner_test',path)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    control=mod.arm_config('control')
    assert {k for k in control if control[k]!=mod.arm_config('no_suppression')[k]}=={'lambda_suppress'}
    assert {k for k in control if control[k]!=mod.arm_config('lr003')[k]}=={'initial_lr'}
    a=mod.arm_config('all_query_control'); b=mod.arm_config('reconstruction')
    assert {k for k in a if a[k]!=b[k]}=={'lambda_reconstruction'}
    with pytest.raises(ValueError): mod.arm_config('unknown')


@pytest.mark.parametrize('arm,weight', [('all_query_control', 0.), ('reconstruction', .25)])
@pytest.mark.parametrize('epoch', [0, 1, 50, 51])
def test_actual_train_steps_collate_across_epoch_phase_and_preserve_loss(arm, weight, epoch):
    from nnunetv2.utilities.collate_outputs import collate_outputs
    t = toy_trainer(weight)
    t.device = torch.device('cpu')
    t.is_ddp = False
    t.screen_arm, t.query_every, t.num_iterations_per_epoch = arm, 4, 250
    t.current_epoch, t._screen_iteration = epoch, 0
    t._screen_diagnostics = []
    t._query_coverage = np.zeros((5, 3), dtype=np.int64)
    t._group_sample_counter_epoch = np.zeros(5, dtype=np.int64)
    t.grad_scaler, t.grad_clip_norm = None, 12.
    t.optimizer = torch.optim.SGD(t.network.parameters(), lr=.001)
    t._get_optimizer_parameters = lambda: list(t.network.parameters())
    t._extract_group_ids_for_batch = lambda batch, target: batch['group_id']
    t.use_text_contrastive = t.use_text_generated_slot_head = False
    t.text_contrast_weight, t.loss_in_fp32 = 0., True
    batch = dict(data=torch.randn(2, 1, 2, 3, 4),
                 target=torch.randint(0, 14, (2, 1, 2, 3, 4)), group_id=torch.tensor([0, 4]))
    outputs = []
    for _ in range(4):
        if t.use_all_queries():
            expected, _, _ = t.joint_loss(batch['data'], batch['target'], False)
            expected = expected.detach().numpy()
        else:
            ids = batch['group_id']
            remapped, valid, active = t._remap_target_for_group(batch['target'], ids)
            expected = t._compute_structured_loss(t.network(batch['data'], ids), remapped, valid, active).detach().numpy()
        output = t.train_step(batch)
        np.testing.assert_allclose(output['loss'], expected, rtol=1e-6)
        outputs.append(output)
    assert len(t._screen_diagnostics) == 1
    assert all(set(x) == set(outputs[0]) for x in outputs)
    for ordered in (outputs, outputs[::-1]):
        collated = collate_outputs(ordered)
        assert len(collated['loss']) == 4
        assert np.isfinite(collated['loss']).all()
