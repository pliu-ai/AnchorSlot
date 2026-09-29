from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from nnunetv2.training.loss.dynamic_distribution_weighting import DynamicDistributionWeighting
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.structured_loss_er_dynamic import (
    StructuredConditionalLoss, StructuredLossConfig,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.structured_loss_er_dynamic_ddw import (
    AnchorSlotDynamicDistributionLoss,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic import (
    nnUNetTrainerStructuredConditionalERDynamic,
)
from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic_ddw import (
    nnUNetTrainerStructuredConditionalERDynamicDDW,
)


def test_warmup_preserves_full_loss_and_gradients():
    torch.manual_seed(33)
    cfg = StructuredLossConfig(batch_dice=False)
    base, ddw = StructuredConditionalLoss(cfg), AnchorSlotDynamicDistributionLoss(cfg)
    x = torch.randn(3, 9, 3, 4, 5, requires_grad=True)
    y = torch.randint(0, 9, (3, 1, 3, 4, 5))
    mask = torch.rand_like(y.float()) > 0.2
    active = torch.ones(3, 3, dtype=torch.bool)
    active[1, 2] = False
    y[1][y[1] == 7] = 0  # slot 3 is inactive for this sample
    ddw.set_sample_class_weights(torch.ones(3, 9))
    a, b = base(x, y, mask, active), ddw(x, y, mask, active)
    assert torch.allclose(a, b, atol=1e-6)
    ga = torch.autograd.grad(a, x, retain_graph=True)[0]
    gb = torch.autograd.grad(b, x)[0]
    assert torch.allclose(ga, gb, atol=1e-6)


def test_plainconv_ddw_does_not_replace_architecture():
    assert (nnUNetTrainerStructuredConditionalERDynamicDDW.build_network_architecture
            is nnUNetTrainerStructuredConditionalERDynamic.build_network_architecture)


def minimal_trainer(tmp_path):
    t = object.__new__(nnUNetTrainerStructuredConditionalERDynamicDDW)
    t.ddw_ema, t.ddw_alpha = 0.9, 0.25
    t.ddw_min_weight, t.ddw_max_weight = 0.5, 2.0
    t.ddw_warmup_epochs, t.ddw_ramp_epochs, t.ddw_eps = 5, 20, 1e-6
    t.ddw_tracker = DynamicDistributionWeighting(32, neutral_class_ids=(0,), **t._ddw_config())
    t.loss = AnchorSlotDynamicDistributionLoss()
    t.network = torch.nn.Linear(3, 2)
    t.optimizer = torch.optim.SGD(t.network.parameters(), lr=1e-5, momentum=0.9)
    t.grad_scaler = None
    t.logger = SimpleNamespace(get_checkpoint=lambda: {}, load_checkpoint=lambda state: None)
    t.local_rank, t.is_ddp, t.device = 0, False, torch.device('cpu')
    t.disable_checkpointing, t.was_initialized = False, True
    t.current_epoch, t._best_ema = 24, 0.5
    t.my_init_kwargs, t.inference_allowed_mirroring_axes = {}, (0, 1, 2)
    t.output_folder = str(tmp_path)
    t.enable_epoch_checkpoint_snapshots, t.epoch_checkpoint_snapshot_start = True, 5
    t.epoch_checkpoint_snapshot_every = 5
    t.reset_optimizer_on_load = t.freeze_conditioning_on_load = False
    return t


def test_checkpoint_restores_weights_ema_optimizer_epoch_and_next_step(tmp_path):
    torch.manual_seed(4)
    t = minimal_trainer(tmp_path)
    x = torch.randn(4, 3)

    def step(trainer):
        counts = torch.arange(32).float()
        trainer.ddw_tracker.update(counts, torch.full((32,), 32.0))
        trainer.optimizer.zero_grad()
        loss = trainer.network(x).square().mean() * trainer.ddw_tracker.weights(trainer.current_epoch)[3]
        loss.backward()
        trainer.optimizer.step()
        return loss.detach()

    step(t)
    t.save_checkpoint(str(tmp_path / 'checkpoint_latest.pth'))
    a = torch.load(tmp_path / 'checkpoint_latest.pth', weights_only=False)
    b = torch.load(tmp_path / 'checkpoint_ep25.pth', weights_only=False)
    assert a['current_epoch'] == b['current_epoch'] == 25
    assert torch.equal(a['ddw_state']['tracker']['class_frequency'], b['ddw_state']['tracker']['class_frequency'])
    t.current_epoch += 1
    expected = step(t)
    params = deepcopy(t.network.state_dict())
    tracker = deepcopy(t.ddw_tracker.state_dict())
    t.load_checkpoint(tmp_path / 'checkpoint_latest.pth')
    assert t.current_epoch == 25
    actual = step(t)
    assert torch.equal(expected, actual)
    for key, value in params.items():
        assert torch.equal(value, t.network.state_dict()[key])
    for key, value in tracker.items():
        assert torch.equal(value, t.ddw_tracker.state_dict()[key])
    broken = deepcopy(a)
    del broken['ddw_state']
    with pytest.raises(ValueError, match='requires saved ddw_state'):
        t.load_checkpoint(broken)
    broken = deepcopy(a)
    broken['ddw_state']['config']['alpha'] = 1.0
    with pytest.raises(ValueError, match='configuration differs'):
        t.load_checkpoint(broken)


def test_invalid_voxels_and_inactive_slots_do_not_update_distribution(tmp_path):
    t = minimal_trainer(tmp_path)
    t.ddw_tracker.ema = 0.0
    logits = torch.full((1, 9, 1, 1, 2), -3.)
    logits[:, 7] = 100.  # inactive conditional slot 3
    logits[:, 5] = 5.    # golgi_mem
    logits[..., 1] = -3.
    logits[:, 1, ..., 1] = 200.  # masked voxel must not count as ECS
    valid = torch.tensor([True, False]).reshape(1, 1, 1, 1, 2)
    old = t.ddw_tracker.class_frequency.clone()
    t._update_ddw_from_main_output(logits, valid, torch.tensor([[True, True, False]]), torch.tensor([1]))
    assert t.ddw_tracker.class_frequency[7] == 1.
    assert t.ddw_tracker.class_frequency[1] == 0.
    assert t.ddw_tracker.class_frequency[4] == old[4]  # unqueried mito unchanged
    weights = t._sample_output_weights(torch.tensor([1]), torch.device('cpu'))
    assert weights[0, 0] == weights[0, 8] == 1.
