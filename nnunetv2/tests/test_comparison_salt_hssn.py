"""Equation-level regression tests for the SALT/HSSN CellMap adaptations."""
import math
from contextlib import nullcontext

import pytest
import torch
from torch.nn import functional as F

from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.hierarchy import (
    HSSN_HIERARCHY, SALT_HIERARCHY, LabelHierarchy,
)
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.salt import (
    SALTLoss, SALT_NUM_LOGITS, salt_leaf_log_probs, salt_node_log_probs,
)
from nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.hssn import (
    HSSNLoss, HSSN_NUM_LOGITS, HSSNProjectionHead, hssn_leaf_scores,
    hssn_triplet_loss, hssn_triplet_weight, sample_hssn_triplets,
)


TOY = LabelHierarchy((-1, 3, 3, -1), 3)


def test_cellmap_tree_covers_original_leaf_ids_and_balances_hssn_paths():
    assert SALT_NUM_LOGITS == 45 and HSSN_NUM_LOGITS == 50
    assert SALT_HIERARCHY.parents[4:7] == (32, 32, 32)
    assert SALT_HIERARCHY.parents[17:19] == (44, 44)
    assert SALT_HIERARCHY.parents[27] == -1
    assert {len(p) for p in HSSN_HIERARCHY.paths[:32]} == {2}
    assert sum(len(c) for _, c in SALT_HIERARCHY.children) == 45
    distances = HSSN_HIERARCHY.leaf_distances()
    assert distances[4, 5] == 2 and distances[4, 7] == 4
    assert distances[0, 1] == 4 and torch.equal(distances, distances.T)


def test_salt_probabilities_obey_tree_marginals_and_sum_to_one():
    torch.manual_seed(4)
    logits = torch.randn(2, 45, 3, 2, 2, requires_grad=True)
    probs = salt_node_log_probs(logits).exp()
    torch.testing.assert_close(probs[:, :32].sum(1), torch.ones_like(probs[:, 0]))
    for parent, children in SALT_HIERARCHY.children:
        if parent != -1:
            torch.testing.assert_close(probs[:, children].sum(1), probs[:, parent])
    toy = salt_node_log_probs(torch.zeros(1, 4, 1), TOY).exp()
    torch.testing.assert_close(toy.flatten(), torch.tensor([.5, .25, .25, .5]))


def test_salt_ancestor_ce_is_not_leaf_nll_and_has_expected_gradient():
    logits = torch.zeros(1, 4, 1, requires_grad=True)
    loss = SALTLoss(dice_weight=0, hierarchy=TOY)(logits, torch.tensor([[1]]))
    assert loss.item() == pytest.approx(math.log(8))
    loss.backward()
    # Root decision is supervised twice, leaf decision once.
    torch.testing.assert_close(logits.grad.flatten(), torch.tensor([1., -.5, .5, -1.]))


def test_salt_matches_dense_reachability_ce_dice_reference():
    torch.manual_seed(8)
    logits = torch.randn(2, 45, 2, 2, requires_grad=True)
    labels = torch.tensor([[[4, 5], [7, 27]], [[0, 31], [17, 1]]])
    probs = salt_node_log_probs(logits).exp()
    reachability = torch.tensor([[int(node in SALT_HIERARCHY.paths[leaf]) for node in range(45)]
                                for leaf in range(32)], dtype=probs.dtype)
    dense = reachability[labels].movedim(-1, 1)
    ce = -(dense * probs.log()).sum(1).flatten(1).mean(1)
    dice = 1 - 2 * (dense * probs).flatten(2).sum(-1) / (
        dense.flatten(2).sum(-1) + probs.flatten(2).sum(-1) + 1e-5)
    expected = (ce + dice.mean(1)).mean()
    actual = SALTLoss()(logits, labels[:, None])
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


@pytest.mark.parametrize("method", ["salt", "hssn"])
def test_ignored_voxel_values_and_gradients_do_not_affect_loss(method):
    channels = 45 if method == "salt" else 50
    criterion = SALTLoss(ignore_label=255) if method == "salt" else HSSNLoss(ignore_label=255)
    torch.manual_seed(10)
    logits = torch.randn(1, channels, 4, requires_grad=True)
    labels = torch.tensor([[[4, 255, 7, 999]]])
    valid = torch.tensor([[[1, 1, 1, 0]]], dtype=torch.bool)
    original = labels.clone()
    # Ignored values can be nonfinite: selection must happen before activation.
    with torch.no_grad():
        logits[:, :, 1] = float("nan")
        logits[:, :, 3] = float("inf")
    actual = criterion(logits, labels, valid_mask=valid)
    reference = criterion(logits[:, :, [0, 2]], labels[:, :, [0, 2]])
    torch.testing.assert_close(actual, reference)
    actual.backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad[:, :, [1, 3]]) == 0
    assert torch.equal(labels, original)


@pytest.mark.parametrize("method", ["salt", "hssn"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_all_ignored_is_finite_graph_zero(method, dtype):
    channels = 45 if method == "salt" else 50
    logits = torch.full((2, channels, 3), float("nan"), dtype=dtype, requires_grad=True)
    labels = torch.full((2, 3), -1)
    criterion = SALTLoss() if method == "salt" else HSSNLoss()
    result = criterion(logits, labels)
    assert result.item() == 0 and result.dtype == torch.float32
    result.backward()
    assert torch.count_nonzero(logits.grad) == 0


@pytest.mark.parametrize("method", ["salt", "hssn"])
def test_extreme_half_logits_still_give_finite_loss_and_gradients(method):
    channels = 45 if method == "salt" else 50
    logits = torch.linspace(-1000, 1000, channels, dtype=torch.float16)[None, :, None].requires_grad_()
    criterion = SALTLoss() if method == "salt" else HSSNLoss()
    loss = criterion(logits, torch.tensor([[4]]))
    assert torch.isfinite(loss) and loss.dtype == torch.float32
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_hssn_focal_tree_min_matches_direct_paper_equation_and_gradient():
    torch.manual_seed(5)
    logits = torch.randn(1, 4, 3, requires_grad=True)
    labels = torch.tensor([[0, 1, 2]])
    probability = logits.sigmoid()
    values = []
    for voxel, leaf in enumerate(labels[0].tolist()):
        per_voxel = []
        for node in range(4):
            if node in TOY.paths[leaf]:
                score = probability[0, list(TOY.paths[node]), voxel].min()
                per_voxel.append(-((1 - score) ** 2) * score.log())
            else:
                score = probability[0, list(TOY.descendants[node]), voxel].max()
                per_voxel.append(-(score ** 2) * (1 - score).log())
        values.append(torch.stack(per_voxel).sum())
    expected = torch.stack(values).mean()
    actual = HSSNLoss(hierarchy=TOY).segmentation_loss(logits, labels)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                               torch.autograd.grad(expected, logits)[0])


def test_hssn_positive_ancestor_and_negative_descendant_constraints_route_gradients():
    # Leaf 1 positive; its low-scoring parent receives the positive constraint.
    logits = torch.tensor([[[0.], [2.], [-3.], [-2.]]], requires_grad=True)
    HSSNLoss(hierarchy=TOY, gamma=0).segmentation_loss(logits, torch.tensor([[1]])).backward()
    assert logits.grad[0, 1, 0] == 0
    assert logits.grad[0, 3, 0] < 0
    # Target 0; high leaf 1 makes the entire other subtree violate its negative.
    logits2 = torch.tensor([[[0.], [2.], [-3.], [-2.]]], requires_grad=True)
    HSSNLoss(hierarchy=TOY, gamma=0).segmentation_loss(logits2, torch.tensor([[0]])).backward()
    assert logits2.grad[0, 3, 0] == 0
    assert logits2.grad[0, 1, 0] > .88


def test_hssn_path_inference_combines_leaf_and_parent_without_depth_bias():
    scores = hssn_leaf_scores(torch.zeros(1, 50, 1))
    torch.testing.assert_close(scores, torch.ones_like(scores))
    logits = torch.full((1, 50, 1), -4.)
    logits[:, 4] = 4  # high leaf, low parent
    logits[:, 7] = 2
    logits[:, 33] = 2  # stronger path, although lower standalone leaf score
    assert hssn_leaf_scores(logits).argmax(1).item() == 7


def test_hssn_tree_triplet_uses_paper_cosine_and_dynamic_margin():
    embeddings = torch.tensor([[1., 0.], [0., 1.], [1., 0.]], requires_grad=True)
    labels = torch.tensor([4, 5, 7])
    value = hssn_triplet_loss(embeddings, labels, torch.tensor([[0, 1, 2]]))
    # d_pos=.5, d_neg=0; tree distances 2 and 4, height2 -> margin=.35.
    assert value.item() == pytest.approx(.85)
    value.backward()
    assert torch.isfinite(embeddings.grad).all() and embeddings.grad.abs().sum() > 0
    with pytest.raises(ValueError, match="tree distance"):
        hssn_triplet_loss(embeddings, labels, torch.tensor([[0, 2, 1]]))


def test_hssn_sampling_is_bounded_valid_and_reproducible_with_no_ignored_negatives():
    labels = torch.tensor([4] * 250 + [5] * 250 + [7] * 250 + [0] * 250 + [-1] * 100)
    torch.manual_seed(12)
    triplets = sample_hssn_triplets(labels, max_triplets_per_class=17)
    torch.manual_seed(12)
    assert torch.equal(triplets, sample_hssn_triplets(labels, max_triplets_per_class=17))
    assert 0 < len(triplets) <= 4 * 17
    a, p, n = labels[triplets].unbind(1)
    distances = HSSN_HIERARCHY.leaf_distances()
    assert (a >= 0).all() and (p >= 0).all() and (n >= 0).all()
    assert (distances[a, p] < distances[a, n]).all()
    assert (triplets[:, 0] != triplets[:, 1]).all()
    assert torch.bincount(a).max() <= 17


def test_hssn_projection_receives_gradient_with_coarse_feature_grid():
    torch.manual_seed(13)
    projection = HSSNProjectionHead(4, projection_dim=8)
    features = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
    coarse_labels = torch.tensor([[[[4, 5], [7, 4]], [[5, 7], [4, -1]]]])
    target = F.interpolate(coarse_labels[:, None].float(), size=(4, 4, 4), mode="nearest").long()
    projected = projection(features)
    loss = HSSNLoss(max_triplets_per_class=5).feature_loss(projected, target, progress=1)
    assert loss.item() > 0
    loss.backward()
    assert features.grad.abs().sum() > 0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in projection.parameters())


@pytest.mark.parametrize("labels", [torch.zeros(1, 2, 2, 2, dtype=torch.long),
                                    torch.full((1, 2, 2, 2), -1, dtype=torch.long)])
def test_hssn_no_valid_triplet_keeps_projection_parameters_in_graph(labels):
    projection = HSSNProjectionHead(4, projection_dim=8)
    result = HSSNLoss().feature_loss(projection(torch.randn(1, 4, 2, 2, 2)), labels)
    assert result.item() == 0
    result.backward()
    assert all(p.grad is not None and torch.count_nonzero(p.grad) == 0 for p in projection.parameters())


def test_hssn_schedule_and_separate_feature_loss_match_combined_call():
    assert hssn_triplet_weight(0) == 0
    assert hssn_triplet_weight(.5) == pytest.approx(.25)
    assert hssn_triplet_weight(1) == .5
    loss = HSSNLoss(max_triplets_per_class=2)
    logits = torch.randn(1, 50, 2, 2)
    features = torch.randn(1, 8, 2, 2)
    target = torch.tensor([[[4, 5], [7, 4]]])
    torch.manual_seed(14)
    separate = loss(logits, target) + loss.feature_loss(features, target, progress=.5)
    torch.manual_seed(14)
    combined = loss(logits, target, features=features, progress=.5)
    torch.testing.assert_close(separate, combined)


@pytest.mark.parametrize("criterion,channels", [(SALTLoss(), 45), (HSSNLoss(), 50)])
def test_unmasked_invalid_label_is_rejected(criterion, channels):
    with pytest.raises(ValueError, match="outside the leaf"):
        criterion(torch.zeros(1, channels, 1), torch.tensor([[32]]))


@pytest.mark.parametrize("kind,channels", [(SALTLoss, 45), (HSSNLoss, 50)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_node_checkpoint_matches_noncheckpoint_values_and_gradients_under_amp(kind, channels, dtype):
    torch.manual_seed(202)
    initial = torch.randn(2, channels, 2, 3, 2).to(dtype)
    # Several different paths and batches catch late-bound checkpoint metadata.
    target = torch.tensor([[[[0, 4], [5, 6], [7, 27]], [[31, 17], [18, 3], [255, 1]]],
                           [[[29, 25], [26, 2], [8, 9]], [[10, 13], [14, 15], [16, 30]]]])
    valid = torch.ones_like(target, dtype=torch.bool)
    valid[1, 1, 2, 1] = False
    results, gradients = [], []
    for use_checkpoint in (False, True):
        logits = initial.clone().requires_grad_()
        criterion = kind(ignore_label=255, checkpoint_nodes=use_checkpoint)
        context = nullcontext() if dtype == torch.float32 else torch.autocast("cpu", dtype=dtype)
        with context:
            loss = criterion(logits, target[:, None], valid_mask=valid[:, None])
        # Backward outside autocast, as in the actual trainer.
        gradient, = torch.autograd.grad(loss, logits)
        assert torch.isfinite(loss) and torch.isfinite(gradient).all()
        assert loss.dtype == torch.float32
        results.append(loss.detach())
        gradients.append(gradient)
    torch.testing.assert_close(results[0], results[1], rtol=0, atol=0)
    torch.testing.assert_close(gradients[0], gradients[1])
    # A separate non-autocast FP32 evaluation on the same quantized inputs
    # verifies that AMP never changes the criterion's FP32 arithmetic.
    reference_logits = initial.float().requires_grad_()
    reference_loss = kind(ignore_label=255, checkpoint_nodes=False)(
        reference_logits, target, valid_mask=valid)
    reference_gradient, = torch.autograd.grad(reference_loss, reference_logits)
    torch.testing.assert_close(results[1], reference_loss.detach(), rtol=0, atol=0)
    torch.testing.assert_close(gradients[1], reference_gradient.to(dtype))
    assert torch.count_nonzero(gradients[1][1, :, 1, 2, 1]) == 0
    assert torch.count_nonzero(gradients[1][0, :, 1, 2, 0]) == 0


@pytest.mark.parametrize("kind,channels", [(SALTLoss, 45), (HSSNLoss, 50)])
def test_node_checkpoint_reduces_unique_saved_floating_storage(kind, channels):
    """Measure actual autograd saved storage on CPU, not a GPU memory estimate."""
    torch.manual_seed(203)
    initial = torch.randn(1, channels, 4, 4, 4)
    target = torch.arange(64).remainder(32).reshape(1, 4, 4, 4)
    retained_bytes = []
    for use_checkpoint in (False, True):
        storage = {}

        def pack(tensor):
            if tensor.is_floating_point() and tensor.numel():
                # Repeated checkpoint references/views share one storage; do
                # not count that allocation once for every hierarchy node.
                allocation = tensor.untyped_storage()
                storage[allocation.data_ptr()] = allocation.nbytes()
            return tensor

        logits = initial.clone().requires_grad_()
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            loss = kind(checkpoint_nodes=use_checkpoint)(logits, target)
        retained_bytes.append(sum(storage.values()))
        loss.backward()
        assert torch.isfinite(logits.grad).all()
    assert retained_bytes[1] < retained_bytes[0] * .75, retained_bytes


@pytest.mark.parametrize("kind,channels,module", [
    (SALTLoss, 45, "nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.salt"),
    (HSSNLoss, 50, "nnunetv2.training.nnUNetTrainer.variants.comparison_baselines.hssn"),
])
def test_node_checkpoint_not_called_without_a_backward_graph(kind, channels, module, monkeypatch):
    import importlib

    def forbidden(*args, **kwargs):
        raise AssertionError("checkpoint should not run without a backward graph")

    monkeypatch.setattr(importlib.import_module(module), "checkpoint", forbidden)
    labels = torch.tensor([[4, 5, 7]])
    criterion = kind(checkpoint_nodes=True)
    assert torch.isfinite(criterion(torch.randn(1, channels, 3), labels))
    with torch.no_grad():
        assert torch.isfinite(criterion(torch.randn(1, channels, 3, requires_grad=True), labels))
