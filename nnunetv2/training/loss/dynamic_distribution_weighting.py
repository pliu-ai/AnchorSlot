from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn


class DynamicDistributionWeighting(nn.Module):
    """EMA prediction-frequency weights for dynamically imbalanced classes.

    Counts and denominators may be class-specific. This is useful for
    conditional models, where only the classes selected by the current query
    are observable in a training sample. DDP workers synchronize the sufficient
    statistics before updating their local EMA, so every rank uses identical
    weights.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        ema: float = 0.99,
        alpha: float = 0.5,
        min_weight: float = 0.2,
        max_weight: float = 8.0,
        eps: float = 1e-6,
        warmup_epochs: int = 0,
        ramp_epochs: int = 0,
        neutral_class_ids: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if not 0.0 <= ema < 1.0:
            raise ValueError("ema must be in [0, 1)")
        if alpha < 0.0:
            raise ValueError("alpha must be >= 0")
        if min_weight <= 0.0 or max_weight < min_weight:
            raise ValueError("DDW bounds require 0 < min_weight <= max_weight")
        if eps <= 0.0:
            raise ValueError("eps must be > 0")
        if warmup_epochs < 0 or ramp_epochs < 0:
            raise ValueError("warmup_epochs and ramp_epochs must be >= 0")
        neutral = tuple(sorted({int(i) for i in neutral_class_ids}))
        if any(i < 0 or i >= num_classes for i in neutral):
            raise ValueError("neutral_class_ids contains an out-of-range class")

        self.num_classes = int(num_classes)
        self.ema = float(ema)
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.eps = float(eps)
        self.warmup_epochs = int(warmup_epochs)
        self.ramp_epochs = int(ramp_epochs)
        self.neutral_class_ids = neutral

        self.register_buffer(
            "class_frequency",
            torch.full((self.num_classes,), 1.0 / self.num_classes, dtype=torch.float32),
        )
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, counts: torch.Tensor, denominators: torch.Tensor) -> None:
        """Update the EMA from per-class predicted counts and opportunities."""
        counts = counts.detach().to(dtype=torch.float32)
        denominators = denominators.detach().to(device=counts.device, dtype=torch.float32)
        if counts.shape != (self.num_classes,) or denominators.shape != (self.num_classes,):
            raise ValueError(
                f"expected DDW statistics of shape ({self.num_classes},), "
                f"got counts={tuple(counts.shape)}, denominators={tuple(denominators.shape)}"
            )

        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            dist.all_reduce(denominators, op=dist.ReduceOp.SUM)

        if self.class_frequency.device != counts.device:
            self.class_frequency = self.class_frequency.to(counts.device)
            self.num_updates = self.num_updates.to(counts.device)

        observed = denominators > 0
        if not torch.any(observed):
            return
        batch_frequency = counts[observed] / denominators[observed].clamp_min(1.0)
        # Boolean indexing produces a copy, so this must be an explicit indexed
        # assignment rather than chained in-place operations.
        self.class_frequency[observed] = (
            self.ema * self.class_frequency[observed]
            + (1.0 - self.ema) * batch_frequency
        )
        self.num_updates.add_(1)

    def _ramp_factor(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        return min(1.0, max(0.0, (epoch - self.warmup_epochs + 1) / self.ramp_epochs))

    @torch.no_grad()
    def weights(self, epoch: int, device: torch.device | None = None) -> torch.Tensor:
        frequency = self.class_frequency
        active = torch.ones(self.num_classes, dtype=torch.bool, device=frequency.device)
        if self.neutral_class_ids:
            active[list(self.neutral_class_ids)] = False

        result = torch.ones_like(frequency)
        if torch.any(active) and self.alpha > 0.0:
            balanced = (1.0 / frequency[active].clamp_min(self.eps)).pow(self.alpha)
            balanced = balanced / balanced.mean().clamp_min(self.eps)
            balanced = balanced.clamp(self.min_weight, self.max_weight)
            # Restore mean one after clipping when possible, then enforce bounds.
            balanced = (balanced / balanced.mean().clamp_min(self.eps)).clamp(
                self.min_weight, self.max_weight
            )
            result[active] = balanced

        ramp = self._ramp_factor(int(epoch))
        result = 1.0 + ramp * (result - 1.0)
        if self.neutral_class_ids:
            result[list(self.neutral_class_ids)] = 1.0
        return result.to(device=device) if device is not None else result

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, ema={self.ema}, alpha={self.alpha}, "
            f"bounds=({self.min_weight}, {self.max_weight}), "
            f"warmup_epochs={self.warmup_epochs}, ramp_epochs={self.ramp_epochs}, "
            f"neutral_class_ids={self.neutral_class_ids}"
        )
