from __future__ import annotations

import os
from typing import Optional, Sequence, Union

import torch
from torch import nn

from .condition_encoding import GroupConditionEncoder


class _MedNeXtDeepSupervisionProxy:
    """Expose nnU-Net's decoder.deep_supervision API for MedNeXt."""

    def __init__(self, backbone: nn.Module) -> None:
        self.backbone = backbone

    @property
    def deep_supervision(self) -> bool:
        return bool(getattr(self.backbone, "do_ds", False))

    @deep_supervision.setter
    def deep_supervision(self, enabled: bool) -> None:
        if hasattr(self.backbone, "do_ds"):
            self.backbone.do_ds = bool(enabled)


class StructuredConditionalMedNeXt(nn.Module):
    """
    Wrapper for MedNeXt backbones that injects dynamic-group conditioning.

    MedNeXt does not expose the same encoder/decoder internals as the default
    nnUNet backbones used in StructuredConditionalUNet, so we condition at the
    input feature level via FiLM-style modulation.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_groups: int,
        num_input_channels: int,
        cond_dim: int = 64,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.decoder = _MedNeXtDeepSupervisionProxy(backbone)
        self.num_groups = int(num_groups)
        self.num_input_channels = int(num_input_channels)
        default_group = int(os.environ.get("NNUNET_STRUCTCOND_INFER_GROUP_ID", "0"))
        self.default_infer_group_id = int(max(0, min(default_group, self.num_groups - 1)))
        self.clamp_input_conditioning = str(
            os.environ.get("NNUNET_STRUCTCOND_CLAMP_INPUT_CONDITIONING", "0")
        ).lower() in {"1", "true", "yes", "y"}
        self.input_gamma_min = float(os.environ.get("NNUNET_STRUCTCOND_INPUT_GAMMA_MIN", "0.1"))
        self.input_gamma_max = float(os.environ.get("NNUNET_STRUCTCOND_INPUT_GAMMA_MAX", "2.5"))
        self.input_beta_abs_max = float(os.environ.get("NNUNET_STRUCTCOND_INPUT_BETA_ABS_MAX", "5.0"))
        if self.clamp_input_conditioning and self.input_gamma_min >= self.input_gamma_max:
            raise ValueError(
                "Expected NNUNET_STRUCTCOND_INPUT_GAMMA_MIN "
                "< NNUNET_STRUCTCOND_INPUT_GAMMA_MAX."
            )
        if self.clamp_input_conditioning and self.input_beta_abs_max <= 0:
            raise ValueError("NNUNET_STRUCTCOND_INPUT_BETA_ABS_MAX must be > 0.")

        self.condition_encoder = GroupConditionEncoder(num_groups=self.num_groups, embedding_dim=int(cond_dim))
        self.input_affine = nn.Linear(self.condition_encoder.output_dim, 2 * self.num_input_channels)

        # Identity initialization: gamma = 1, beta = 0.
        nn.init.zeros_(self.input_affine.weight)
        nn.init.zeros_(self.input_affine.bias)
        with torch.no_grad():
            self.input_affine.bias[: self.num_input_channels].fill_(1.0)

    def _normalize_group_ids(self, x: torch.Tensor, group_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if group_ids is None:
            return torch.full(
                (x.shape[0],),
                int(self.default_infer_group_id),
                dtype=torch.long,
                device=x.device,
            )
        return group_ids.reshape(-1).to(device=x.device, dtype=torch.long)

    def _apply_input_conditioning(
        self,
        x: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> torch.Tensor:
        cond_vec = self.condition_encoder(group_ids, batch_size=x.shape[0], device=x.device)
        gamma_beta = self.input_affine(cond_vec)
        gamma, beta = torch.split(gamma_beta, self.num_input_channels, dim=1)
        if self.clamp_input_conditioning:
            gamma = gamma.clamp(min=self.input_gamma_min, max=self.input_gamma_max)
            beta = beta.clamp(min=-self.input_beta_abs_max, max=self.input_beta_abs_max)
        view_shape = [x.shape[0], self.num_input_channels] + [1] * (x.ndim - 2)
        gamma = gamma.view(*view_shape)
        beta = beta.view(*view_shape)
        return gamma * x + beta

    def forward(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Sequence[torch.Tensor]]:
        group_ids = self._normalize_group_ids(x, group_ids)
        x = self._apply_input_conditioning(x, group_ids)
        output = self.backbone(x)
        # During inference, nnUNet expects a single logits tensor.
        if not self.training:
            return get_main_output(output)
        return output


def get_main_output(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
