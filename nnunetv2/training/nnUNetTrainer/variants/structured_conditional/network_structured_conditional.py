from __future__ import annotations

import os
from typing import Dict, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from .condition_encoding import (
    FiLMModulation,
    GroupConditionEncoder,
    TextConditionEncoder,
    TextConditionedGroupEmbedding,
)
from .text_generated_slot_head import TextGeneratedSlotHead


class StructuredConditionalUNet(nn.Module):
    """
    nnUNet backbone wrapper with decoder/head-side conditioning.

    Design goals:
    - shared encoder for all conditions
    - fixed output head with 11 channels
    - condition injection mainly in decoder stages (FiLM)
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_groups: int,
        num_output_channels: int,
        cond_dim: int = 64,
        condition_mode: str = "learned",
        group_text_matrix: Optional[torch.Tensor] = None,
        text_fusion: str = "concat_mlp",
        freeze_text_embeddings: bool = True,
        use_text_contrastive: bool = False,
        text_contrast_text_dim: Optional[int] = None,
        text_contrast_dim: int = 128,
        use_text_generated_slot_head: bool = False,
        text_generated_slot_embeddings: Optional[torch.Tensor] = None,
        text_generated_slot_group_to_labels: Optional[Mapping[int, Sequence[str]]] = None,
        text_generated_slot_hidden_dim: int = 512,
        text_generated_slot_use_bias: bool = True,
        text_generated_slot_normalize_weight: bool = True,
        text_generated_slot_mode: str = "residual",
        text_generated_slot_alpha: float = 1.0,
        text_generated_slot_start_channel: int = 5,
        text_generated_slot_num_slots: int = 3,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.encoder = backbone.encoder
        self.decoder = backbone.decoder
        self.num_groups = int(num_groups)
        self.num_output_channels = int(num_output_channels)
        self.use_text_contrastive = bool(use_text_contrastive)
        self.use_text_generated_slot_head = bool(use_text_generated_slot_head)
        default_group = int(os.environ.get("NNUNET_STRUCTCOND_INFER_GROUP_ID", "0"))
        self.default_infer_group_id = int(max(0, min(default_group, self.num_groups - 1)))

        self.condition_mode = str(condition_mode)
        if self.condition_mode in ("learned_text", "text_fusion"):
            self.condition_encoder = TextConditionedGroupEmbedding(
                num_groups=self.num_groups,
                cond_dim=int(cond_dim),
                group_text_matrix=group_text_matrix,
                use_text_conditioning=True,
                text_fusion=text_fusion,
                freeze_text_embeddings=freeze_text_embeddings,
            )
        elif self.condition_mode in ("text", "text_init"):
            if group_text_matrix is None:
                raise ValueError(f"condition_mode='{self.condition_mode}' requires group_text_matrix [num_groups, D]")
            self.condition_encoder = TextConditionEncoder(
                num_groups=self.num_groups,
                group_text_matrix=group_text_matrix,
                hidden_dim=int(cond_dim),
                learnable_text=(self.condition_mode == "text_init"),
            )
        elif self.condition_mode == "learned":
            self.condition_encoder = GroupConditionEncoder(num_groups=self.num_groups, embedding_dim=int(cond_dim))
        else:
            raise ValueError(
                "unknown condition_mode="
                f"'{self.condition_mode}' (expected 'learned', 'learned_text', 'text', or 'text_init')"
            )

        # Segmentation layers consume decoder feature maps. We modulate those features
        # right before they are projected into logits.
        decoder_stage_channels = [int(seg_layer.in_channels) for seg_layer in self.decoder.seg_layers]
        self.decoder_film = nn.ModuleList(
            [FiLMModulation(cond_dim=self.condition_encoder.output_dim, channels=c) for c in decoder_stage_channels]
        )
        self.text_generated_slot_mode = str(text_generated_slot_mode).strip().lower()
        self.text_generated_slot_alpha = float(text_generated_slot_alpha)
        self.text_generated_slot_start_channel = int(text_generated_slot_start_channel)
        self.text_generated_slot_num_slots = int(text_generated_slot_num_slots)
        self.text_generated_slot_group_to_labels: Dict[int, Sequence[str]] = {}
        self.latest_text_generated_slot_stats: Dict[str, object] = {}
        if self.use_text_generated_slot_head:
            if self.text_generated_slot_mode not in {"residual", "replace"}:
                raise ValueError("text_generated_slot_mode must be 'residual' or 'replace'")
            if text_generated_slot_embeddings is None:
                raise ValueError("use_text_generated_slot_head=True requires text_generated_slot_embeddings")
            if text_generated_slot_group_to_labels is None:
                raise ValueError("use_text_generated_slot_head=True requires text_generated_slot_group_to_labels")
            if text_generated_slot_embeddings.ndim != 3:
                raise ValueError(
                    "text_generated_slot_embeddings must be [num_groups, num_slots, text_dim], "
                    f"got {tuple(text_generated_slot_embeddings.shape)}"
                )
            if text_generated_slot_embeddings.shape[0] < self.num_groups:
                raise ValueError("text_generated_slot_embeddings must include every group")
            if text_generated_slot_embeddings.shape[1] != self.text_generated_slot_num_slots:
                raise ValueError(
                    "text_generated_slot_embeddings slot dimension must match text_generated_slot_num_slots"
                )
            end_channel = self.text_generated_slot_start_channel + self.text_generated_slot_num_slots
            if self.text_generated_slot_start_channel < 0 or end_channel > self.num_output_channels:
                raise ValueError("text-generated slot channel range is outside output channels")
            text_dim = int(text_generated_slot_embeddings.shape[2])
            self.register_buffer(
                "text_generated_slot_embeddings",
                text_generated_slot_embeddings.detach().float(),
                persistent=True,
            )
            self.text_generated_slot_group_to_labels = {
                int(group_id): tuple(str(label) for label in labels)
                for group_id, labels in text_generated_slot_group_to_labels.items()
            }
            self.text_generated_slot_heads = nn.ModuleList(
                [
                    TextGeneratedSlotHead(
                        in_channels=int(channels),
                        text_embedding_dim=text_dim,
                        num_slots=self.text_generated_slot_num_slots,
                        hidden_dim=int(text_generated_slot_hidden_dim),
                        use_bias=bool(text_generated_slot_use_bias),
                        normalize_weight=bool(text_generated_slot_normalize_weight),
                    )
                    for channels in decoder_stage_channels
                ]
            )
        if self.use_text_contrastive:
            if text_contrast_text_dim is None or int(text_contrast_text_dim) <= 0:
                raise ValueError("use_text_contrastive=True requires text_contrast_text_dim > 0")
            contrast_dim = int(text_contrast_dim)
            if contrast_dim <= 0:
                raise ValueError("text_contrast_dim must be > 0")
            last_decoder_channels = int(decoder_stage_channels[-1])
            self.visual_text_proj = nn.Sequential(
                nn.Conv3d(last_decoder_channels, contrast_dim, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv3d(contrast_dim, contrast_dim, kernel_size=1),
            )
            self.text_contrast_proj = nn.Sequential(
                nn.Linear(int(text_contrast_text_dim), contrast_dim),
                nn.SiLU(inplace=True),
                nn.Linear(contrast_dim, contrast_dim),
            )

    def _merge_text_generated_dynamic_slots(
        self,
        normal_logits: torch.Tensor,
        dynamic_slot_logits: torch.Tensor,
    ) -> torch.Tensor:
        start = self.text_generated_slot_start_channel
        end = start + self.text_generated_slot_num_slots
        if dynamic_slot_logits.shape[1] != self.text_generated_slot_num_slots:
            raise ValueError(
                f"dynamic_slot_logits has {dynamic_slot_logits.shape[1]} slots, "
                f"expected {self.text_generated_slot_num_slots}"
            )
        if self.text_generated_slot_mode == "replace":
            return torch.cat((normal_logits[:, :start], dynamic_slot_logits, normal_logits[:, end:]), dim=1)
        merged_dynamic = normal_logits[:, start:end] + self.text_generated_slot_alpha * dynamic_slot_logits
        return torch.cat((normal_logits[:, :start], merged_dynamic, normal_logits[:, end:]), dim=1)

    def _normalize_group_ids(self, x: torch.Tensor, group_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Normalize optional group IDs for forward compatibility with generic nnUNet
        inference entrypoints that only call `network(x)`.
        """
        if group_ids is None:
            return torch.full(
                (x.shape[0],),
                int(self.default_infer_group_id),
                dtype=torch.long,
                device=x.device,
            )
        return group_ids.reshape(-1).to(device=x.device, dtype=torch.long)

    def _forward_conditioned(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor],
        return_features: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        group_ids = self._normalize_group_ids(x, group_ids)
        skips = self.encode(x)
        return self.decode_from_skips(skips, group_ids, return_features=return_features)

    def encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        skips: List[torch.Tensor] = []
        for stage in self.encoder.stages:
            x = stage(x)
            skips.append(x)
        return skips

    def decode_from_skips(
        self,
        skips: List[torch.Tensor],
        group_ids: Optional[torch.Tensor],
        return_features: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        if group_ids is None:
            group_ids = torch.full(
                (skips[0].shape[0],),
                int(self.default_infer_group_id),
                dtype=torch.long,
                device=skips[0].device,
            )
        else:
            group_ids = group_ids.reshape(-1).to(device=skips[0].device, dtype=torch.long)
        cond_vec = self.condition_encoder(group_ids, batch_size=skips[0].shape[0], device=skips[0].device)

        lres_input = skips[-1]
        seg_outputs: List[torch.Tensor] = []
        decoder_last_feature: Optional[torch.Tensor] = None

        for stage_idx in range(len(self.decoder.stages)):
            x = self.decoder.transpconvs[stage_idx](lres_input)
            x = torch.cat((x, skips[-(stage_idx + 2)]), dim=1)
            x = self.decoder.stages[stage_idx](x)
            x = self.decoder_film[stage_idx](x, cond_vec)
            decoder_last_feature = x

            if self.decoder.deep_supervision:
                logits = self.decoder.seg_layers[stage_idx](x)
                if self.use_text_generated_slot_head:
                    dynamic_logits = self.text_generated_slot_heads[stage_idx](
                        x,
                        group_ids,
                        self.text_generated_slot_embeddings,
                        self.text_generated_slot_group_to_labels,
                    )
                    logits = self._merge_text_generated_dynamic_slots(logits, dynamic_logits)
                    self.latest_text_generated_slot_stats = dict(
                        self.text_generated_slot_heads[stage_idx].last_stats
                    )
                seg_outputs.append(logits)
            elif stage_idx == (len(self.decoder.stages) - 1):
                logits = self.decoder.seg_layers[-1](x)
                if self.use_text_generated_slot_head:
                    dynamic_logits = self.text_generated_slot_heads[-1](
                        x,
                        group_ids,
                        self.text_generated_slot_embeddings,
                        self.text_generated_slot_group_to_labels,
                    )
                    logits = self._merge_text_generated_dynamic_slots(logits, dynamic_logits)
                    self.latest_text_generated_slot_stats = dict(self.text_generated_slot_heads[-1].last_stats)
                seg_outputs.append(logits)

            lres_input = x

        seg_outputs = seg_outputs[::-1]
        output = seg_outputs if self.decoder.deep_supervision else seg_outputs[0]
        if return_features:
            if decoder_last_feature is None:
                raise RuntimeError("decoder_last_feature was not produced")
            return {"logits": output, "decoder_last": decoder_last_feature}
        return output

    def forward(
        self,
        x: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
        return_features: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return self._forward_conditioned(x, group_ids, return_features=return_features)


def get_main_output(output: Union[torch.Tensor, Sequence[torch.Tensor]]) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output
