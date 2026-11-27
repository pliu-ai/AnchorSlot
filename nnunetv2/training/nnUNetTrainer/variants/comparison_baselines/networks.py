"""Expose 32-channel inference while retaining method-specific training outputs."""
from __future__ import annotations

import torch
from torch import nn


class HierarchyUNet(nn.Module):
    """An unconditioned nnU-Net with SALT or HSSN prediction heads.

    ``forward(x)`` follows nnU-Net's inference contract. ``return_aux=True``
    returns raw node logits and (HSSN only) the feature embedding for its loss.
    HSSN uses the 1/4-resolution decoder feature to bound 3D projection memory.
    """

    def __init__(self, backbone: nn.Module, method: str):
        super().__init__()
        if method not in {"salt", "hssn"}:
            raise ValueError(f"Unknown hierarchy method: {method}")
        self.backbone = backbone
        self.method = method
        self.feature_stage = max(0, len(backbone.decoder.stages) - 3)
        if method == "hssn":
            from .hssn import HSSNProjectionHead
            channels = backbone.encoder.output_channels[-(self.feature_stage + 2)]
            self.projection = HSSNProjectionHead(channels)

    @property
    def decoder(self):
        return self.backbone.decoder

    def leaf_scores(self, logits):
        if self.method == "salt":
            from .salt import salt_leaf_log_probs
            return salt_leaf_log_probs(logits)
        from .hssn import hssn_leaf_scores
        return hssn_leaf_scores(logits)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = None
        if self.method == "salt":
            raw = self.backbone(x)
        else:
            skips = self.backbone.encoder(x)
            x = skips[-1]
            outputs = []
            for i, stage in enumerate(self.decoder.stages):
                x = self.decoder.transpconvs[i](x)
                x = stage(torch.cat((x, skips[-(i + 2)]), dim=1))
                if return_aux and i == self.feature_stage:
                    features = self.projection(x)
                if self.decoder.deep_supervision or i == len(self.decoder.stages) - 1:
                    outputs.append(self.decoder.seg_layers[i](x))
            raw = outputs[::-1] if self.decoder.deep_supervision else outputs[-1]
        if return_aux:
            return {"raw": raw, "features": features}
        if isinstance(raw, (list, tuple)):
            return [self.leaf_scores(o) for o in raw]
        return self.leaf_scores(raw)
