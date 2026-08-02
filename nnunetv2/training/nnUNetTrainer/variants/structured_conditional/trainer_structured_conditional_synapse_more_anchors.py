from __future__ import annotations

import os

import torch

from . import dataloader_structured_conditional_synapse_variant as dataloader_variant
from . import inference_structured_conditional_synapse_variant as inference_variant
from . import label_mapping_synapse_more_anchors as label_mapping
from . import metrics_structured_conditional_synapse_variant as metrics_variant
from . import structured_loss_synapse_twoslot as loss_variant
from . import trainer_structured_conditional_synapse as base


def _patch_base_module() -> None:
    os.environ["NNUNET_STRUCTCOND_SYNAPSE_MAPPING_MODULE"] = label_mapping.__name__
    base.NUM_DYNAMIC_GROUPS = label_mapping.NUM_DYNAMIC_GROUPS
    base.NUM_OUTPUT_CHANNELS = label_mapping.NUM_OUTPUT_CHANNELS
    base.infer_present_groups_from_segmentation = label_mapping.infer_present_groups_from_segmentation
    base.sample_group_id_for_case = label_mapping.sample_group_id_for_case
    base.remap_original_to_structured = label_mapping.remap_original_to_structured
    base.StructuredConditionalDataLoader3D = dataloader_variant.StructuredConditionalDataLoader3D
    base.StructuredConditionalLoss = loss_variant.StructuredConditionalLoss
    base.StructuredLossConfig = loss_variant.StructuredLossConfig
    base.predict_logits_all_groups = inference_variant.predict_logits_all_groups
    base.predict_logits_for_group = inference_variant.predict_logits_for_group
    base.reconstruct_original_labels_from_all_groups = inference_variant.reconstruct_original_labels_from_all_groups
    base.empty_validation_accumulators = metrics_variant.empty_validation_accumulators
    base.compute_group_confusion_from_logits = metrics_variant.compute_group_confusion_from_logits
    base.build_validation_report = metrics_variant.build_validation_report


class nnUNetTrainerStructuredConditionalSynapseMoreAnchors(base.nnUNetTrainerStructuredConditionalSynapse):
    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
    ):
        _patch_base_module()
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
