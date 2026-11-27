#!/usr/bin/env python3
"""Predict with a group-head AnchorSlot model and dump compact probabilities for offline merges.

Usage: predict_dump_gh.py DUMP_DIR -- <predict_structured_conditional.py arguments>
Requires NNUNET_STRUCTCOND_APPEND_GROUP_LOGITS=1 so the network appends its 18 group-head logits to the 9 structured
channels; the unchanged sliding-window predictor then carries 27 channels. This wrapper splits them:
  * per-group exports and the online merged prediction use the 9 structured channels only (legacy merge);
  * DUMP_DIR/<case>.npy (float16, 63 x D x H x W at 4 nm): channels 0-44 as in the AnchorSlot dump
    (mean background, mean anchors, per group: slots + other), channels 45-62 the group-head softmax averaged over
    the 13 query runs. DUMP_DIR/channels.json describes every channel; <case>.pkl holds the export properties.
For a model without a group head (9 channels) only the 45 compact channels are written.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

N_MAIN = 9


def run() -> None:
    dump = Path(sys.argv[1]).resolve()
    assert sys.argv[2] == "--", "usage: predict_dump_gh.py DUMP_DIR -- <predictor args>"
    sys.argv = [sys.argv[0]] + sys.argv[3:]
    code = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0])
    sys.path.insert(0, str(code / "scripts"))
    import predict_structured_conditional as psc
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.label_mapping_er_dynamic import (
        BACKGROUND_CHANNEL, FIXED_OUTPUT_TO_ORIGINAL, NUM_DYNAMIC_GROUPS, OTHER_CHANNEL,
        get_conditional_channel_to_original_label)
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_structured_conditional_er_dynamic_group_head import (
        NUM_GROUP_HEAD_CLASSES)

    dump.mkdir(parents=True, exist_ok=True)
    fixed = sorted(FIXED_OUTPUT_TO_ORIGINAL.items())
    channels = [dict(index=0, group=-1, kind="background_mean", label=0)]
    channels += [dict(index=1 + i, group=-1, kind="anchor_mean", label=int(lab)) for i, (_, lab) in enumerate(fixed)]
    for g in range(NUM_DYNAMIC_GROUPS):
        for ch, lab in sorted(get_conditional_channel_to_original_label(g).items()):
            channels.append(dict(index=len(channels), group=g, kind="slot", label=int(lab), output_channel=int(ch)))
        channels.append(dict(index=len(channels), group=g, kind="other", label=-1, output_channel=int(OTHER_CHANNEL)))
    n_compact = len(channels)
    head_channels = [dict(index=n_compact + k, group=-1, kind="group_head", label=k) for k in range(NUM_GROUP_HEAD_CLASSES)]
    state = {"case": None, "has_head": None}

    orig_one = psc.StructuredConditionalPredictor._decode_one_group_logits_to_output_labels
    orig_all = psc.StructuredConditionalPredictor._decode_all_groups_merged_original

    def one_group(self, logits, props, group_id, num_processes_segmentation_export):
        return orig_one(self, logits[:N_MAIN], props, group_id, num_processes_segmentation_export)

    def all_groups(self, logits_by_group, props, num_processes_segmentation_export):
        n_ch = int(next(iter(logits_by_group.values())).shape[1])
        has_head = n_ch == N_MAIN + NUM_GROUP_HEAD_CLASSES
        if not has_head and n_ch != N_MAIN:
            raise RuntimeError(f"unexpected logit channels {n_ch}")
        if state["has_head"] is None:
            state["has_head"] = has_head
            (dump / "channels.json").write_text(json.dumps(channels + (head_channels if has_head else []), indent=1) + "\n")
        main = {g: l[:, :N_MAIN] for g, l in logits_by_group.items()}
        probs = {g: torch.softmax(l[0].float(), dim=0) for g, l in main.items()}
        out = [torch.stack([probs[g][BACKGROUND_CHANNEL] for g in range(NUM_DYNAMIC_GROUPS)]).mean(0)]
        for ch, _ in fixed:
            out.append(torch.stack([probs[g][ch] for g in range(NUM_DYNAMIC_GROUPS)]).mean(0))
        for c in channels[5:]:
            out.append(probs[c["group"]][c["output_channel"]])
        if has_head:
            head = torch.stack([torch.softmax(l[0, N_MAIN:].float(), dim=0) for l in logits_by_group.values()]).mean(0)
            out.extend(list(head))
        arr = torch.stack(out).to(torch.float16).numpy()
        np.save(dump / f"{state['case']}.npy", arr)
        with open(dump / f"{state['case']}.pkl", "wb") as f:
            pickle.dump(props, f)
        print(f"[dump] {state['case']}: {arr.shape} float16 (group head: {has_head})", flush=True)
        del probs, out, arr
        return orig_all(self, main, props, num_processes_segmentation_export)

    psc.StructuredConditionalPredictor._decode_one_group_logits_to_output_labels = one_group
    psc.StructuredConditionalPredictor._decode_all_groups_merged_original = all_groups
    builtin_print = print

    def capture_print(*args, **kwargs):
        if args and isinstance(args[0], str) and args[0].startswith("\nPredicting "):
            state["case"] = args[0].split("Predicting ", 1)[1].strip()
        return builtin_print(*args, **kwargs)

    psc.print = capture_print
    psc.main()


if __name__ == "__main__":
    run()
