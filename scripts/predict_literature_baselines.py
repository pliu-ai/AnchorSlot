"""One case per process; all-query fusion on the restored crop grid, then NIfTI export."""
import argparse
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerLiterature import (
    nnUNetTrainerUniSegCellMap, nnUNetTrainerCoNeMOSCellMap, nnUNetTrainerMISOStyle32)
from literature_baselines.models import TASKS, QueryView
from literature_baselines.prediction import merge_candidates, restore_grid, resample_logits

TRAINERS = dict(uniseg=nnUNetTrainerUniSegCellMap, conemos=nnUNetTrainerCoNeMOSCellMap, miso=nnUNetTrainerMISOStyle32)


def load_model(model, checkpoint, method, device):
    plans = PlansManager(json.loads((model/'plans.json').read_text()))
    configuration = plans.get_configuration('3d_lowres_large_patch_bs4')
    dataset = json.loads((model/'dataset.json').read_text())
    state = torch.load(model/'fold_0'/checkpoint, map_location='cpu', weights_only=False)
    assert state['trainer_name'] == TRAINERS[method].__name__
    net = TRAINERS[method].build_network_architecture(
        configuration.network_arch_class_name, configuration.network_arch_init_kwargs,
        configuration.network_arch_init_kwargs_req_import, 1, 32, False)
    net.load_state_dict(state['network_weights'], strict=True)
    net.eval().to(device)
    return net, plans, configuration, dataset, state


def predict_case(args):
    torch.set_num_threads(args.threads)
    start = time.perf_counter()
    device = torch.device(args.device)
    model = Path(args.model)
    net, plans, configuration, dataset, state = load_model(model, args.checkpoint, args.method, device)
    data, _, props = configuration.preprocessor_class(verbose=False).run_case(
        [args.input], None, plans, configuration, dataset)
    data = torch.from_numpy(data)
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not args.no_tta,
                                perform_everything_on_device=True, device=device, verbose=False,
                                verbose_preprocessing=False, allow_tqdm=False)
    predictor.manual_initialization(net, plans, configuration, None, dataset,
                                    state['trainer_name'], state['inference_allowed_mirroring_axes'])
    network_seconds = 0.0
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    queries = range(14) if args.method == 'uniseg' else range(31) if args.method == 'conemos' else [None]
    shape = props['shape_after_cropping_and_before_resampling']
    best_probability = np.zeros(shape, dtype=np.float32)
    segmentation = np.zeros(shape, dtype=np.uint8)
    for query in queries:
        if query is not None:
            predictor.network = QueryView(net, query)
            # The predictor uses only this field for accumulation allocation.
            from types import SimpleNamespace
            predictor.label_manager = SimpleNamespace(num_segmentation_heads=5 if args.method == 'uniseg' else 1)
        tick = time.perf_counter()
        logits = predictor.predict_sliding_window_return_logits(data)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        network_seconds += time.perf_counter() - tick
        logits = resample_logits(logits, props, plans, configuration)
        if not torch.isfinite(logits).all():
            raise FloatingPointError('Nonfinite prediction')
        if args.method == 'uniseg':
            labels = TASKS[query]
            p = logits[:len(labels)+1].softmax(0).numpy()
            merge_candidates(best_probability, segmentation, p[1:], labels, p[0])
        elif args.method == 'conemos':
            p = logits.sigmoid().numpy()
            merge_candidates(best_probability, segmentation, p, (query + 1,), 0.5)
        else:
            segmentation = logits.argmax(0).numpy().astype(np.uint8)
        del logits
        print('QUERY_COMPLETE', query, 'network_seconds', round(network_seconds, 2), flush=True)
    segmentation = restore_grid(segmentation, props, plans)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_name(out.name.removesuffix('.nii.gz') + '.writing.nii.gz')
    plans.image_reader_writer_class().write_seg(segmentation, str(temporary), props)
    os.replace(temporary, out)
    sha = hashlib.sha256()
    with (model/'fold_0'/args.checkpoint).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            sha.update(block)
    report = dict(method=args.method, checkpoint=args.checkpoint, checkpoint_sha256=sha.hexdigest(),
                  completed_epochs=state['current_epoch'], labels_present=np.unique(segmentation).tolist(),
                  shape=list(segmentation.shape), queries=len(queries), cached_encoder=False,
                  tta=not args.no_tta, network_seconds=network_seconds, total_seconds=time.perf_counter()-start,
                  parameters=sum(p.numel() for p in net.parameters()),
                  peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if device.type == 'cuda' else None,
                  gpu=torch.cuda.get_device_name() if device.type == 'cuda' else 'cpu')
    out.with_suffix('.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--method', choices=TRAINERS, required=True)
    p.add_argument('--model', required=True)
    p.add_argument('--checkpoint', default='checkpoint_final.pth')
    p.add_argument('--input', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--no-tta', action='store_true')
    predict_case(p.parse_args())
