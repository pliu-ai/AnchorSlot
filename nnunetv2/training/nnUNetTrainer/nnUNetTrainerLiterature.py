"""Three explicitly adapted CellMap controls, with the same data and update budget."""
import json
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from nnunetv2.training.nnUNetTrainer.nnUNetTrainerRalph import nnUNetTrainerRalph
import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_module
from literature_baselines.models import TASKS, UniSegCellMap, CoNeMOSCellMap
from literature_baselines.losses import deep_loss, valid_voxels, remap_task


class LiteratureTrainer(nnUNetTrainerRalph):
    METHOD = None

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.seed = int(os.environ.get('LITERATURE_SEED', '1234'))
        torch.manual_seed(self.seed + self.local_rank)
        np.random.seed(self.seed + self.local_rank)
        random.seed(self.seed + self.local_rank)
        self.num_iterations_per_epoch = int(os.environ.get('LITERATURE_TRAIN_ITERS', '250'))
        self.num_val_iterations_per_epoch = int(os.environ.get('LITERATURE_VAL_ITERS', '25'))
        self.save_every = 25
        self._step_index = 0
        self._val_index = 0
        self.query_exposure = np.zeros(14 if self.METHOD == 'uniseg' else 31, dtype=np.int64)
        self.supervised_voxels = 0
        meta = json.loads(Path(os.environ['LITERATURE_DATA_JSON']).read_text())
        self.train_cases = set(meta['train_cases'])
        self.annotations = meta['annotated_ids']
        self.present_ids = meta['present_ids']
        self.frequencies = {int(k): v for k, v in meta['annotated_crop_frequency'].items()}
        if set(self.train_cases) - set(self.annotations):
            raise ValueError('Missing annotation metadata')
        self.print_to_log_file('Literature adaptation', self.METHOD, 'seed', self.seed,
                               'budget', self.num_epochs * self.num_iterations_per_epoch,
                               'ignore padding and unknown labels; validation excludes crop126')

    @classmethod
    def build_network_architecture(cls, architecture_class_name, arch_init_kwargs,
                                   arch_init_kwargs_req_import, num_input_channels,
                                   num_output_channels, enable_deep_supervision=True):
        outputs = {'uniseg': 5, 'conemos': 1, 'miso': 32}[cls.METHOD]
        backbone = nnUNetTrainerRalph.build_network_architecture(
            architecture_class_name, arch_init_kwargs, arch_init_kwargs_req_import,
            num_input_channels, outputs, enable_deep_supervision)
        if cls.METHOD == 'uniseg':
            return UniSegCellMap(backbone)
        if cls.METHOD == 'conemos':
            return CoNeMOSCellMap(backbone)
        return backbone

    def _build_loss(self):
        return nn.Identity()  # train/validation_step use metadata-aware deep_loss

    def do_split(self):
        train, val = super().do_split()
        assert set(train) == self.train_cases
        val = [k for k in val if not k.endswith('_crop126')]
        assert len(train) == 202 and len(val) == 16
        assert set(val).isdisjoint(train)
        return train, val

    @staticmethod
    def _keep_invalid(transform):
        # nnU-Net ordinarily turns -1 padding into background; keep it ignored.
        transform.transforms = [t for t in transform.transforms
                                if t.__class__.__name__ != 'RemoveLabelTansform']
        return transform

    def get_training_transforms(self, *args, **kwargs):
        return self._keep_invalid(super().get_training_transforms(*args, **kwargs))

    @staticmethod
    def get_validation_transforms(*args, **kwargs):
        return LiteratureTrainer._keep_invalid(nnUNetTrainerRalph.get_validation_transforms(*args, **kwargs))

    def repeat_probabilities(self, keys):
        if not keys or not set(keys).issubset(self.train_cases):
            return None
        factors = [min(8.0, max([1.0] + [np.sqrt(0.5 / self.frequencies[c])
                   for c in self.present_ids[k] if self.frequencies[c] > 0])) for k in keys]
        factors = np.asarray(factors, dtype=np.float64)
        return factors / factors.sum()

    def get_dataloaders(self):
        if self.METHOD != 'miso':
            return super().get_dataloaders()
        original = base_module.nnUNetDataLoader3D
        def repeated_loader(data, *args, **kwargs):
            p = self.repeat_probabilities(list(data.keys()))
            if p is not None:
                kwargs['sampling_probabilities'] = p
                self.print_to_log_file('MISO repeat probabilities', 'min', float(p.min()), 'max', float(p.max()))
            return original(data, *args, **kwargs)
        base_module.nnUNetDataLoader3D = repeated_loader
        try:
            return super().get_dataloaders()
        finally:
            base_module.nnUNetDataLoader3D = original

    def annotation_tensor(self, keys):
        annotation = torch.zeros((len(keys), 32), dtype=torch.bool, device=self.device)
        for b, key in enumerate(keys):
            ids = self.annotations[key]  # fail closed on unknown cases
            annotation[b, ids] = True
            annotation[b, 0] = len(ids) == 31
        return annotation

    def choose_query(self, target, annotated, validation=False):
        if self.METHOD == 'miso':
            return None
        groups = TASKS if self.METHOD == 'uniseg' else tuple((c,) for c in range(1, 32))
        serial = self._val_index if validation else self._step_index
        rng = np.random.default_rng(self.seed + self.local_rank * 1000003 +
                                    self.current_epoch * 100003 + serial + (500000003 if validation else 0))
        queries = []
        for b in range(len(target)):
            present = set(int(c) for c in torch.unique(target[b]).tolist() if c > 0)
            eligible = [q for q, labels in enumerate(groups) if bool(annotated[b, list(labels)].all())]
            positives = [q for q in eligible if present.intersection(groups[q])]
            negatives = [q for q in eligible if q not in positives]
            pool = positives if rng.random() < 0.8 else negatives
            pool = pool or eligible
            if not pool:
                raise RuntimeError('No annotated task is available')
            queries.append(int(rng.choice(pool)))
        return torch.tensor(queries, device=self.device)

    def _batch(self, batch, validation=False):
        data = batch['data'].to(self.device, non_blocking=True)
        targets = batch['target']
        targets = [t.to(self.device, non_blocking=True) for t in targets] if isinstance(targets, list) else targets.to(self.device)
        highest = targets[0] if isinstance(targets, list) else targets
        annotation = self.annotation_tensor(batch['keys'])
        query = self.choose_query(highest, annotation, validation)
        return data, targets, annotation, query

    def train_step(self, batch):
        data, target, annotation, query = self._batch(batch)
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext():
            output = self.network(data, query) if query is not None else self.network(data)
            loss = deep_loss(self.METHOD, output, target, query, annotation)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f'{self.METHOD}: nonfinite loss')
        if self.grad_scaler is not None:
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        else:
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), 12)
            self.optimizer.step()
        self._step_index += 1
        if query is not None:
            for q in query.tolist():
                self.query_exposure[q] += 1
        highest = target[0] if isinstance(target, list) else target
        self.supervised_voxels += int(valid_voxels(highest, annotation).sum())
        if self._step_index <= 2:
            self.print_to_log_file('step diagnostic', 'loss', float(loss), 'grad_norm', float(norm),
                                   'query', None if query is None else query.tolist())
        return {'loss': loss.detach().cpu().numpy()}

    @torch.no_grad()
    def validation_step(self, batch):
        data, targets, annotation, query = self._batch(batch, True)
        with torch.autocast('cuda') if self.device.type == 'cuda' else nullcontext():
            outputs = self.network(data, query) if query is not None else self.network(data)
            loss = deep_loss(self.METHOD, outputs, targets, query, annotation)
        self._val_index += 1
        z = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        y = targets[0] if isinstance(targets, list) else targets
        valid = valid_voxels(y, annotation)
        tp, fp, fn = [np.zeros(31, dtype=np.float64) for _ in range(3)]
        for b in range(len(y)):
            if self.METHOD == 'uniseg':
                labels = TASKS[int(query[b])]
                pred = z[b, :len(labels)+1].argmax(0)
                masks = [(c, pred == j) for j, c in enumerate(labels, 1)]
            elif self.METHOD == 'conemos':
                masks = [(int(query[b]) + 1, z[b, 0] > 0)]
            else:
                pred = z[b].argmax(0)
                masks = [(c, pred == c) for c in range(1, 32) if annotation[b, c]]
            for c, predicted in masks:
                truth = y[b, 0] == c
                v = valid[b]
                tp[c-1] += int((predicted & truth & v).sum())
                fp[c-1] += int((predicted & ~truth & v).sum())
                fn[c-1] += int((~predicted & truth & v).sum())
        return dict(loss=loss.detach().cpu().numpy(), tp_hard=tp, fp_hard=fp, fn_hard=fn)

    def on_train_epoch_start(self):
        self._step_index = 0
        self._val_index = 0
        super().on_train_epoch_start()

    def save_checkpoint(self, filename):
        # Atomic rename prevents a timeout from corrupting the resume checkpoint.
        temporary = filename + '.writing'
        super().save_checkpoint(temporary)
        if self.local_rank == 0 and not self.disable_checkpointing:
            os.replace(temporary, filename)
            metadata = dict(method=self.METHOD, completed_epochs=self.current_epoch + 1,
                            optimizer_updates=(self.current_epoch + 1) * self.num_iterations_per_epoch,
                            query_exposure_rank0_since_resume=self.query_exposure.tolist(),
                            valid_voxels_rank0_since_resume=self.supervised_voxels,
                            seed=self.seed, global_batch_size=4,
                            peak_training_gpu_allocated_bytes=torch.cuda.max_memory_allocated() if self.device.type == 'cuda' else None,
                            device=torch.cuda.get_device_name() if self.device.type == 'cuda' else 'cpu')
            Path(filename + '.json').write_text(json.dumps(metadata, indent=2) + '\n')

    def perform_actual_validation(self, save_probabilities=False):
        self.print_to_log_file('Full-volume evaluation is a separate dependent job using predict.py and the pinned CellMap evaluator.')


class nnUNetTrainerUniSegCellMap(LiteratureTrainer):
    METHOD = 'uniseg'


class nnUNetTrainerCoNeMOSCellMap(LiteratureTrainer):
    METHOD = 'conemos'


class nnUNetTrainerMISOStyle32(LiteratureTrainer):
    METHOD = 'miso'
