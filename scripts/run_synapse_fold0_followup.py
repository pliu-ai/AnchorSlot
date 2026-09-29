#!/usr/bin/env python3
"""Frozen, smoke-gated, fixed-budget fold-0 screen, with resumable short jobs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

BASE = Path('/projects/weilab/liupeng/runs/Anchorslot/synapse_consistency_20260906_v1')
DATA = Path('/projects/weilab/liupeng/data/nnUNet')
ARMS = ('control', 'no_suppression', 'lr003', 'all_query_control', 'reconstruction')
TRAINER = 'nnUNetTrainerSynapseReconstructionScreen'
MODEL = f'Dataset210_Synapse13/{TRAINER}__nnUNetPlans__3d_fullres/fold_0'
PYTHON = '/projects/weilab/liupeng/conda/envs/ssl_seg/bin/python'
NEW_FILES = (
    'nnunetv2/training/nnUNetTrainer/variants/structured_conditional/synapse_reconstruction.py',
    'nnunetv2/training/nnUNetTrainer/variants/structured_conditional/trainer_synapse_reconstruction.py',
    'nnunetv2/tests/test_synapse_reconstruction.py',
    'scripts/run_synapse_fold0_followup.py',
    'scripts/continue_synapse_masked.py',
    'scripts/snapshot_synapse_checkpoints.py',
    'scripts/tests/test_synapse_continuation.py',
    'scripts/tests/test_synapse_checkpoint_snapshot.py',
    'scripts/tests/test_synapse_followup_recovery.py',
)


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def digest(path):
    sha = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def verify(m):
    for path, expected in m['locked_files'].items():
        if digest(path) != expected:
            raise ValueError(f'Locked input changed: {path}')
    source = Path(m['source'])
    expected = read(Path(m['root']) / 'source_sha256.json')
    actual = {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')}
    if actual != expected:
        raise ValueError('Frozen source modified or files added/removed')


def arm_config(arm):
    if arm not in ARMS:
        raise ValueError(arm)
    return dict(initial_lr=.003 if arm == 'lr003' else .001,
                lambda_suppress=0. if arm == 'no_suppression' else .1,
                lambda_reconstruction=.25 if arm == 'reconstruction' else 0.,
                all_query_every=4 if arm in ('all_query_control', 'reconstruction') else None)


def training_contract(m, arm):
    return dict(arm=arm, configuration=m['arms'][arm], seed=m['seed'], epochs=m['epochs'],
                schedule_horizon=m['schedule_horizon'], source=m['source'], fold=0,
                train_cases=m['reference']['train_cases'], validation_cases=m['reference']['validation_cases'])


def recover_inputs(m, parent_root):
    """Snapshot completed runs without modifying the original failed screen."""
    import torch
    from snapshot_synapse_checkpoints import stable_copy
    parent_root = parent_root.resolve()
    parent = read(parent_root / 'submission.json')
    verify(parent)
    for key in ('seed', 'epochs', 'schedule_horizon', 'iterations', 'val_iterations', 'arms', 'reference'):
        if parent[key] != m[key]:
            raise ValueError(f'Recovery experiment differs: {key}')
    root = Path(m['root'])
    recovery = dict(parent_root=str(parent_root), arms={})
    for arm in ARMS:
        if arm in ('all_query_control', 'reconstruction'):
            recovery['arms'][arm] = dict(completed_epochs=0, action='restart from original random seed; failed epoch-1 run preserved')
            continue
        old = parent_root / 'models' / arm / MODEL
        if read(old / 'experiment_contract.json') != training_contract(parent, arm):
            raise ValueError(f'Parent contract mismatch: {arm}')
        status = read(old / 'segment_status.json')
        if status['completed_epochs'] != 50 or status['screen_complete']:
            raise ValueError(f'Unexpected recovery boundary: {arm}')
        snapshots = root / 'recovery_inputs' / arm
        snapshots.mkdir(parents=True)
        folder = root / 'models' / arm / MODEL
        folder.mkdir(parents=True)
        files = {}
        for policy in ('latest', 'best'):
            name = f'checkpoint_{policy}.pth'
            snapshot = snapshots / name
            info = stable_copy(old / name, snapshot)
            m['locked_files'][str(snapshot)] = info['sha256']
            shutil.copy2(snapshot, folder / name)
            if digest(folder / name) != info['sha256']:
                raise ValueError('Recovery copy checksum mismatch')
            files[policy] = dict(source=str(old / name), snapshot=str(snapshot), **info)
        saved = torch.load(snapshots / 'checkpoint_latest.pth', map_location='cpu', weights_only=False)
        if saved['current_epoch'] != 50 or not saved['optimizer_state']['state']:
            raise ValueError(f'Incomplete recovery checkpoint: {arm}')
        if len(saved['logging']['train_losses']) != 50:
            raise ValueError(f'Incomplete recovery logging: {arm}')
        save(folder / 'experiment_contract.json', training_contract(m, arm))
        save(folder / 'segment_status.json', status)
        recovery['arms'][arm] = dict(completed_epochs=50, action='resume exact network, optimizer, EMA and logger', files=files)
        del saved
    m['recovery'] = recovery


def prepare(args):
    root = args.root.resolve()
    if root.exists():
        raise FileExistsError('Use a new run root; will not overwrite or duplicate jobs')
    repo = Path(__file__).resolve().parents[1]
    ref = BASE / 'models/Dataset210_Synapse13/nnUNetTrainerSynapseMasked__nnUNetPlans__3d_fullres/fold_0/experiment_contract.json'
    contract = read(ref)
    if contract['fold'] != 0 or contract['arm'] != 'masked':
        raise ValueError('Wrong reference contract')
    split_path = DATA / 'nnUNet_preprocessed/Dataset210_Synapse13/splits_final.json'
    split = read(split_path)[0]
    if split != {'train': contract['train_cases'], 'val': contract['validation_cases']}:
        raise ValueError('Split changed')
    if len(split['train']) != 24 or len(split['val']) != 6 or set(split['train']) & set(split['val']):
        raise ValueError('Wrong split or train/validation overlap')
    for name, sha in read(BASE / 'source_sha256.json').items():
        if name.endswith('.py') and digest(BASE / 'source' / name) != sha:
            raise ValueError(f'Parent source changed: {name}')
    for name in NEW_FILES:
        if not (repo / name).is_file():
            raise FileNotFoundError(name)
    root.mkdir(parents=True)
    (root / 'logs').mkdir()
    source = root / 'source'
    shutil.copytree(BASE / 'source', source, ignore=shutil.ignore_patterns('__pycache__', '*.pyc', '.pytest_cache'))
    for name in NEW_FILES:
        (source / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / name, source / name)
    shutil.copy2(__file__, root / 'runner.py')
    shutil.copy2(repo / 'scripts/slurm/synapse_fold0_followup.sbatch', root / 'job.sbatch')
    save(root / 'source_sha256.json', {str(p.relative_to(source)): digest(p) for p in source.rglob('*.py')})
    locks = [ref, split_path, DATA / 'nnUNet_preprocessed/Dataset210_Synapse13/nnUNetPlans.json',
             DATA / 'nnUNet_raw/Dataset210_Synapse13/dataset.json', root / 'runner.py', root / 'job.sbatch',
             root / 'source_sha256.json']
    m = dict(root=str(root), source=str(source), data_root=str(DATA), created_utc=datetime.now(timezone.utc).isoformat(),
             reference=contract, seed=1234, epochs=250, schedule_horizon=1000, iterations=250,
             val_iterations=50, arms={arm: arm_config(arm) for arm in ARMS}, jobs={},
             locked_files={str(p): digest(p) for p in locks},
             evaluation='same six cases; frozen corrected evaluator; native and preprocessed; no TTA or PP',
             endpoint='epoch 250 final primary; online-EMA best within screen secondary',
             initialization='same random network initialization; no checkpoint or teacher warm start',
             caveat='250-epoch from-scratch SCREEN using the first 250 epochs of a 1000-epoch LR schedule. '
                    'Not comparable as a final winner against trained 732/1000/2000-epoch models. '
                    'All-query control and reconstruction share query count but exceed single-query compute. '
                    'Same six reused development scans; no independent-test or statistical-superiority claim. '
                    'Segment restarts reseed augmentation; identical restart boundaries across all arms.')
    if args.recover_from is not None:
        recover_inputs(m, args.recover_from)
    save(root / 'submission.json', m)
    verify(m)
    if args.submit:
        submit_jobs(m)
    print(json.dumps({'root': str(root), 'jobs': m['jobs']}, indent=2))


def submit_jobs(m):
    root = Path(m['root'])
    if m['jobs']:
        raise FileExistsError('Jobs already submitted; do not duplicate')
    def submit(name, phase, flags):
        cmd = ['sbatch', '--parsable', '--export=ALL', f'--job-name={name}',
               f'--output={root}/logs/%x_%A_%a.out', f'--error={root}/logs/%x_%A_%a.err',
               *flags, str(root / 'job.sbatch')]
        env = dict(os.environ, FOLLOWUP_DIR=str(root), PHASE=phase)
        jid = subprocess.check_output(cmd, env=env, text=True).strip().split(';')[0]
        m['jobs'][name] = {'job_id': jid, 'phase': phase, 'command': cmd}
        save(root / 'submission.json', m)
        return jid
    smoke = submit('as_f0_screen_smoke', 'smoke', ['--time=01:00:00'])
    # Each task checkpoints/requeues itself every 50 epochs. A single array
    # preserves the global two-training-GPU cap without whole-array barriers.
    previous = submit('as_f0_screen_train', 'advance',
                      ['--array=0-4%2', '--requeue', '--open-mode=append', f'--dependency=afterok:{smoke}'])
    evaluation = submit('as_f0_screen_eval', 'evaluate',
                        ['--array=0-4%2', f'--dependency=aftercorr:{previous}'])
    submit('as_f0_screen_collect', 'collect', ['--gres=none', '--mem=4G', '--cpus-per-task=1',
           '--time=00:10:00', f'--dependency=afterany:{evaluation}'])


def environment(m, arm, smoke=False):
    for key in list(os.environ):
        if key.startswith(('NNUNET_STRUCTCOND_', 'SYNAPSE_SCREEN_')):
            del os.environ[key]
    os.environ.update(m['reference']['runtime_environment'])
    source, root, data = Path(m['source']), Path(m['root']), Path(m['data_root'])
    os.environ.update({'PYTHONPATH': str(source), 'nnUNet_raw': str(data / 'nnUNet_raw'),
                       'nnUNet_preprocessed': str(data / 'nnUNet_preprocessed'),
                       'nnUNet_results': str(root / ('smoke_models' if smoke else 'models') / arm),
                       'nnUNet_compile': 'f', 'NNUNET_USE_WANDB': '0', 'nnUNet_n_proc_DA': '0',
                       'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '4',
                       'SYNAPSE_SCREEN_ARM': arm,
                       'NNUNET_STRUCTCOND_NUM_EPOCHS': str(m['epochs']),
                       'NNUNET_STRUCTCOND_INITIAL_LR': str(m['arms'][arm]['initial_lr']),
                       'NNUNET_STRUCTCOND_LAMBDA_SUPPRESS': str(m['arms'][arm]['lambda_suppress'])})
    sys.path[:0] = [str(source), str(source / 'scripts')]


def make_trainer(m, arm):
    import torch
    from run_synapse_consistency import seed_everything
    from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
    from nnunetv2.training.nnUNetTrainer.variants.structured_conditional.trainer_synapse_reconstruction import nnUNetTrainerSynapseReconstructionScreen
    seed_everything(m['seed'])
    torch.set_num_threads(4)
    data = Path(m['data_root'])
    plans = read(data / 'nnUNet_preprocessed/Dataset210_Synapse13/nnUNetPlans.json')
    dataset = read(data / 'nnUNet_raw/Dataset210_Synapse13/dataset.json')
    t = nnUNetTrainerSynapseReconstructionScreen(plans, '3d_fullres', 0, dataset, device=torch.device('cuda'))
    t.num_epochs, t.num_iterations_per_epoch, t.num_val_iterations_per_epoch = m['epochs'], m['iterations'], m['val_iterations']
    t.oversample_foreground_percent, t.save_every = .66, 25
    t.initialize()
    t.lr_scheduler = PolyLRScheduler(t.optimizer, t.initial_lr, m['schedule_horizon'])
    if t.batch_size != 2 or list(t.configuration_manager.patch_size) != [48, 192, 192]:
        raise ValueError('Unexpected batch or patch size')
    return t


def network_digest(network):
    sha = hashlib.sha256()
    for name, value in network.state_dict().items():
        sha.update(name.encode())
        sha.update(value.detach().cpu().contiguous().numpy().tobytes())
    return sha.hexdigest()


def smoke_arm(m, selected_arm):
    import time
    import torch
    from run_synapse_consistency import finish_loaders
    from continue_synapse_masked import assert_state_equal
    rows = []
    for arm in (selected_arm,):
        environment(m, arm, True)
        t = make_trainer(m, arm)
        start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        init_sha = network_digest(t.network)
        if rows and init_sha != rows[0]['initial_network_sha256']:
            raise AssertionError('Arms do not have identical initial weights')
        try:
            t.dataloader_train, t.dataloader_val = t.get_dataloaders()
            before = next(t.network.parameters()).detach().clone()
            outputs, epoch_keys = [], []
            for epoch in (0, 1):
                assert t.current_epoch == epoch
                t.on_epoch_start()
                t.on_train_epoch_start()
                # Keep the actual 250-step global query schedule. Four updates
                # cover an entire query cycle at both epoch-start phases.
                values = [t.train_step(next(t.dataloader_train)) for _ in range(4)]
                if not all(set(x) == set(values[0]) for x in values):
                    raise AssertionError('Train output schema differs within epoch')
                epoch_keys.append(sorted(values[0]))
                outputs.extend(values)
                t.on_train_epoch_end(values)
                with torch.no_grad():
                    t.on_validation_epoch_start()
                    val = t.validation_step(next(t.dataloader_val))
                    t.on_validation_epoch_end([val])
                t.on_epoch_end()
            if torch.equal(before, next(t.network.parameters()).detach()):
                raise AssertionError('Encoder did not update')
            if not all(torch.isfinite(torch.as_tensor(x['loss'])) for x in outputs + [val]):
                raise FloatingPointError('Nonfinite smoke loss')
            path = Path(t.output_folder) / 'checkpoint_smoke.pth'
            t.current_epoch -= 1
            t.save_checkpoint(str(path))
            t.current_epoch += 1
            saved = torch.load(path, map_location='cpu', weights_only=False)
            with torch.no_grad():
                next(t.network.parameters()).add_(1)
            t.load_checkpoint(saved)
            assert_state_equal(t.network.state_dict(), saved['network_weights'])
            assert_state_equal(t.optimizer.state_dict(), saved['optimizer_state'])
            assert_state_equal(t.logger.get_checkpoint(), saved['logging'])
            assert t.current_epoch == 2
            restored_epoch = None
            recovery = m.get('recovery', {}).get('arms', {}).get(arm, {})
            if recovery.get('completed_epochs') == 50:
                recovered = torch.load(recovery['files']['latest']['snapshot'], map_location='cpu', weights_only=False)
                t.load_checkpoint(recovered)
                assert_state_equal(t.network.state_dict(), recovered['network_weights'])
                assert_state_equal(t.optimizer.state_dict(), recovered['optimizer_state'])
                assert_state_equal(t.logger.get_checkpoint(), recovered['logging'])
                assert t._best_ema == recovered['_best_ema']
                assert t.current_epoch == 50
                restored_epoch = 50
                del recovered
            row = dict(arm=arm, initial_network_sha256=init_sha, encoder_updated=True, reload_exact=True,
                       smoke_epochs=2, updates_per_smoke_epoch=4, query_schedule_iterations=250,
                       epoch_output_keys=epoch_keys, recovery_restore_epoch=restored_epoch,
                       train_losses=[float(x['loss']) for x in outputs], validation_loss=float(val['loss']),
                       peak_gpu_gb=torch.cuda.max_memory_allocated()/1e9, seconds=time.monotonic()-start)
            rows.append(row)
            print(json.dumps(row), flush=True)
        finally:
            finish_loaders(t)
        del t, before, outputs, saved
        gc.collect()
        torch.cuda.empty_cache()
    save(Path(m['root']) / f'smoke_{selected_arm}.json', rows[0])


def smoke(m):
    # nnU-Net caches environment paths at module import. Separate processes also
    # prevent CUDA/augmentation RNG state from leaking between smoke arms.
    root = Path(m['root'])
    rows = []
    for arm in ARMS:
        subprocess.run([PYTHON, str(root / 'runner.py'), '--root', str(root),
                        '--phase', 'smoke-arm', '--arm', arm], check=True)
        rows.append(read(root / f'smoke_{arm}.json'))
    if len({r['initial_network_sha256'] for r in rows}) != 1:
        raise AssertionError('Arms did not start from identical random weights')
    save(root / 'smoke_passed.json', {'passed': True, 'rows': rows, 'runner_sha256': digest(__file__)})


def train(m, arm, stop):
    import torch
    from run_synapse_consistency import finish_loaders
    from continue_synapse_masked import assert_state_equal
    root = Path(m['root'])
    gate = read(root / 'smoke_passed.json')
    if not gate['passed'] or gate['runner_sha256'] != digest(__file__):
        raise ValueError('Exact runner has not passed GPU smoke')
    if stop not in (50, 100, 150, 200, 250):
        raise ValueError('Unplanned segment endpoint')
    t = make_trainer(m, arm)
    folder = Path(t.output_folder)
    contract = training_contract(m, arm)
    cp = folder / 'experiment_contract.json'
    latest, final = folder / 'checkpoint_latest.pth', folder / 'checkpoint_final.pth'
    if cp.exists():
        if read(cp) != contract:
            raise ValueError('Training contract mismatch')
        if final.exists():
            print('Already completed; no resubmission', flush=True)
            return
        if not latest.exists():
            raise FileNotFoundError('Existing run lacks checkpoint: will not restart silently')
        saved = torch.load(latest, map_location='cpu', weights_only=False)
        t.load_checkpoint(saved)
        assert_state_equal(t.network.state_dict(), saved['network_weights'])
        assert_state_equal(t.optimizer.state_dict(), saved['optimizer_state'])
        assert_state_equal(t.logger.get_checkpoint(), saved['logging'])
        if t._best_ema != saved['_best_ema']:
            raise ValueError('Best EMA not restored')
        if t.current_epoch != stop - 50:
            raise ValueError('Resume epoch does not match planned segment boundary')
    else:
        if stop != 50:
            raise FileNotFoundError('Predecessor missing; will not initialize a later segment')
        save(cp, contract)
    t.lr_scheduler.step(t.current_epoch)
    try:
        t.on_train_start()
        while t.current_epoch < stop:
            t.on_epoch_start()
            t.on_train_epoch_start()
            outputs = [t.train_step(next(t.dataloader_train)) for _ in range(t.num_iterations_per_epoch)]
            t.on_train_epoch_end(outputs)
            with torch.no_grad():
                t.on_validation_epoch_start()
                values = [t.validation_step(next(t.dataloader_val)) for _ in range(t.num_val_iterations_per_epoch)]
                t.on_validation_epoch_end(values)
            t.on_epoch_end()
        if stop == m['epochs']:
            t.on_train_end()
        else:
            t.current_epoch -= 1
            t.save_checkpoint(str(latest))
            t.current_epoch += 1
        save(folder / 'segment_status.json', {'completed_epochs': t.current_epoch, 'screen_complete': stop == m['epochs'],
                                            'lr': t.optimizer.param_groups[0]['lr']})
    finally:
        finish_loaders(t)


def next_endpoint(completed, total=250):
    if completed not in range(0, total + 1, 50):
        raise ValueError('Not a completed 50-epoch segment; refusing silent partial recovery')
    return None if completed == total else completed + 50


def advance(m, arm):
    folder = Path(m['root']) / 'models' / arm / MODEL
    status = folder / 'segment_status.json'
    completed = read(status)['completed_epochs'] if status.exists() else 0
    stop = next_endpoint(completed, m['epochs'])
    if stop is None:
        if not (folder / 'checkpoint_final.pth').is_file():
            raise FileNotFoundError('Final checkpoint missing')
        return
    train(m, arm, stop)
    result = read(status)
    if result['completed_epochs'] != stop:
        raise ValueError('Segment did not reach its endpoint')
    if not result['screen_complete']:
        print(f'{arm}: saved epoch {stop}; requesting independent requeue', flush=True)
        # Deliberate code consumed by the batch wrapper, never retry failures.
        raise SystemExit(75)


def evaluate(m, arm):
    root = Path(m['root'])
    for policy in ('final', 'best'):
        cp = root / 'models' / arm / MODEL / f'checkpoint_{policy}.pth'
        output = root / 'evaluation' / arm / policy
        if (output / 'summary.json').exists():
            raise FileExistsError('Completed evaluation will not be overwritten')
        subprocess.run([PYTHON, str(Path(m['source'])/'scripts/evaluate_synapse_consistency.py'),
                        '--checkpoint', str(cp), '--fold', '0', '--data-root', m['data_root'],
                        '--mode', 'masked', '--native', '--output', str(output)], check=True)


def collect(m):
    root = Path(m['root'])
    rows, missing = [], []
    from snapshot_synapse_checkpoints import PROTOCOL_KEYS
    reference = read(BASE / 'evaluation_fold0/masked/final/summary.json')['manifest']
    for arm in ARMS:
        for policy in ('final', 'best'):
            p = root / 'evaluation' / arm / policy / 'summary.json'
            if not p.exists():
                missing.append(f'{arm}/{policy}')
                continue
            report = read(p)
            meta = report['manifest']
            if {k: meta[k] for k in PROTOCOL_KEYS} != {k: reference[k] for k in PROTOCOL_KEYS}:
                raise ValueError('Evaluation protocol changed')
            cp = root / 'models' / arm / MODEL / f'checkpoint_{policy}.pth'
            if digest(cp) != meta['checkpoint_sha256'] or meta['fold'] != 0 or meta['mode'] != 'masked':
                raise ValueError('Checkpoint identity mismatch')
            if policy == 'final' and meta['epoch'] != m['epochs']:
                raise ValueError('Wrong screen endpoint')
            native = read(p.parent / 'masked/summary_native.json')['foreground_mean']
            metric = report['metrics']['masked']
            rows.append(dict(arm=arm, policy=policy, epoch=meta['epoch'], preprocessed_dice=metric['mean_dice'],
                             preprocessed_iou=metric['mean_iou'], native_dice=native['Dice'], native_iou=native['IoU']))
    result = dict(complete=not missing, missing=missing, rows=rows, caveat=m['caveat'])
    save(root / 'comparison_no_tta.json', result)
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--phase', default='prepare', choices=('prepare','submit','smoke','smoke-arm','train','advance','evaluate','collect'))
    p.add_argument('--arm', choices=ARMS, default='control')
    p.add_argument('--stop-epoch', type=int, default=50)
    p.add_argument('--submit', action='store_true')
    p.add_argument('--recover-from', type=Path)
    args = p.parse_args()
    if args.phase == 'prepare':
        prepare(args)
        return
    m = read(args.root/'submission.json')
    verify(m)
    if args.phase == 'submit':
        submit_jobs(m)
        print(json.dumps(m['jobs'], indent=2))
        return
    environment(m, args.arm, args.phase.startswith('smoke'))
    if args.phase == 'smoke': smoke(m)
    elif args.phase == 'smoke-arm': smoke_arm(m, args.arm)
    elif args.phase == 'train': train(m, args.arm, args.stop_epoch)
    elif args.phase == 'advance': advance(m, args.arm)
    elif args.phase == 'evaluate': evaluate(m, args.arm)
    else: collect(m)


if __name__ == '__main__':
    main()
