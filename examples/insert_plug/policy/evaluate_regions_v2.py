"""Evaluate one ACT checkpoint on 100 fresh InD and 100 central-square OOD layouts.

Geometry is frozen before execution. Every layout receives one episode; failed
and interrupted attempts are retained rather than resampled or silently retried.
Spawned workers reuse their loaded policy, resetting it for every new scene.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from copy import deepcopy
import csv
from datetime import datetime
import fcntl
import json
from itertools import zip_longest
import math
import multiprocessing
import os
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

import yaml

from mujoco_tools.paths import DATA_DIR, WORKSPACE_ROOT, checkpoint_dataset_path, relocated_checkpoint_path, relocated_evaluation_path
from . import evaluate, summarize


SPLITS = ('ind', 'ood')
EPISODES = 100
N_ACTION_STEPS = 32
DURATION = 60.0
_WORKER_POLICY = None
RUNTIME_SOURCES = (
    'examples/insert_plug/policy/evaluate_regions_v2.py',
    'examples/insert_plug/policy/rollout.py',
    'examples/insert_plug/policy/summarize.py',
    'examples/insert_plug/environment/evaluation_layouts_v2.py',
    'examples/insert_plug/environment/scene.py',
    'examples/insert_plug/environment/simulation.py',
    'src/mujoco_tools/lerobot_policy.py',
    'src/mujoco_tools/cameras.py',
    'third_party/lerobot/src/lerobot/policies/act/modeling_act.py',
    'third_party/lerobot/src/lerobot/policies/act/configuration_act.py',
    'third_party/lerobot/src/lerobot/policies/rgbd.py',
)


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read_json(path):
    return summarize.load_json(path)


def read_run_record(path):
    """Use current locations in memory while preserving fingerprinted JSON files."""
    record = read_json(path)
    for item in (record, record.get('reference_evaluation', {})):
        if isinstance(item.get('output'), str):
            item['output'] = str(relocated_evaluation_path(item['output']))
    return record


def episode_directory(configuration, row):
    return Path(configuration['output']) / 'episodes' / f"episode_{row['episode_index']:06d}"


def expected_scene(configuration, row):
    scene = deepcopy(configuration['scene_snapshot'])
    scene['board'].update(source_xy=row['source_xy'], target_xy=row['target_xy'])
    return scene


def validate_policy(saved):
    if saved.get('type') != 'act' or saved.get('chunk_size') != 64:
        raise ValueError('This evaluation requires an ACT checkpoint with chunk_size=64')
    if saved.get('temporal_ensemble_coeff') is not None:
        raise ValueError('This evaluation requires ACT temporal ensembling to be disabled')
    expected = {f'observation.images.{camera}': f'observation.depth.{camera}'
                for camera in ('head', 'wrist')}
    if saved.get('depth_camera_map') != expected or saved.get('depth_unit') != 'm':
        raise ValueError('This evaluation requires paired head/wrist metric RGB-D inputs')
    return evaluate.policy_settings(saved, N_ACTION_STEPS)


def validate_training_dataset(checkpoint):
    training = read_json(Path(checkpoint) / 'train_config.json')
    dataset = training['dataset']
    expected = (DATA_DIR / 'insert_plug_v2/lerobot').resolve()
    if checkpoint_dataset_path(dataset['root']) != expected or dataset.get('episodes') is not None:
        raise ValueError('Checkpoint must be trained on the complete insert_plug_v2 dataset without episode filters')
    return dataset


def prepare(args):
    """Freeze both manifests and all run contracts before any policy is loaded."""
    from mujoco_tools.lerobot_policy import resolve_checkpoint
    from ..environment import evaluation_layouts_v2 as layouts
    from ..environment.scene import load_config

    checkpoint = resolve_checkpoint(evaluate.workspace_path(args.checkpoint))
    saved = read_json(checkpoint / 'config.json')
    settings = validate_policy(saved)
    training_dataset = validate_training_dataset(checkpoint)
    scene_path = evaluate.workspace_path(args.scene_config)
    training_path = evaluate.workspace_path(args.training_manifest)
    scene = load_config(scene_path)
    if scene['task']['control_hz'] != 10 or scene['physics']['timestep'] != .001:
        raise ValueError('Expected 10 Hz control and 1 kHz physics')
    started = now()
    outputs = {split: evaluate.workspace_path(getattr(args, f'{split}_output') or
               Path(args.output_root) / evaluate.batch_output_path('act_insert_plug_v2_rgbd', split, started))
               for split in SPLITS}
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a)
           for a, b in [(outputs['ind'], outputs['ood'])]):
        raise ValueError('InD and OOD directories must be separate, nonnested directories')
    recorded_root = evaluate.workspace_path(training_dataset['root'])
    for path in outputs.values():
        for protected in (checkpoint, recorded_root, checkpoint_dataset_path(recorded_root)):
            if path == protected or path.is_relative_to(protected):
                raise ValueError('Evaluation output cannot be inside the checkpoint or training dataset')
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Output exists; use explicit --resume to preserve completed episodes: {path}')
    manifests = {
        split: layouts.make_manifest(scene, scene_path, training_path, split=split,
                                     seed=getattr(args, f'{split}_seed'))
        for split in SPLITS
    }
    for manifest in manifests.values():
        layouts.validate_manifest(manifest, scene, scene_path, training_path)
    checkpoint_files = {str(path.relative_to(checkpoint)): evaluate.fingerprint(path)
                        for path in sorted(checkpoint.rglob('*')) if path.is_file()}
    runtime_sources = {relative: evaluate.fingerprint(WORKSPACE_ROOT / relative)
                       for relative in RUNTIME_SOURCES}
    configurations = {}
    for split, output in outputs.items():
        output.mkdir(parents=True, exist_ok=False)
        (output / 'configs').mkdir()
        (output / 'episodes').mkdir()
        (output / 'configs/scene_config.yaml').write_bytes(scene_path.read_bytes())
        (output / 'configs/training_manifest.yaml').write_bytes(training_path.read_bytes())
        manifest_path = output / 'configs/positions_config.yaml'
        layouts.write_manifest(manifest_path, manifests[split])
        configuration = dict(
            format_version=1, protocol='insert_plug_v2_regions_100x1', started_at=started,
            output=str(output), evaluation_split=split, checkpoint=str(checkpoint),
            checkpoint_config=saved, checkpoint_files=checkpoint_files,
            training_dataset=training_dataset,
            policy_type='act', policy_settings=settings, scene_snapshot=scene,
            positions=manifests[split]['positions'],
            positions_manifest_sha256=evaluate.fingerprint(manifest_path)['sha256'],
            frozen_config_files={name: evaluate.fingerprint(output / 'configs' / name)
                                 for name in ('scene_config.yaml', 'training_manifest.yaml', 'positions_config.yaml')},
            runtime_source_files=runtime_sources,
            workers=args.workers, backend=args.backend, device=args.device,
            n_action_steps=N_ACTION_STEPS, duration_s=DURATION, chunk_size=64,
            temporal_ensemble_coeff=None, control_hz=10, physics_hz=1000,
            stop_on_contact=False, contact_diagnostics=False, contact_policy='off',
            episodes_requested=EPISODES, episodes_per_position=1, layout_selection='fresh_before_policy_outcomes',
            video=True, video_layout='head_left_wrist_right', viewer=False,
            failed_layouts_replaced=False, interrupted_attempts_retried=False,
        )
        evaluate.write_json(output / 'eval_config.json', configuration)
        configurations[split] = configuration
    return configurations


def verify_provenance(configuration):
    """Reject drift in frozen geometry, checkpoint bytes, or executed source."""
    from ..environment import evaluation_layouts_v2 as layouts

    output = Path(configuration['output'])
    for name, expected in configuration['frozen_config_files'].items():
        if evaluate.fingerprint(output / 'configs' / name) != expected:
            raise ValueError(f'Frozen configuration changed: {name}')
    manifest = yaml.safe_load((output / 'configs/positions_config.yaml').read_text())
    scene_path = output / 'configs/scene_config.yaml'
    scene = yaml.safe_load(scene_path.read_text())
    if scene != configuration['scene_snapshot'] or manifest['positions'] != configuration['positions']:
        raise ValueError('Frozen scene/positions disagree with eval_config.json')
    if (manifest['split'] != configuration['evaluation_split']
            or evaluate.fingerprint(output / 'configs/positions_config.yaml')['sha256']
            != configuration['positions_manifest_sha256']):
        raise ValueError('Frozen manifest split/hash disagrees with eval_config.json')
    layouts.validate_manifest(manifest, scene, evaluate.workspace_path(manifest['base_config']),
                              evaluate.workspace_path(manifest['training_manifest']['path']))
    checkpoint = relocated_checkpoint_path(configuration['checkpoint'])
    actual = {str(path.relative_to(checkpoint)): evaluate.fingerprint(path)
              for path in sorted(checkpoint.rglob('*')) if path.is_file()}
    if actual != configuration['checkpoint_files']:
        raise ValueError('Checkpoint files changed after evaluation was prepared')
    for relative, expected in configuration['runtime_source_files'].items():
        if evaluate.fingerprint(WORKSPACE_ROOT / relative) != expected:
            raise ValueError(f'Runtime source changed after evaluation was prepared: {relative}')
    saved = read_json(checkpoint / 'config.json')
    if (saved != configuration['checkpoint_config']
            or validate_policy(saved) != configuration['policy_settings']):
        raise ValueError('Saved policy configuration/settings disagree with checkpoint')
    if validate_training_dataset(checkpoint) != configuration['training_dataset']:
        raise ValueError('Training dataset provenance changed')
    contract = dict(protocol='insert_plug_v2_regions_100x1', n_action_steps=32, duration_s=60.0,
                    chunk_size=64, temporal_ensemble_coeff=None, control_hz=10, physics_hz=1000,
                    stop_on_contact=False, contact_diagnostics=False, contact_policy='off',
                    episodes_requested=100, episodes_per_position=1)
    for key, value in contract.items():
        if configuration.get(key) != value:
            raise ValueError(f'Unexpected saved execution contract: {key}')


def load_existing(args):
    if not args.ind_output or not args.ood_output:
        raise ValueError('--resume/--audit-only requires both --ind-output and --ood-output')
    from mujoco_tools.lerobot_policy import resolve_checkpoint
    checkpoint = resolve_checkpoint(evaluate.workspace_path(args.checkpoint))
    configurations = {}
    for split in SPLITS:
        output = evaluate.workspace_path(getattr(args, f'{split}_output'))
        configuration = read_run_record(output / 'eval_config.json')
        if (configuration['output'] != str(output) or configuration['evaluation_split'] != split
                or relocated_checkpoint_path(configuration['checkpoint']) != checkpoint):
            raise ValueError(f'Resume target/checkpoint mismatch: {split}')
        if configuration['device'] != args.device or configuration['backend'] != args.backend:
            raise ValueError('Resume must retain the original device and rendering backend')
        if configuration['workers'] != args.workers:
            raise ValueError('Resume must retain the original worker count')
        verify_provenance(configuration)
        configurations[split] = configuration
    outputs = [Path(configuration['output']) for configuration in configurations.values()]
    if outputs[0] == outputs[1] or outputs[0].is_relative_to(outputs[1]) or outputs[1].is_relative_to(outputs[0]):
        raise ValueError('InD and OOD directories must be separate, nonnested directories')
    return configurations


def write_infrastructure_failure(folder, row, error):
    """Record an attempted but incomplete episode without inventing physics data."""
    if (folder / 'result.json').exists() or (folder / 'infrastructure_failure.json').exists():
        return
    evaluate.write_json(folder / 'infrastructure_failure.json', dict(
        episode_index=row['episode_index'], position=row['name'], seed=row['seed'],
        split=row['split'], success=False, stop_reason='infrastructure_failure',
        runtime_error=str(error), recorded_at=now(), rerun=False,
    ))


def pending_episodes(configuration):
    """Resume only untouched layouts; interrupted attempts stay in the denominator."""
    pending = []
    for row in configuration['positions']:
        folder = episode_directory(configuration, row)
        if folder.is_symlink() or (folder.exists() and not folder.is_dir()):
            raise ValueError(f'Episode destination must be a regular directory: {folder}')
        if not folder.exists():
            pending.append(row)
        elif not (folder / 'result.json').exists() and not (folder / 'infrastructure_failure.json').exists():
            write_infrastructure_failure(folder, row, 'Previous attempt ended without a complete result; preserved')
    return pending


def _initialize_worker(backend):
    os.environ.pop('PYTHONPATH', None)
    os.environ.update(evaluate.ENVIRONMENT)
    os.environ.update(MUJOCO_GL=backend, PYTHONNOUSERSITE='1', HF_DATASETS_OFFLINE='1')
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def execute_episode(configuration, row):
    """Execute one frozen layout. Policy loading occurs once per worker process."""
    global _WORKER_POLICY
    folder = episode_directory(configuration, row)
    folder.mkdir(exist_ok=False)
    evaluate.write_json(folder / 'started.json', dict(started_at=now(), pid=os.getpid(), layout=row))
    try:
        from mujoco_tools.lerobot_policy import LeRobotPolicy
        from .rollout import _seed_policy, run_episode, validate_contract
        from ..environment.scene import build_scene
        from ..environment.simulation import Simulation
        if _WORKER_POLICY is None:
            _WORKER_POLICY = LeRobotPolicy(configuration['checkpoint'], device=configuration['device'],
                                           n_action_steps=N_ACTION_STEPS)
        if Path(_WORKER_POLICY.checkpoint) != relocated_checkpoint_path(configuration['checkpoint']):
            raise ValueError('A worker cannot reuse a policy from a different checkpoint')
        scene = expected_scene(configuration, row)
        validate_contract(scene, _WORKER_POLICY)
        build_scene(scene, folder / 'scene.xml')
        _seed_policy(row['seed'])
        sim = Simulation(folder / 'scene.xml', scene, render=True)
        try:
            result = run_episode(sim, _WORKER_POLICY, duration=DURATION, video_path=folder / 'rollout.mp4',
                                 viewer=False, realtime=False, stop_on_contact=False,
                                 contact_diagnostics=False, quiet=True)
        finally:
            sim.close()
        result.update(episode_index=0, trial_index=row['episode_index'], repetition=0, seed=row['seed'],
                      position=row['name'], split=configuration['evaluation_split'],
                      config=scene, policy=_WORKER_POLICY.metadata(), scene='scene.xml')
        evaluate.write_json(folder / 'result.json', result)
        return dict(split=configuration['evaluation_split'], episode_index=row['episode_index'],
                    stop_reason=result['stop_reason'], success=result['success'])
    except Exception:
        error = traceback.format_exc()
        write_infrastructure_failure(folder, row, error)
        return dict(split=configuration['evaluation_split'], episode_index=row['episode_index'],
                    stop_reason='infrastructure_failure', success=False, error=error)


def check_success_metrics(result):
    """Recompute successful final seating/release/withdrawal checks from saved metrics."""
    if not result['task_success']:
        return
    metrics, config = result['final_metrics'], result['config']
    clearance = config['board']['hole_radius'] - config['peg']['shaft_radius']
    limits = dict(radial_error_m=clearance, mouth_axis_error_m=clearance,
                  seating_error_m=min(config['task']['seating_tolerance'], .0007),
                  max_peg_penetration_m=.00015, tilt_rad=.10,
                  linear_speed_m_s=.005, angular_speed_rad_s=.1)
    for key, limit in limits.items():
        value = metrics.get(key)
        summarize.require(type(value) in (int, float) and math.isfinite(value) and 0 <= value < limit,
                          f'Successful episode violates physical criterion {key}')
    summarize.require(metrics['gripper_contacts'] == [], 'Successful peg remains gripped')
    summarize.require(metrics['aperture_m'] > 2 * config['peg']['head_radius'] + .004,
                      'Successful episode lacks gripper release')
    head_offset = config['peg']['shaft_length'] + config['peg']['head_height'] / 2
    head_z = metrics['peg_tip_xyz'][2] + math.cos(metrics['tilt_rad']) * head_offset
    summarize.require(metrics['tcp_xyz'][2] > head_z + .04, 'Successful episode lacks TCP withdrawal')


def audit_episode(configuration, row, *, video_check='full'):
    """Audit one real 100x1 identity and reuse the shared physical/video checks."""
    folder = episode_directory(configuration, row)
    result = read_json(folder / 'result.json')
    for key, expected in dict(episode_index=0, trial_index=row['episode_index'], position=row['name'],
                              repetition=0, seed=row['seed'], split=configuration['evaluation_split']).items():
        summarize.require(result.get(key) == expected and type(result.get(key)) is type(expected),
                          f'Episode identity mismatch: {key}')
    summarize.require(result['config'] == expected_scene(configuration, row), 'Scene configuration drift')
    for key, expected in dict(contact_policy='off', contact_diagnostics=False, stop_on_contact=False,
                              observation_modalities='rgbd', depth_rendered=True, depth_unit='m').items():
        summarize.require(result.get(key) == expected, f'Execution contract mismatch: {key}')
    policy = result['policy']
    summarize.require(policy['depth_camera_map'] == configuration['checkpoint_config']['depth_camera_map'],
                      'RGB-D camera pairing changed')
    summarize.require(policy['config_sha256'] == configuration['checkpoint_files']['config.json']['sha256'],
                      'Loaded policy configuration differs from frozen checkpoint')
    check_success_metrics(result)
    evidence = summarize.audit_episode(
        folder, result, position=row['name'], repetition=0, layout=row, video_check=video_check,
        checkpoint=relocated_checkpoint_path(configuration['checkpoint']), n_action_steps=N_ACTION_STEPS, duration=DURATION,
        policy_config=configuration['checkpoint_config'], expected_seed=row['seed'],
    )
    evidence.update(episode_index=row['episode_index'], position=row['name'], seed=row['seed'],
                    split=configuration['evaluation_split'], result_sha256=evaluate.fingerprint(folder / 'result.json'))
    return result, evidence


def summarize_split(configuration, *, audit=False, video_check='full'):
    """Count all 100 requested layouts, keeping physical and infrastructure failures separate."""
    rows, audits, errors = [], [], []
    for layout in configuration['positions']:
        folder = episode_directory(configuration, layout)
        row = dict(episode_index=layout['episode_index'], position=layout['name'], seed=layout['seed'],
                   source_xy=layout['source_xy'], target_xy=layout['target_xy'],
                   success=False, status='pending', simulation_seconds=None,
                   reported_stop_reason=None, runtime_failure=False)
        try:
            if (folder / 'result.json').exists():
                result = read_json(folder / 'result.json')
                row['reported_stop_reason'] = result.get('stop_reason')
                row['runtime_failure'] = result.get('stop_reason') in summarize.INFRASTRUCTURE_REASONS
                if audit:
                    result, evidence = audit_episode(configuration, layout, video_check=video_check)
                    audits.append(evidence)
                row.update(status=result['stop_reason'], success=result['success'],
                           simulation_seconds=result['simulation_seconds'],
                           report=str((folder / 'result.json').relative_to(configuration['output'])),
                           video=str((folder / 'rollout.mp4').relative_to(configuration['output'])))
            elif (folder / 'infrastructure_failure.json').exists():
                failure = read_json(folder / 'infrastructure_failure.json')
                for key, value in dict(episode_index=layout['episode_index'], position=layout['name'],
                                       seed=layout['seed'], split=configuration['evaluation_split']).items():
                    summarize.require(failure.get(key) == value, f'Infrastructure identity mismatch: {key}')
                row.update(status='infrastructure_failure', runtime_error=failure['runtime_error'])
            elif folder.exists():
                row['status'] = 'started_incomplete'
        except Exception as exc:
            row.update(status='artifact_error', success=False, error=f'{type(exc).__name__}: {exc}')
            errors.append(dict(episode_index=layout['episode_index'], error=row['error']))
        rows.append(row)
    counts = Counter(row['status'] for row in rows)
    successes = sum(row['success'] is True for row in rows)
    pending = counts['pending'] + counts['started_incomplete']
    summary = dict(
        protocol=configuration['protocol'], evaluation_split=configuration['evaluation_split'],
        checkpoint=configuration['checkpoint'], output=configuration['output'], updated_at=now(),
        episodes_requested=EPISODES, episodes_terminal=EPISODES-pending,
        unique_layouts=EPISODES, episodes_per_layout=1, pending=pending,
        successes=successes, success_rate=successes / EPISODES,
        success_rate_denominator=EPISODES, complete=pending == 0,
        physical_failures=sum(counts[reason] for reason in summarize.PHYSICAL_REASONS if reason != 'success'),
        runtime_failures=sum(row['runtime_failure'] for row in rows),
        infrastructure_failures=counts['infrastructure_failure'], artifact_errors=errors,
        outcome_counts=dict(counts), full_audit=bool(audit and video_check == 'full'),
        artifacts_validated=bool(audit and not errors and pending == 0 and not counts['infrastructure_failure']),
        results=rows,
    )
    output = Path(configuration['output'])
    evaluate.write_json(output / ('summary.json' if audit else 'progress.json'), summary)
    if audit:
        evaluate.write_json(output / 'audit.json', dict(updated_at=now(), video_check=video_check,
                                                       episodes=audits, errors=errors))
        with (output / 'episodes.csv').open('w', newline='', encoding='utf8') as stream:
            fields = ['episode_index', 'position', 'seed', 'source_xy', 'target_xy',
                      'status', 'success', 'simulation_seconds', 'report', 'video', 'runtime_error', 'error',
                      'reported_stop_reason', 'runtime_failure']
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    return summary


def run(configurations, workers):
    # Alternate splits throughout the run to share the same resource conditions.
    queues = [[(configuration, row) for row in pending_episodes(configuration)]
              for configuration in configurations.values()]
    jobs = [job for pair in zip_longest(*queues) for job in pair if job is not None]
    for configuration in configurations.values():
        summarize_split(configuration)
    if jobs:
        backend = next(iter(configurations.values()))['backend']
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=_initialize_worker, initargs=(backend,)) as pool:
            futures = {pool.submit(execute_episode, configuration, row): (configuration, row)
                       for configuration, row in jobs}
            for future in as_completed(futures):
                configuration, row = futures[future]
                try:
                    event = future.result()
                except Exception as exc:
                    folder = episode_directory(configuration, row)
                    if folder.exists():
                        write_infrastructure_failure(folder, row, f'Worker process failed: {exc}')
                    event = dict(split=configuration['evaluation_split'], episode_index=row['episode_index'],
                                 stop_reason='worker_process_failure', error=str(exc))
                print(json.dumps(dict(event='episode_complete', **event), ensure_ascii=False), flush=True)
                summarize_split(configuration)
    summaries = {}
    for split, configuration in configurations.items():
        verify_provenance(configuration)
        summaries[split] = summarize_split(configuration, audit=True)
    return summaries


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--scene-config', default='config/insert_plug.yaml')
    parser.add_argument('--training-manifest', default='config/insert_plug_positions_v2.yaml')
    parser.add_argument('--output-root', default='outputs/test')
    parser.add_argument('--ind-output')
    parser.add_argument('--ood-output')
    parser.add_argument('--ind-seed', type=int)
    parser.add_argument('--ood-seed', type=int)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--backend', choices=('egl', 'osmesa'), default='egl')
    parser.add_argument('--device', choices=('cuda', 'cpu', 'auto'), default='cuda')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--resume', action='store_true')
    modes.add_argument('--audit-only', action='store_true')
    modes.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    configurations = load_existing(args) if args.resume or args.audit_only else prepare(args)
    with ExitStack() as stack:
        # Lock in a fixed order, preventing concurrent resume/audit processes.
        for configuration in sorted(configurations.values(), key=lambda item: item['output']):
            handle = stack.enter_context((Path(configuration['output']) / 'evaluation.lock').open('a'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            print(json.dumps({split: configuration['output'] for split, configuration in configurations.items()}))
            return 0
        if args.audit_only:
            summaries = {split: summarize_split(configuration, audit=True)
                         for split, configuration in configurations.items()}
        else:
            summaries = run(configurations, args.workers)
        print(json.dumps({split: {key: summary[key] for key in (
            'output', 'successes', 'episodes_requested', 'complete', 'runtime_failures',
            'infrastructure_failures', 'artifacts_validated')}
            for split, summary in summaries.items()}, ensure_ascii=False, indent=2))
        return 0 if all(summary['artifacts_validated'] and not summary['runtime_failures']
                        for summary in summaries.values()) else 1


if __name__ == '__main__':
    sys.exit(main())
