"""Evaluate DiT Diffusion on the exact 100+100 layouts of a completed ACT v2 run.

Reference geometry is copied byte for byte before policy loading. Outcomes never
select or replace layouts. Shared execution/audit helpers remain unchanged, so
the completed ACT experiment retains its original source-hash evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
import fcntl
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path

import yaml

from mujoco_tools.paths import DATA_DIR, WORKSPACE_ROOT, checkpoint_dataset_path, relocated_checkpoint_path
from . import evaluate, evaluate_regions_v2 as base, summarize


SPLITS = base.SPLITS
EPISODES = base.EPISODES
N_ACTION_STEPS = base.N_ACTION_STEPS
DURATION = base.DURATION
PROTOCOL = 'insert_plug_v2_regions_100x1'
FROZEN_NAMES = ('scene_config.yaml', 'training_manifest.yaml', 'positions_config.yaml')
REFERENCE_NAMES = ('eval_config.json', 'summary.json', 'audit.json', 'episodes.csv',
                   *(f'configs/{name}' for name in FROZEN_NAMES))
RUNTIME_SOURCES = tuple(path for path in base.RUNTIME_SOURCES if '/policies/act/' not in path) + (
    'examples/insert_plug/policy/evaluate_dit_regions_v2.py',
    'examples/insert_plug/policy/evaluate.py',
    'third_party/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py',
    'third_party/lerobot/src/lerobot/policies/diffusion/configuration_diffusion.py',
    'third_party/lerobot/src/lerobot/policies/diffusion/modeling_dit.py',
    'third_party/lerobot/src/lerobot/policies/diffusion/processor_diffusion.py',
)
CONTRACT = dict(protocol=PROTOCOL, n_action_steps=32, duration_s=60.0, chunk_size=64,
                temporal_ensemble_coeff=None, control_hz=10, physics_hz=1000,
                stop_on_contact=False, contact_diagnostics=False, contact_policy='off',
                episodes_requested=100, episodes_per_position=1,
                video=True, video_layout='head_left_wrist_right', viewer=False,
                failed_layouts_replaced=False, interrupted_attempts_retried=False)
DIT_CONTRACT = dict(CONTRACT, policy_type='diffusion', denoiser_type='dit', horizon=64,
                    n_obs_steps=1, noise_scheduler_type='DDPM', num_train_timesteps=100,
                    num_inference_steps=100, prediction_type='epsilon',
                    layout_selection='exact_reference_geometry_without_outcome_selection')

# These helpers contain no ACT-only policy validation. Spawned workers own the
# shared module's policy cache; no monkeypatching or source rewriting is used.
now = base.now
read_json = base.read_json
episode_directory = base.episode_directory
expected_scene = base.expected_scene
execute_episode = base.execute_episode
pending_episodes = base.pending_episodes
audit_episode = base.audit_episode
summarize_split = base.summarize_split
_initialize_worker = base._initialize_worker


def require_fields(value, expected, context):
    for key, wanted in expected.items():
        actual = value.get(key)
        if actual != wanted or (isinstance(wanted, (bool, int)) and type(actual) is not type(wanted)):
            raise ValueError(f'{context}: unexpected {key}')


def file_inventory(root, names):
    result = {}
    for name in names:
        path = Path(root) / name
        summarize.regular_file(path)
        result[name] = evaluate.fingerprint(path)
    return result


def checkpoint_inventory(checkpoint):
    names = [str(path.relative_to(checkpoint)) for path in sorted(checkpoint.rglob('*')) if path.is_file()]
    required = {'config.json', 'train_config.json', 'model.safetensors',
                'policy_preprocessor.json', 'policy_postprocessor.json'}
    if not required.issubset(names):
        raise ValueError('Checkpoint lacks model, configuration or saved processors')
    return file_inventory(checkpoint, names)


def validate_policy(saved):
    require_fields(saved, dict(type='diffusion', denoiser_type='dit', horizon=64,
                               n_obs_steps=1, n_action_steps=64, noise_scheduler_type='DDPM',
                               num_train_timesteps=100, prediction_type='epsilon'), 'DiT checkpoint')
    inference_steps = saved.get('num_inference_steps')
    if inference_steps is not None and (type(inference_steps) is not int or inference_steps != 100):
        raise ValueError('DiT checkpoint must retain all 100 DDPM inference steps')
    expected_map = {f'observation.images.{camera}': f'observation.depth.{camera}'
                    for camera in ('head', 'wrist')}
    features = {
        'observation.state': dict(type='STATE', shape=[8]),
        **{f'observation.images.{camera}': dict(type='VISUAL', shape=[3, 256, 256])
           for camera in ('head', 'wrist')},
        **{f'observation.depth.{camera}': dict(type='VISUAL', shape=[1, 256, 256])
           for camera in ('head', 'wrist')},
    }
    if (saved.get('depth_camera_map') != expected_map or saved.get('depth_unit') != 'm'
            or saved.get('input_features') != features
            or saved.get('output_features') != {'action': dict(type='ACTION', shape=[8])}):
        raise ValueError('DiT evaluation requires paired head/wrist 256px metric RGB-D and 8-D state/action')
    if saved.get('normalization_mapping') != dict(VISUAL='MEAN_STD', STATE='MIN_MAX', ACTION='MIN_MAX'):
        raise ValueError('DiT checkpoint normalization must retain its RGB-D MIN_MAX action/state preset')
    if saved.get('temporal_ensemble_coeff') is not None:
        raise ValueError('ACT temporal ensembling cannot apply to DiT Diffusion')
    return evaluate.policy_settings(saved, N_ACTION_STEPS)


def validate_training_dataset(checkpoint):
    dataset = read_json(Path(checkpoint) / 'train_config.json')['dataset']
    expected = (DATA_DIR / 'insert_plug_v2/lerobot').resolve()
    if (checkpoint_dataset_path(dataset['root']) != expected
            or dataset.get('episodes') is not None or dataset.get('exclude_episodes') is not None):
        raise ValueError('Checkpoint must use the complete insert_plug_v2 dataset without episode filters')
    return dataset


def comparable_dataset(dataset):
    """Compare training configurations across the dataset directory move."""
    return dict(dataset, root=checkpoint_dataset_path(dataset['root']))


def validate_geometry(scene, manifest, split):
    from ..environment import evaluation_layouts_v2 as layouts
    if manifest.get('split') != split:
        raise ValueError('Reference/frozen manifest split mismatch')
    layouts.validate_manifest(manifest, scene, evaluate.workspace_path(manifest['base_config']),
                              evaluate.workspace_path(manifest['training_manifest']['path']))
    if (scene['task']['control_hz'] != 10 or scene['physics']['timestep'] != .001
            or scene['recording']['sample_hz'] != 10
            or scene['cameras']['width'] != 256 or scene['cameras']['height'] != 256):
        raise ValueError('Reference scene must retain 10 Hz/1 kHz and 256px cameras')


def load_reference(path, split):
    """Validate the complete reference; success/failure never chooses any row."""
    path = evaluate.workspace_path(path)
    files = file_inventory(path, REFERENCE_NAMES)
    cfg = base.read_run_record(path / 'eval_config.json')
    report = base.read_run_record(path / 'summary.json')
    audit = read_json(path / 'audit.json')
    require_fields(cfg, dict(CONTRACT, output=str(path), evaluation_split=split,
                            policy_type='act', layout_selection='fresh_before_policy_outcomes'), 'Reference ACT')
    require_fields(report, dict(protocol=PROTOCOL, evaluation_split=split, output=str(path),
                               checkpoint=cfg['checkpoint'], episodes_requested=100, episodes_terminal=100,
                               unique_layouts=100, episodes_per_layout=1, pending=0, complete=True,
                               full_audit=True, artifacts_validated=True, runtime_failures=0,
                               infrastructure_failures=0, artifact_errors=[]), 'Reference summary')
    if (audit.get('video_check') != 'full' or audit.get('errors') != []
            or len(audit.get('episodes', [])) != 100):
        raise ValueError('Reference ACT must have a complete successful artifact audit')
    frozen = {name: files[f'configs/{name}'] for name in FROZEN_NAMES}
    if frozen != cfg.get('frozen_config_files'):
        raise ValueError('Reference frozen configuration hashes changed')
    scene = yaml.safe_load((path / 'configs/scene_config.yaml').read_text())
    manifest = yaml.safe_load((path / 'configs/positions_config.yaml').read_text())
    validate_geometry(scene, manifest, split)
    if (scene != cfg['scene_snapshot'] or manifest['positions'] != cfg['positions']
            or frozen['positions_config.yaml']['sha256'] != cfg['positions_manifest_sha256']):
        raise ValueError('Reference scene/positions disagree with its evaluation configuration')
    if (frozen['scene_config.yaml']['sha256'] != manifest['base_config_sha256']
            or frozen['training_manifest.yaml']['sha256'] != manifest['training_manifest']['sha256']):
        raise ValueError('Reference frozen scene/training bytes differ from manifest sources')
    rows = report.get('results', [])
    audit_rows = {row['episode_index']: row for row in audit['episodes']}
    if len(rows) != 100 or set(audit_rows) != set(range(100)):
        raise ValueError('Reference results/audits must cover all 100 unique layouts')
    for index, (row, layout) in enumerate(zip(rows, manifest['positions'])):
        require_fields(row, dict(episode_index=index, position=f'pos{index}', seed=index,
                                 source_xy=layout['source_xy'], target_xy=layout['target_xy']),
                       'Reference result identity')
        require_fields(audit_rows[index], dict(episode_index=index, position=f'pos{index}',
                                               seed=index, split=split), 'Reference audit identity')
        if audit_rows[index].get('video_audit', {}).get('full_decode') is not True:
            raise ValueError('Reference video audit must fully decode every episode')
    return dict(output=str(path), files=files), cfg, scene, manifest


def prepare(args):
    from mujoco_tools.lerobot_policy import resolve_checkpoint
    checkpoint = resolve_checkpoint(evaluate.workspace_path(args.checkpoint))
    saved = read_json(checkpoint / 'config.json')
    settings = validate_policy(saved)
    dataset = validate_training_dataset(checkpoint)
    references = {split: load_reference(getattr(args, f'reference_{split}_output'), split) for split in SPLITS}
    ind, ood = references['ind'], references['ood']
    if (ind[2] != ood[2]
            or relocated_checkpoint_path(ind[1]['checkpoint']) != relocated_checkpoint_path(ood[1]['checkpoint'])
            or ind[3]['regions'] != ood[3]['regions']
            or ind[1]['frozen_config_files']['training_manifest.yaml']
            != ood[1]['frozen_config_files']['training_manifest.yaml']
            or not (comparable_dataset(ind[1]['training_dataset']) == comparable_dataset(ood[1]['training_dataset'])
                    == comparable_dataset(dataset))):
        raise ValueError('References must share the ACT checkpoint, scene, regions and complete v2 dataset')
    started = now()
    outputs = {split: evaluate.workspace_path(getattr(args, f'{split}_output') or
               Path(args.output_root) / evaluate.batch_output_path('diffusion_dit_insert_plug_v2_rgbd', split, started))
               for split in SPLITS}
    a, b = outputs.values()
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('InD and OOD directories must be separate, nonnested directories')
    recorded_root = evaluate.workspace_path(dataset['root'])
    protected = [checkpoint, recorded_root, checkpoint_dataset_path(recorded_root),
                 *(Path(reference[0]['output']) for reference in references.values())]
    for path in outputs.values():
        if any(path == prior or path.is_relative_to(prior) or prior.is_relative_to(path) for prior in protected):
            raise ValueError('Output overlaps a protected checkpoint, dataset or reference directory')
        if path.exists() or path.is_symlink():
            raise FileExistsError(f'Output exists; use explicit --resume: {path}')
    checkpoint_files = checkpoint_inventory(checkpoint)
    runtime_sources = file_inventory(WORKSPACE_ROOT, RUNTIME_SOURCES)
    configurations = {}
    for split, output in outputs.items():
        reference, _, scene, manifest = references[split]
        output.mkdir(parents=True, exist_ok=False)
        (output / 'configs').mkdir()
        (output / 'episodes').mkdir()
        for name in FROZEN_NAMES:
            with (output / 'configs' / name).open('xb') as stream:
                stream.write((Path(reference['output']) / 'configs' / name).read_bytes())
        configuration = dict(
            DIT_CONTRACT, format_version=1, started_at=started, output=str(output),
            evaluation_split=split, checkpoint=str(checkpoint), checkpoint_config=saved,
            checkpoint_files=checkpoint_files, training_dataset=dataset, policy_settings=settings,
            scene_snapshot=scene, positions=manifest['positions'], reference_evaluation=reference,
            positions_manifest_sha256=reference['files']['configs/positions_config.yaml']['sha256'],
            frozen_config_files=file_inventory(output / 'configs', FROZEN_NAMES),
            runtime_source_files=runtime_sources, workers=args.workers, backend=args.backend, device=args.device,
        )
        verify_provenance(configuration)
        evaluate.write_json(output / 'eval_config.json', configuration)
        configurations[split] = configuration
    return configurations


def verify_provenance(configuration):
    require_fields(configuration, DIT_CONTRACT, 'DiT execution contract')
    output, checkpoint = Path(configuration['output']), relocated_checkpoint_path(configuration['checkpoint'])
    if file_inventory(output / 'configs', FROZEN_NAMES) != configuration['frozen_config_files']:
        raise ValueError('Frozen configuration changed')
    if checkpoint_inventory(checkpoint) != configuration['checkpoint_files']:
        raise ValueError('Checkpoint files changed after evaluation was prepared')
    if file_inventory(WORKSPACE_ROOT, RUNTIME_SOURCES) != configuration['runtime_source_files']:
        raise ValueError('Runtime source files changed after evaluation was prepared')
    saved = read_json(checkpoint / 'config.json')
    if saved != configuration['checkpoint_config'] or validate_policy(saved) != configuration['policy_settings']:
        raise ValueError('Saved policy configuration/settings disagree with checkpoint')
    if validate_training_dataset(checkpoint) != configuration['training_dataset']:
        raise ValueError('Training dataset provenance changed')
    reference, ref_cfg, scene, manifest = load_reference(configuration['reference_evaluation']['output'],
                                                       configuration['evaluation_split'])
    if reference != configuration['reference_evaluation']:
        raise ValueError('Reference evaluation files changed after preparation')
    if comparable_dataset(ref_cfg['training_dataset']) != comparable_dataset(configuration['training_dataset']):
        raise ValueError('DiT and ACT training dataset references differ')
    if (scene != configuration['scene_snapshot'] or manifest['positions'] != configuration['positions']
            or reference['files']['configs/positions_config.yaml']['sha256']
            != configuration['positions_manifest_sha256']):
        raise ValueError('Saved scene/positions differ from the exact reference geometry')
    for name in FROZEN_NAMES:
        if configuration['frozen_config_files'][name] != reference['files'][f'configs/{name}']:
            raise ValueError('Frozen geometry must be byte-identical to the ACT reference')


def load_existing(args):
    from mujoco_tools.lerobot_policy import resolve_checkpoint
    if not args.ind_output or not args.ood_output:
        raise ValueError('--resume/--audit-only requires both --ind-output and --ood-output')
    checkpoint = resolve_checkpoint(evaluate.workspace_path(args.checkpoint))
    configurations = {}
    for split in SPLITS:
        output = evaluate.workspace_path(getattr(args, f'{split}_output'))
        configuration = base.read_run_record(output / 'eval_config.json')
        if (configuration['output'] != str(output) or configuration['evaluation_split'] != split
                or relocated_checkpoint_path(configuration['checkpoint']) != checkpoint):
            raise ValueError('Resume output/split/checkpoint mismatch')
        if configuration['device'] != args.device or configuration['backend'] != args.backend:
            raise ValueError('Resume must retain the original device and rendering backend')
        if configuration['workers'] != args.workers:
            raise ValueError('Resume must retain the original worker count')
        if Path(configuration['reference_evaluation']['output']) != evaluate.workspace_path(
                getattr(args, f'reference_{split}_output')):
            raise ValueError('Resume must retain the original reference evaluation')
        verify_provenance(configuration)
        configurations[split] = configuration
    a, b = (Path(configuration['output']) for configuration in configurations.values())
    if a == b or a.is_relative_to(b) or b.is_relative_to(a):
        raise ValueError('InD and OOD directories must be separate, nonnested directories')
    return configurations


def run(configurations, workers):
    for configuration in configurations.values():
        verify_provenance(configuration)
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
                        base.write_infrastructure_failure(folder, row, f'Worker process failed: {exc}')
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
    for split in SPLITS:
        parser.add_argument(f'--reference-{split}-output', required=True)
        parser.add_argument(f'--{split}-output')
    parser.add_argument('--output-root', default='outputs/test')
    parser.add_argument('--workers', type=int, default=4)
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
    raise SystemExit(main())
