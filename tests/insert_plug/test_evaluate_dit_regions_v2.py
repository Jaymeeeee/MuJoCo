"""DiT evaluation must preserve ACT geometry and reject changed run contracts."""
from concurrent.futures import Future
from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from examples.insert_plug.environment import ring_layouts
from examples.insert_plug.policy import evaluate_regions_v2 as act
from examples.insert_plug.policy import evaluate_dit_regions_v2 as runner
from mujoco_tools import paths


def write_json(path, data):
    path.write_text(json.dumps(data))


@pytest.fixture
def sources(tmp_path):
    scene = dict(board=dict(source_xy=[.45, -.07], target_xy=[.45, .07]),
                 task=dict(control_hz=10), physics=dict(timestep=.001),
                 recording=dict(sample_hz=10), cameras=dict(width=256, height=256))
    scene_path, training_path = tmp_path/'scene.yaml', tmp_path/'training.yaml'
    scene_path.write_text(yaml.safe_dump(scene))
    training = ring_layouts.make_manifest(scene, scene_path)
    training['selection_complete'] = True
    training_path.write_text(yaml.safe_dump(training))
    features = {'observation.state': dict(type='STATE', shape=[8]),
                **{f'observation.images.{camera}': dict(type='VISUAL', shape=[3, 256, 256])
                   for camera in ('head', 'wrist')},
                **{f'observation.depth.{camera}': dict(type='VISUAL', shape=[1, 256, 256])
                   for camera in ('head', 'wrist')}}
    common = dict(n_action_steps=64, n_obs_steps=1, input_features=features,
                  output_features={'action': dict(type='ACTION', shape=[8])},
                  depth_camera_map={f'observation.images.{camera}': f'observation.depth.{camera}'
                                    for camera in ('head', 'wrist')}, depth_unit='m')
    checkpoints = {}
    for model in ('act', 'dit'):
        # The ACT reference predates the move, and the DiT checkpoint uses the new path.
        data_root = 'outputs/datasets' if model == 'act' else 'datasets'
        dataset = dict(root=str(runner.WORKSPACE_ROOT/data_root/'insert_plug_v2/lerobot'),
                       repo_id='local/insert_plug_v2', episodes=None, exclude_episodes=None)
        checkpoint = tmp_path/model/'artifacts/checkpoints/030000/pretrained_model'
        checkpoint.mkdir(parents=True)
        config = dict(common, type='act', chunk_size=64, temporal_ensemble_coeff=None) if model == 'act' else dict(
            common, type='diffusion', denoiser_type='dit', horizon=64, noise_scheduler_type='DDPM',
            num_train_timesteps=100, num_inference_steps=None, prediction_type='epsilon',
            normalization_mapping=dict(VISUAL='MEAN_STD', STATE='MIN_MAX', ACTION='MIN_MAX'))
        write_json(checkpoint/'config.json', config)
        write_json(checkpoint/'train_config.json', dict(dataset=dataset))
        (checkpoint/'model.safetensors').write_bytes(b'synthetic checkpoint, never loaded')
        for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
            write_json(checkpoint/name, dict(steps=[]))
        checkpoints[model] = checkpoint
    act_args = act.parse_args([
        '--checkpoint', str(checkpoints['act']), '--scene-config', str(scene_path),
        '--training-manifest', str(training_path), '--device', 'cpu', '--backend', 'osmesa',
        '--ind-output', str(tmp_path/'outputs/test/act_insert_plug_v2_rgbd_ind_20261112_191102'),
        '--ood-output', str(tmp_path/'outputs/test/act_insert_plug_v2_rgbd_ood_20261112_191102'),
    ])
    references = act.prepare(act_args)
    for split, cfg in references.items():
        output = Path(cfg['output'])
        # Synthetic reports exercise reference metadata checks without physics.
        for layout in cfg['positions']:
            folder = act.episode_directory(cfg, layout)
            folder.mkdir()
            write_json(folder/'result.json', dict(success=False, stop_reason='timeout', simulation_seconds=60))
        summary = act.summarize_split(cfg)
        summary.update(full_audit=True, artifacts_validated=True)
        write_json(output/'summary.json', summary)
        write_json(output/'audit.json', dict(video_check='full', errors=[],
                    episodes=[dict(episode_index=i, position=f'pos{i}', seed=i, split=split,
                                   video_audit=dict(full_decode=True)) for i in range(100)]))
        (output/'episodes.csv').write_text('episode_index,success\n'+''.join(f'{i},False\n' for i in range(100)))
    protected = [scene_path, training_path,
                 *(p for c in checkpoints.values() for p in c.iterdir()),
                 *(Path(cfg['output'])/name for cfg in references.values() for name in runner.REFERENCE_NAMES)]
    return dict(checkpoint=checkpoints['dit'], references=references,
                original_bytes={path: path.read_bytes() for path in protected})


def arguments(sources, tmp_path, *extra):
    return runner.parse_args([
        '--checkpoint', str(sources['checkpoint']),
        '--reference-ind-output', sources['references']['ind']['output'],
        '--reference-ood-output', sources['references']['ood']['output'],
        '--ind-output', str(tmp_path/'dit_ind'), '--ood-output', str(tmp_path/'dit_ood'),
        '--device', 'cpu', '--backend', 'osmesa', '--workers', '4', *extra,
    ])


@pytest.fixture
def configurations(sources, tmp_path):
    return runner.prepare(arguments(sources, tmp_path))


def test_default_directories_group_splits_under_dit_model(sources, tmp_path, monkeypatch):
    args = arguments(sources, tmp_path, '--output-root', str(tmp_path/'test'))
    args.ind_output = args.ood_output = None
    monkeypatch.setattr(runner, 'now', lambda: '2026-11-11T16:30:45+00:00')
    for split, cfg in runner.prepare(args).items():
        assert Path(cfg['output']) == tmp_path/'test/diffusion_dit_insert_plug_v2_rgbd'/split/'20261112_003045'


def test_prepare_copies_all_geometry_bytes_and_200_paired_identities(configurations, sources):
    for split, cfg in configurations.items():
        reference = sources['references'][split]
        assert cfg['positions'] == reference['positions']
        assert len(cfg['positions']) == 100
        assert [row['seed'] for row in cfg['positions']] == list(range(100))
        assert cfg['policy_type'] == 'diffusion' and cfg['denoiser_type'] == 'dit'
        assert cfg['policy_settings']['checkpoint_n_action_steps'] == 64
        assert cfg['n_action_steps'] == 32 and cfg['horizon'] == cfg['chunk_size'] == 64
        assert cfg['noise_scheduler_type'] == 'DDPM' and cfg['num_inference_steps'] == 100
        assert cfg['layout_selection'] == 'exact_reference_geometry_without_outcome_selection'
        assert cfg['reference_evaluation']['output'] == reference['output']
        assert cfg['training_dataset']['root'] == str(runner.WORKSPACE_ROOT/'datasets/insert_plug_v2/lerobot')
        assert reference['training_dataset']['root'] == str(runner.WORKSPACE_ROOT/'outputs/datasets/insert_plug_v2/lerobot')
        for name in runner.FROZEN_NAMES:
            assert (Path(cfg['output'])/'configs'/name).read_bytes() == (
                Path(reference['output'])/'configs'/name).read_bytes()
        assert cfg['duration_s'] == 60 and cfg['control_hz'] == 10 and cfg['physics_hz'] == 1000
        assert not cfg['failed_layouts_replaced'] and not cfg['interrupted_attempts_retried']
        assert len(cfg['runtime_source_files']) == 15
        assert not any('/policies/act/' in name for name in cfg['runtime_source_files'])
        assert all(any(name.endswith(suffix) for name in cfg['runtime_source_files']) for suffix in (
            'evaluate.py', 'evaluate_regions_v2.py', 'modeling_dit.py', 'processor_diffusion.py'))
        runner.verify_provenance(cfg)
    assert {p: p.read_bytes() for p in sources['original_bytes']} == sources['original_bytes']


@pytest.mark.parametrize('field,value', [
    ('type', 'act'), ('denoiser_type', None), ('denoiser_type', 'unet'), ('denoiser_type', 'typo'),
    ('horizon', 32), ('n_obs_steps', 2), ('n_action_steps', 32), ('noise_scheduler_type', 'DDIM'),
    ('num_train_timesteps', 50), ('num_inference_steps', 10), ('prediction_type', 'sample'),
    ('depth_unit', 'mm'), ('temporal_ensemble_coeff', .01),
])
def test_rejects_incompatible_policy_before_writing_outputs(sources, tmp_path, field, value):
    path = sources['checkpoint']/'config.json'
    config = json.loads(path.read_text())
    config[field] = value
    write_json(path, config)
    with pytest.raises(ValueError):
        runner.prepare(arguments(sources, tmp_path))
    assert not (tmp_path/'dit_ind').exists() and not (tmp_path/'dit_ood').exists()


@pytest.mark.parametrize('mutation', ['wrong_dataset', 'episodes', 'exclude_episodes', 'different_reference_dataset'])
def test_rejects_different_or_filtered_training_data(sources, tmp_path, mutation):
    path = sources['checkpoint']/'train_config.json'
    document = json.loads(path.read_text())
    if mutation == 'wrong_dataset':
        document['dataset']['root'] = str(runner.WORKSPACE_ROOT/'datasets/insert_plug_v1/lerobot')
    elif mutation == 'different_reference_dataset':
        document['dataset']['repo_id'] = 'another/config'
    else:
        document['dataset'][mutation] = [5]
    write_json(path, document)
    with pytest.raises(ValueError, match='dataset'):
        runner.prepare(arguments(sources, tmp_path))
    assert not (tmp_path/'dit_ind').exists()


@pytest.mark.parametrize('mutation', ['incomplete', 'unaudited', 'wrong_seed', 'wrong_split', 'changed_geometry'])
def test_rejects_invalid_reference_before_preparation(sources, tmp_path, mutation):
    root = Path(sources['references']['ind']['output'])
    if mutation == 'changed_geometry':
        with (root/'configs/positions_config.yaml').open('a') as stream:
            stream.write('\n# unexpected mutation\n')
    else:
        path = root/'summary.json'
        summary = json.loads(path.read_text())
        if mutation == 'incomplete':
            summary['complete'] = False
        elif mutation == 'unaudited':
            summary['artifacts_validated'] = False
        elif mutation == 'wrong_seed':
            summary['results'][44]['seed'] = 440
        else:
            summary['evaluation_split'] = 'ood'
        write_json(path, summary)
    with pytest.raises(ValueError):
        runner.prepare(arguments(sources, tmp_path))
    assert not (tmp_path/'dit_ind').exists()


@pytest.mark.parametrize('protected', ['checkpoint', 'reference', 'dataset', 'ancestor'])
def test_output_cannot_overlap_preserved_artifacts(sources, tmp_path, protected, monkeypatch):
    paths = dict(checkpoint=sources['checkpoint']/'evaluation',
                 reference=Path(sources['references']['ind']['output'])/'evaluation',
                 dataset=runner.WORKSPACE_ROOT/'datasets/insert_plug_v2/lerobot/evaluation',
                 ancestor=tmp_path/'dit')
    if protected == 'dataset':
        train_config = sources['checkpoint']/'train_config.json'
        write_json(train_config, {'dataset': sources['references']['ind']['training_dataset']})
    monkeypatch.setattr(Path, 'mkdir', lambda *args, **kwargs: pytest.fail('Unexpected output creation'))
    with pytest.raises(ValueError, match='protected'):
        runner.prepare(arguments(sources, tmp_path, '--ind-output', str(paths[protected])))
    assert not (tmp_path/'dit_ood').exists()


@pytest.mark.parametrize('mutation', ['index', 'seed', 'split', 'full_decode'])
def test_reference_requires_all_100_video_audits_with_exact_identities(sources, tmp_path, mutation):
    path = Path(sources['references']['ind']['output'])/'audit.json'
    audit = json.loads(path.read_text())
    row = audit['episodes'][0]
    if mutation == 'index':
        row['episode_index'] = 100
    elif mutation == 'seed':
        row['seed'] = 50
    elif mutation == 'split':
        row['split'] = 'ood'
    else:
        row['video_audit']['full_decode'] = False
    write_json(path, audit)
    with pytest.raises(ValueError):
        runner.prepare(arguments(sources, tmp_path))
    assert not (tmp_path/'dit_ind').exists()


def test_resume_resolves_moved_runs_and_fingerprinted_references(sources, tmp_path, monkeypatch):
    test_root = tmp_path/'outputs/test'
    monkeypatch.setattr(paths, 'TEST_DIR', test_root)
    args = arguments(sources, tmp_path, '--resume')
    for split in runner.SPLITS:
        setattr(args, f'{split}_output', str(test_root/f'diffusion_dit_insert_plug_v2_rgbd_{split}_20261112_202421'))
    prepared = runner.prepare(args)
    for split, config in prepared.items():
        for group, recorded in [('act_insert_plug_v2_rgbd', config['reference_evaluation']['output']),
                                ('diffusion_dit_insert_plug_v2_rgbd', config['output'])]:
            old = Path(recorded)
            new = test_root/group/split/old.name[-15:]
            new.parent.mkdir(parents=True)
            old.rename(new)

    restored = runner.load_existing(args)
    for split, config in restored.items():
        output = Path(config['output'])
        reference = Path(config['reference_evaluation']['output'])
        assert output == test_root/'diffusion_dit_insert_plug_v2_rgbd'/split/'20261112_202421'
        assert reference == test_root/'act_insert_plug_v2_rgbd'/split/'20261112_191102'
        assert json.loads((output/'eval_config.json').read_text()) == prepared[split]
        assert config['reference_evaluation']['files'] == prepared[split]['reference_evaluation']['files']


def test_existing_results_are_preserved_and_resume_restores_configuration(configurations, sources, tmp_path):
    originals = {Path(c['output'])/'eval_config.json': (Path(c['output'])/'eval_config.json').read_bytes()
                 for c in configurations.values()}
    with pytest.raises(FileExistsError):
        runner.prepare(arguments(sources, tmp_path))
    assert runner.load_existing(arguments(sources, tmp_path, '--resume')) == configurations
    assert {p: p.read_bytes() for p in originals} == originals
    assert {p: p.read_bytes() for p in sources['original_bytes']} == sources['original_bytes']


@pytest.mark.parametrize('mutation', ['reference', 'copied_manifest', 'checkpoint', 'source_hash',
                                     'settings', 'contract', 'checkpoint_snapshot'])
def test_resume_rejects_frozen_evidence_drift(configurations, mutation):
    cfg = configurations['ind']
    if mutation == 'reference':
        path = Path(cfg['reference_evaluation']['output'])/'episodes.csv'
        path.write_text(path.read_text()+'changed\n')
    elif mutation == 'copied_manifest':
        path = Path(cfg['output'])/'configs/positions_config.yaml'
        path.write_text(path.read_text()+'\n# changed\n')
    elif mutation == 'checkpoint':
        (Path(cfg['checkpoint'])/'model.safetensors').write_bytes(b'changed')
    elif mutation == 'source_hash':
        cfg['runtime_source_files'][runner.RUNTIME_SOURCES[0]]['sha256'] = 'changed'
    elif mutation == 'settings':
        cfg['policy_settings']['denoiser_type'] = 'unet'
    elif mutation == 'contract':
        cfg['num_inference_steps'] = 10
    else:
        cfg['checkpoint_config']['denoiser_type'] = 'unet'
    with pytest.raises(ValueError):
        runner.verify_provenance(cfg)


@pytest.mark.parametrize('option,value,match', [
    ('--device', 'cuda', 'device'), ('--backend', 'egl', 'backend'),
    ('--workers', '2', 'worker count'), ('--reference-ind-output', '/tmp/wrong-reference', 'reference'),
])
def test_resume_rejects_cli_substitution(configurations, sources, tmp_path, option, value, match):
    with pytest.raises(ValueError, match=match):
        runner.load_existing(arguments(sources, tmp_path, '--resume', option, value))


def test_shared_execution_and_auditing_remain_unmodified():
    assert runner.execute_episode is act.execute_episode
    assert runner.audit_episode is act.audit_episode
    assert runner.pending_episodes is act.pending_episodes
    assert runner.summarize_split is act.summarize_split
    assert runner._initialize_worker is act._initialize_worker


def test_one_spawn_pool_schedules_each_layout_once_in_alternating_splits(configurations, monkeypatch):
    submissions, pools = [], []

    class Pool:
        def __init__(self, **kwargs):
            pools.append(kwargs)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, fn, cfg, row):
            assert fn is act.execute_episode
            submissions.append((cfg['evaluation_split'], row['episode_index'], row['seed']))
            future = Future()
            future.set_result(dict(split=cfg['evaluation_split'], episode_index=row['episode_index'],
                                   stop_reason='timeout', success=False))
            return future

    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(runner, 'summarize_split', lambda cfg, **kwargs: dict(split=cfg['evaluation_split']))
    runner.run(configurations, workers=4)
    assert len(pools) == 1 and pools[0]['max_workers'] == 4
    assert pools[0]['mp_context'].get_start_method() == 'spawn'
    assert submissions == [(split, i, i) for i in range(100) for split in ('ind', 'ood')]


def test_resume_never_retries_physical_runtime_or_interrupted_attempts(configurations):
    cfg = configurations['ind']
    for index, reason in enumerate(('timeout', 'runtime_error', 'infrastructure_failure', 'interrupted')):
        folder = runner.episode_directory(cfg, cfg['positions'][index])
        folder.mkdir()
        if reason in ('timeout', 'runtime_error'):
            write_json(folder/'result.json', dict(stop_reason=reason, success=False))
        elif reason == 'infrastructure_failure':
            act.write_infrastructure_failure(folder, cfg['positions'][index], 'existing failure')
        else:
            write_json(folder/'started.json', dict(started=True))
    assert [row['episode_index'] for row in runner.pending_episodes(cfg)] == list(range(4, 100))
    assert [row['episode_index'] for row in runner.pending_episodes(cfg)] == list(range(4, 100))
    report = json.loads((runner.episode_directory(cfg, cfg['positions'][3])/'infrastructure_failure.json').read_text())
    assert report['rerun'] is False
