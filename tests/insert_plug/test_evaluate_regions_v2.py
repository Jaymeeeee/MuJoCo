"""Fresh v2 evaluation scheduling, immutable provenance, and terminal outcomes."""
from copy import deepcopy
from concurrent.futures import Future
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from examples.insert_plug.environment import evaluation_layouts_v2 as geometry
from examples.insert_plug.environment import ring_layouts
from examples.insert_plug.policy import summarize as audit
from examples.insert_plug.policy import evaluate_regions_v2 as runner


@pytest.fixture
def sources(tmp_path):
    base = dict(
        board=dict(source_xy=[.45, -.07], target_xy=[.45, .07]),
        task=dict(control_hz=10), physics=dict(timestep=.001),
        recording=dict(sample_hz=10), cameras=dict(width=256, height=256),
    )
    scene = tmp_path / 'scene.yaml'
    scene.write_text(yaml.safe_dump(base))
    training = ring_layouts.make_manifest(base, scene)
    training['selection_complete'] = True
    training_path = tmp_path / 'training.yaml'
    training_path.write_text(yaml.safe_dump(training))
    manifests = {}
    for split in ('ind', 'ood'):
        document = geometry.make_manifest(base, scene, training_path, split=split)
        path = tmp_path / f'{split}.yaml'
        geometry.write_manifest(path, document)
        manifests[split] = path

    checkpoint = (tmp_path / 'outputs/train/act_insert_plug_v2_rgbd_20261112_161655'
                  '/artifacts/checkpoints/030000/pretrained_model')
    checkpoint.mkdir(parents=True)
    config = dict(
        type='act', chunk_size=64, n_action_steps=64, n_obs_steps=1,
        temporal_ensemble_coeff=None,
        input_features={
            'observation.state': dict(type='STATE', shape=[8]),
            **{f'observation.images.{camera}': dict(type='VISUAL', shape=[3, 256, 256])
               for camera in ('head', 'wrist')},
            **{f'observation.depth.{camera}': dict(type='VISUAL', shape=[1, 256, 256])
               for camera in ('head', 'wrist')},
        },
        output_features={'action': dict(type='ACTION', shape=[8])},
        depth_camera_map={f'observation.images.{camera}': f'observation.depth.{camera}'
                          for camera in ('head', 'wrist')},
        depth_unit='m',
    )
    (checkpoint / 'config.json').write_text(json.dumps(config))
    (checkpoint / 'train_config.json').write_text(json.dumps(dict(
        dataset=dict(root=str(runner.WORKSPACE_ROOT/'datasets/insert_plug_v2/lerobot'),
                     repo_id='local/insert_plug_v2', episodes=None))))
    (checkpoint / 'model.safetensors').write_bytes(b'synthetic ACT weights; never loaded')
    for name in ('policy_preprocessor.json', 'policy_postprocessor.json'):
        (checkpoint / name).write_text('{"steps": []}')
    paths = [scene, training_path, *manifests.values(), *checkpoint.iterdir()]
    return dict(base=base, scene=scene, training=training_path, manifests=manifests,
                checkpoint=checkpoint, checkpoint_config=config,
                original_bytes={path: path.read_bytes() for path in paths})


def synthetic_episode(folder, sources, layout, *, seed, stop_reason='timeout', duration=.2):
    """Write just enough independent evidence for the reusable episode auditor."""
    folder.mkdir(parents=True)
    config = deepcopy(sources['base'])
    config['board'].update({key: layout[key] for key in ('source_xy', 'target_xy')})
    result = dict(
        position=layout['name'], repetition=0, episode_index=0, seed=seed, config=config,
        policy=dict(checkpoint=str(sources['checkpoint']), policy_type='act', n_action_steps=32,
                    checkpoint_n_action_steps=64, chunk_size=64, temporal_ensemble_coeff=None,
                    config_sha256=hashlib.sha256((sources['checkpoint']/'config.json').read_bytes()).hexdigest(),
                    device='cpu', calls=2),
        success=False, task_success=False, contact_clean=None,
        stop_reason=stop_reason,
        runtime_error='ValueError: invalid action' if stop_reason == 'runtime_error' else None,
        stop_on_contact=False, contact_diagnostics=False, contact_policy='off', contact_checks=0,
        contacts=dict(enabled=False, checks=0, unsafe_contact_count=None,
                      first_violation=None, last_violation=None),
        control_hz=10, action_hold_s=.1, physics_timestep_s=.001, requested_duration_s=duration,
        completed_control_steps=2, attempted_control_steps=2, selected_actions=2,
        simulation_seconds=.2, wall_seconds=.7, physics_checks=200,
        video='rollout.mp4', video_frames=3, video_layout='head_left_wrist_right', scene='scene.xml',
        final_metrics=dict(success=False), last_action=[0]*8,
    )
    (folder/'result.json').write_text(json.dumps(result))
    bodies = ''.join(f'<body name="{side}_socket" pos="{layout[side + "_xy"][0]} '
                     f'{layout[side + "_xy"][1]} 0" />' for side in ('source', 'target'))
    (folder/'scene.xml').write_text(
        f'<mujoco><option timestep="0.001"/><worldbody>{bodies}</worldbody></mujoco>')
    (folder/'rollout.mp4').write_bytes(b'synthetic video; decoding covered independently')
    return result


def test_episode_audit_accepts_fresh_layout_seed_without_legacy_times_ten_rule(sources, tmp_path):
    layout = yaml.safe_load(sources['manifests']['ind'].read_text())['positions'][99]
    folder = tmp_path / 'episode'
    result = synthetic_episode(folder, sources, layout, seed=99)
    kwargs = dict(position='pos99', repetition=0, checkpoint=sources['checkpoint'], layout=layout,
                  video_check='none', n_action_steps=32, duration=.2,
                  policy_config=sources['checkpoint_config'])
    assert audit.audit_episode(folder, result, expected_seed=99, **kwargs)['contact_policy'] == 'off'
    with pytest.raises(ValueError, match='seed must be 990'):
        audit.audit_episode(folder, result, **kwargs)
    with pytest.raises(ValueError, match='seed must be 98'):
        audit.audit_episode(folder, result, expected_seed=98, **kwargs)


def arguments(sources, tmp_path, *extra):
    return runner.parse_args([
        '--checkpoint', str(sources['checkpoint']), '--scene-config', str(sources['scene']),
        '--training-manifest', str(sources['training']),
        '--ind-output', str(tmp_path/'ind_results'), '--ood-output', str(tmp_path/'ood_results'),
        '--device', 'cpu', '--backend', 'osmesa', '--workers', '2', *extra,
    ])


@pytest.fixture
def configurations(sources, tmp_path):
    return runner.prepare(arguments(sources, tmp_path))


def test_default_directories_group_splits_under_model(sources, tmp_path, monkeypatch):
    args = arguments(sources, tmp_path, '--output-root', str(tmp_path/'test'))
    args.ind_output = args.ood_output = None
    monkeypatch.setattr(runner, 'now', lambda: '2026-11-11T16:30:45+00:00')
    for split, cfg in runner.prepare(args).items():
        assert Path(cfg['output']) == tmp_path/'test/act_insert_plug_v2_rgbd'/split/'20261112_003045'


def test_prepare_freezes_200_fresh_pairs_one_episode_each_and_paired_seeds(configurations, sources):
    pairs = set()
    for split, config in configurations.items():
        assert config['episodes_requested'] == 100
        assert config['episodes_per_position'] == 1
        assert len(config['positions']) == 100
        assert [row['name'] for row in config['positions']] == [f'pos{i}' for i in range(100)]
        assert [row['seed'] for row in config['positions']] == list(range(100))
        assert [row['episode_index'] for row in config['positions']] == list(range(100))
        assert config['n_action_steps'] == 32 and config['chunk_size'] == 64
        assert config['duration_s'] == 60 and config['control_hz'] == 10 and config['physics_hz'] == 1000
        assert config['temporal_ensemble_coeff'] is None
        assert config['contact_policy'] == 'off'
        assert not config['failed_layouts_replaced'] and not config['interrupted_attempts_retried']
        assert config['policy_settings']['checkpoint_n_action_steps'] == 64
        assert config['policy_settings']['n_action_steps'] == 32
        manifest = yaml.safe_load((Path(config['output'])/'configs/positions_config.yaml').read_text())
        for side in ('source', 'target'):
            center = np.array(sources['base']['board'][f'{side}_xy'])
            points = np.array([row[f'{side}_xy'] for row in config['positions']])
            deviations = np.abs(points-center)
            assert len(np.unique(points, axis=0)) == 100
            if split == 'ind':
                assert (deviations <= .055).all()
                assert (deviations > .02).any(axis=1).all()
            else:
                assert (deviations <= .02).all()
            assert manifest['regions'][side]['outer_side_m'] == .11
            assert manifest['regions'][side]['ood_side_m'] == .04
        pairs.update(tuple(row['source_xy']+row['target_xy']) for row in config['positions'])
    assert len(pairs) == 200
    assert {path: path.read_bytes() for path in sources['original_bytes']} == sources['original_bytes']


def test_prepare_and_resume_preserve_existing_files(configurations, sources, tmp_path):
    protected = {Path(c['output'])/'eval_config.json': (Path(c['output'])/'eval_config.json').read_bytes()
                 for c in configurations.values()}
    with pytest.raises(FileExistsError):
        runner.prepare(arguments(sources, tmp_path))
    assert {path: path.read_bytes() for path in protected} == protected
    restored = runner.load_existing(arguments(sources, tmp_path, '--resume'))
    assert restored == configurations
    assert {path: path.read_bytes() for path in sources['original_bytes']} == sources['original_bytes']


@pytest.mark.parametrize('protected', ['checkpoint', 'dataset'])
def test_output_cannot_write_inside_checkpoint_or_dataset(sources, tmp_path, protected, monkeypatch):
    locations = dict(checkpoint=sources['checkpoint'], dataset=runner.WORKSPACE_ROOT/'datasets/insert_plug_v2/lerobot')
    nested = locations[protected]/'new_evaluation'
    monkeypatch.setattr(Path, 'mkdir', lambda *args, **kwargs: pytest.fail('Unexpected output creation'))
    with pytest.raises(ValueError, match='inside the checkpoint or training dataset'):
        runner.prepare(arguments(sources, tmp_path, '--ind-output', str(nested)))
    assert not nested.exists() and not (tmp_path/'ood_results').exists()
    assert {path: path.read_bytes() for path in sources['original_bytes']} == sources['original_bytes']


@pytest.mark.parametrize('which', ['checkpoint', 'frozen_scene', 'execution_contract',
                                  'checkpoint_config', 'policy_settings'])
def test_provenance_rejects_drift_before_resuming(configurations, which):
    config = configurations['ind']
    if which == 'checkpoint':
        (Path(config['checkpoint'])/'model.safetensors').write_bytes(b'changed weights')
    elif which == 'frozen_scene':
        with (Path(config['output'])/'configs/scene_config.yaml').open('a') as stream:
            stream.write('\n# changed frozen bytes\n')
    elif which == 'execution_contract':
        config['n_action_steps'] = 64
    elif which == 'checkpoint_config':
        config['checkpoint_config']['n_action_steps'] = 32
    else:
        config['policy_settings']['n_action_steps'] = 64
    with pytest.raises(ValueError):
        runner.verify_provenance(config)


def test_resume_rejects_device_or_checkpoint_substitution(configurations, sources, tmp_path):
    with pytest.raises(ValueError, match='device|backend'):
        runner.load_existing(arguments(sources, tmp_path, '--resume', '--device', 'cuda'))
    with pytest.raises(ValueError, match='worker count'):
        runner.load_existing(arguments(sources, tmp_path, '--resume', '--workers', '4'))
    alternate = tmp_path/'other_checkpoint'
    alternate.mkdir()
    for path in sources['checkpoint'].iterdir():
        (alternate/path.name).write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match='checkpoint mismatch'):
        runner.load_existing(arguments(sources, tmp_path, '--resume', '--checkpoint', str(alternate)))


def test_pending_never_retries_physical_runtime_or_interrupted_failures(configurations):
    config = configurations['ind']
    protected = {}
    for i, status in enumerate(('timeout', 'runtime_error')):
        folder = runner.episode_directory(config, config['positions'][i])
        folder.mkdir()
        path = folder/'result.json'
        path.write_text(json.dumps(dict(stop_reason=status, success=False)))
        protected[path] = path.read_bytes()
    folder = runner.episode_directory(config, config['positions'][2])
    folder.mkdir()
    runner.write_infrastructure_failure(folder, config['positions'][2], 'original setup error')
    protected[folder/'infrastructure_failure.json'] = (folder/'infrastructure_failure.json').read_bytes()
    for index in (3, 4):
        folder = runner.episode_directory(config, config['positions'][index])
        folder.mkdir()
        if index == 3:
            (folder/'started.json').write_text('{"started": true}')
    pending = runner.pending_episodes(config)
    assert [row['episode_index'] for row in pending] == list(range(5, 100))
    assert runner.pending_episodes(config) == pending
    assert {path: path.read_bytes() for path in protected} == protected
    for index in (3, 4):
        failure = json.loads((runner.episode_directory(config, config['positions'][index])/
                              'infrastructure_failure.json').read_text())
        assert failure['rerun'] is False and failure['stop_reason'] == 'infrastructure_failure'


def test_summary_keeps_runtime_and_infrastructure_failures_in_full_denominator(configurations):
    config = configurations['ood']
    for i, layout in enumerate(config['positions']):
        folder = runner.episode_directory(config, layout)
        folder.mkdir()
        if i == 99:
            runner.write_infrastructure_failure(folder, layout, 'render setup failed')
        else:
            status = 'success' if i < 20 else 'runtime_error' if i == 98 else 'timeout'
            (folder/'result.json').write_text(json.dumps(
                dict(stop_reason=status, success=i < 20, simulation_seconds=20 if i < 20 else 60)))
    summary = runner.summarize_split(config)
    assert summary['complete'] and summary['episodes_terminal'] == 100
    assert summary['episodes_requested'] == summary['unique_layouts'] == len(summary['results']) == 100
    assert summary['successes'] == 20 and summary['success_rate'] == .2
    assert summary['success_rate_denominator'] == 100
    assert summary['physical_failures'] == 78
    assert summary['runtime_failures'] == summary['infrastructure_failures'] == 1
    assert summary['outcome_counts'] == {'success': 20, 'timeout': 78, 'runtime_error': 1,
                                         'infrastructure_failure': 1}
    assert not summary['artifacts_validated']


def test_one_worker_pool_schedules_each_of_200_layouts_exactly_once(configurations, monkeypatch):
    submitted, pools = [], []

    class Pool:
        def __init__(self, **kwargs):
            pools.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def submit(self, function, config, row):
            assert function is runner.execute_episode
            submitted.append((config['evaluation_split'], row['episode_index'], row['seed']))
            future = Future()
            future.set_result(dict(split=config['evaluation_split'], episode_index=row['episode_index'],
                                   stop_reason='timeout', success=False))
            return future

    monkeypatch.setattr(runner, 'ProcessPoolExecutor', Pool)
    monkeypatch.setattr(runner, 'verify_provenance', lambda config: None)
    monkeypatch.setattr(runner, 'summarize_split', lambda config, **kwargs: {'split': config['evaluation_split']})
    result = runner.run(configurations, workers=2)
    assert set(result) == {'ind', 'ood'}
    assert len(pools) == 1 and pools[0]['max_workers'] == 2
    assert pools[0]['mp_context'].get_start_method() == 'spawn'
    assert len(submitted) == len(set(submitted)) == 200
    assert set(submitted) == {(split, i, i) for split in ('ind', 'ood') for i in range(100)}
    assert submitted == [(split, i, i) for i in range(100) for split in ('ind', 'ood')]


@pytest.mark.parametrize('mutation', ['old_dataset', 'episode_subset'])
def test_prepare_rejects_wrong_or_partial_training_dataset(sources, tmp_path, mutation):
    path = sources['checkpoint']/'train_config.json'
    config = json.loads(path.read_text())
    if mutation == 'old_dataset':
        config['dataset']['root'] = str(runner.WORKSPACE_ROOT/'datasets/insert_plug_v1/lerobot')
    else:
        config['dataset']['episodes'] = [0, 1, 2]
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='complete insert_plug_v2 dataset'):
        runner.prepare(arguments(sources, tmp_path))
    assert not (tmp_path/'ind_results').exists()
    assert not (tmp_path/'ood_results').exists()


def mock_rollout(monkeypatch):
    from examples.insert_plug.policy import rollout
    from examples.insert_plug.environment import scene, simulation
    from mujoco_tools import lerobot_policy
    events = dict(loads=[], seeds=[], runs=[], closed=0, scenes=[])

    class Policy:
        def __init__(self, checkpoint, **kwargs):
            self.checkpoint = checkpoint
            events['loads'].append(kwargs)

        def metadata(self):
            return {'checkpoint': self.checkpoint}

    class Simulation:
        def __init__(self, path, config, **kwargs):
            self.config = config
            events['scenes'].append(deepcopy(config))

        def close(self):
            events['closed'] += 1

    def run_episode(sim, policy, **kwargs):
        events['runs'].append(kwargs)
        return dict(stop_reason='runtime_error', runtime_error='invalid policy action',
                    success=False, simulation_seconds=.1)

    monkeypatch.setattr(runner, '_WORKER_POLICY', None)
    monkeypatch.setattr(lerobot_policy, 'LeRobotPolicy', Policy)
    monkeypatch.setattr(simulation, 'Simulation', Simulation)
    monkeypatch.setattr(scene, 'build_scene', lambda config, path: path.write_text('<mujoco/>'))
    monkeypatch.setattr(rollout, 'validate_contract', lambda *args: None)
    monkeypatch.setattr(rollout, '_seed_policy', events['seeds'].append)
    monkeypatch.setattr(rollout, 'run_episode', run_episode)
    return events


def test_persistent_worker_keeps_one_policy_but_distinct_scenes_and_failures(configurations, monkeypatch):
    events = mock_rollout(monkeypatch)
    rows = [('ind', 0), ('ood', 99)]
    for split, index in rows:
        config = configurations[split]
        row = config['positions'][index]
        result = runner.execute_episode(config, row)
        assert not result['success'] and result['stop_reason'] == 'runtime_error'
        path = runner.episode_directory(config, row)/'result.json'
        original = path.read_bytes()
        saved = json.loads(original)
        assert saved['episode_index'] == 0 and saved['trial_index'] == index and saved['seed'] == index
        assert saved['position'] == f'pos{index}' and saved['split'] == split
        with pytest.raises(FileExistsError):
            runner.execute_episode(config, row)
        assert path.read_bytes() == original
        assert row not in runner.pending_episodes(config)
    assert events['loads'] == [dict(device='cpu', n_action_steps=32)]
    assert events['seeds'] == [0, 99] and events['closed'] == 2
    for kwargs in events['runs']:
        assert kwargs['duration'] == 60
        assert not kwargs['stop_on_contact'] and not kwargs['contact_diagnostics']
        assert not kwargs['viewer'] and not kwargs['realtime']
        assert kwargs['video_path'].name == 'rollout.mp4'
    assert events['scenes'][0] != events['scenes'][1]


def test_setup_failure_becomes_terminal_without_inventing_physics(configurations, monkeypatch):
    events = mock_rollout(monkeypatch)
    from examples.insert_plug.environment import scene

    def fail(*args):
        raise RuntimeError('model asset missing')

    monkeypatch.setattr(scene, 'build_scene', fail)
    config = configurations['ind']
    row = config['positions'][7]
    event = runner.execute_episode(config, row)
    assert event['stop_reason'] == 'infrastructure_failure'
    folder = runner.episode_directory(config, row)
    assert not (folder/'result.json').exists()
    failure = json.loads((folder/'infrastructure_failure.json').read_text())
    assert 'model asset missing' in failure['runtime_error'] and not failure['rerun']
    assert row not in runner.pending_episodes(config)
    assert not events['runs'] and events['closed'] == 0


def region_episode(configurations, sources):
    config = configurations['ood']
    row = config['positions'][99]
    folder = runner.episode_directory(config, row)
    result = synthetic_episode(folder, sources, row, seed=99, stop_reason='runtime_error', duration=60)
    result.update(trial_index=99, split='ood', observation_modalities='rgbd', depth_rendered=True, depth_unit='m')
    result['policy']['depth_camera_map'] = sources['checkpoint_config']['depth_camera_map']
    (folder/'result.json').write_text(json.dumps(result))
    return config, row, folder, result


def test_region_audit_keeps_runtime_error_as_valid_recorded_outcome(configurations, sources):
    config, row, folder, original = region_episode(configurations, sources)
    result, evidence = runner.audit_episode(config, row, video_check='none')
    assert result == original and result['stop_reason'] == 'runtime_error'
    assert evidence['episode_index'] == evidence['seed'] == 99
    assert evidence['position'] == 'pos99' and evidence['split'] == 'ood'
    assert evidence['result_sha256'] == runner.evaluate.fingerprint(folder/'result.json')


@pytest.mark.parametrize('mutation', ['trial_index', 'seed', 'checkpoint_hash', 'scene', 'depth'])
def test_region_audit_rejects_identity_or_execution_drift(configurations, sources, mutation):
    config, row, folder, result = region_episode(configurations, sources)
    if mutation == 'trial_index':
        result['trial_index'] = 0
    elif mutation == 'seed':
        result['seed'] = 990
    elif mutation == 'checkpoint_hash':
        result['policy']['config_sha256'] = 'wrong-checkpoint'
    elif mutation == 'scene':
        result['config']['board']['source_xy'][0] += .001
    else:
        result['depth_rendered'] = False
    (folder/'result.json').write_text(json.dumps(result))
    with pytest.raises(ValueError):
        runner.audit_episode(config, row, video_check='none')
