"""Collect fresh InD expert episodes with 1 kHz torque and 10 Hz RGB-D.

The original expert, model, action timing and ordinary recording paths are
unchanged. Each episode contains its own HDF5/YAML/XML/head-MP4, directly below
the requested bags directory. Existing valid episodes can be resumed only when
their complete configuration and collection-source fingerprints match.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
from datetime import datetime
import hashlib
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import time
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from mujoco_tools.paths import DATA_DIR, WORKSPACE_ROOT
from .collect_ring_dataset import episode_config as ring_episode_config, validate_manifest
from .torque_recording import IMAGE_HZ, TORQUE_HZ, TORQUE_SCHEMA, SAMPLES_PER_IMAGE


SOURCE_PATHS = (
    'examples/insert_plug/data/collect_torque_dataset.py',
    'examples/insert_plug/data/torque_recording.py',
    'src/mujoco_tools/torque_signals.py',
    'examples/insert_plug/data/dataset.py',
    'examples/insert_plug/data/recording.py',
    'examples/insert_plug/data/replay.py',
    'examples/insert_plug/environment/simulation.py',
    'examples/insert_plug/environment/scene.py',
    'examples/insert_plug/environment/objects.py',
    'examples/insert_plug/control/expert.py',
    'examples/insert_plug/control/planning.py',
    'src/mujoco_tools/robotiq.py',
)


def source_fingerprints():
    return {name: hashlib.sha256((WORKSPACE_ROOT / name).read_bytes()).hexdigest()
            for name in SOURCE_PATHS}


def episode_config(base, manifest, position, manifest_hash, fingerprints):
    config = ring_episode_config(base, manifest, position, manifest_hash)
    if not np.isclose(config['physics']['timestep'], 1/TORQUE_HZ, rtol=0, atol=1e-12):
        raise ValueError('Torque collection requires 1 ms physics steps')
    config['recording'].update(sample_hz=IMAGE_HZ, torque_sample_hz=TORQUE_HZ,
                               torque_cutoff_hz=30.0)
    config['collection'].update(
        dataset_version='insert_plug_torque_v1', torque_schema_version=TORQUE_SCHEMA,
        source_dataset_layout='insert_plug_v2', freshly_executed=True,
        source_sha256=copy.deepcopy(fingerprints),
        mujoco_version=importlib.metadata.version('mujoco'),
        torque_time_interval='[0, final_simulation_time)',
        force_feedback_enabled=False,
    )
    return config


def existing_episode(directory, stem, config):
    from .torque_recording import validate_torque_episode
    if not directory.exists():
        return None
    if directory.is_symlink() or not directory.is_dir():
        raise RuntimeError(f'Unexpected episode path: {directory}')
    expected = {f'{stem}.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
    if {path.name for path in directory.iterdir()} != expected:
        raise RuntimeError(f'Incomplete/extraneous files in {directory}; preserved for inspection')
    if any(path.is_symlink() or not path.is_file() for path in directory.iterdir()):
        raise RuntimeError('Episode files must be ordinary files')
    saved = yaml.safe_load((directory / f'{stem}.yaml').read_text())
    if saved != config:
        raise RuntimeError(f'Existing episode configuration differs: {directory}')
    result = validate_torque_episode(directory / f'{stem}.h5')
    if not result['valid'] or not result['success']:
        raise RuntimeError(f'Existing episode validation failed: {result["errors"]}')
    if result['metadata'].get('config') != config:
        raise RuntimeError('HDF5 configuration differs from YAML')
    if result['metadata'].get('model_sha256') != hashlib.sha256((directory / f'{stem}.xml').read_bytes()).hexdigest():
        raise RuntimeError('Saved model hash differs')
    return result


def record_episode(config, episode_id, output, backend):
    """Run the original Expert with a read-only observer added to every tick."""
    from ..cli import _configure_gl
    _configure_gl(backend)
    import mujoco
    from mujoco_tools.torque_signals import TorqueSignalSampler
    from ..environment.scene import build_scene
    from ..environment.simulation import Simulation
    from ..control.expert import Expert
    from .replay import generate_head_video
    from .torque_recording import TorqueEpisodeWriter, validate_torque_episode

    class RecordingSimulation(Simulation):
        torque_callback = None

        def step(self, action, *, before_substep=None):
            def observe(active):
                if self.torque_callback is not None:
                    self.torque_callback(active)
                if before_substep is not None:
                    before_substep(active)
            return super().step(action, before_substep=observe)

    stem = f'episode_{episode_id:06d}'
    folder = Path(output) / stem
    folder.mkdir()  # Exclusive: a partial directory is never overwritten.
    scene_path = build_scene(config, folder / f'{stem}.xml')
    (folder / f'{stem}.yaml').write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
    sim = RecordingSimulation(scene_path, config, render=True)
    writer = None
    try:
        expert = Expert(sim, realtime=False, verbose=False)
        expert.prepare()
        sampler = TorqueSignalSampler(sim, cutoff_hz=config['recording']['torque_cutoff_hz'])
        metadata = dict(
            seed=episode_id, config=config, task='insert_plug',
            model_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
            model_path=scene_path.name, mujoco_version=mujoco.__version__, renderer_backend=backend,
            expert='Smooth Cartesian plans with measured-grasp suffix and adaptive execution gates',
            strict_collision_checks=bool(config['task'].get('planning', {}).get('strict_collision_checks', False)),
            robot_gravity_compensation=config['robot']['gravity_compensation'],
            geometry_status='user STL visuals and decomposed collisions; SI scale inferred from mm; mass and friction are configurable assumptions',
            policy_inputs=['observation.images.head', 'observation.images.wrist', 'observation.state'],
            depth_enabled=sim.depth_enabled,
            depth_features=['observation.depth.head', 'observation.depth.wrist'],
            camera_calibration=sim.camera_calibration(),
            video_camera='head', video_file=f'{stem}.mp4',
            torque_recording=dict(
                schema_version=TORQUE_SCHEMA, sample_hz=TORQUE_HZ,
                image_hz=IMAGE_HZ, samples_per_image=SAMPLES_PER_IMAGE,
                time_interval='[initial_time, final_time)',
                timing='before_physics_integration_after_active_control_target_applied',
                image_to_torque_index_path='/alignment/image_to_torque_index',
                method=sampler.metadata(),
                force_feedback_enabled=False,
                simulation_truth_is_not_real_robot_measurement=True,
            ),
        )
        metadata.update(expert.recording_sampler.metadata(sim.control_hz))
        for section, name in (('robot', 'robot_asset_manifest'), ('gripper', 'gripper_asset_manifest')):
            model_dir = Path(config[section]['model_dir'])
            if not model_dir.is_absolute():
                model_dir = WORKSPACE_ROOT / model_dir
            path = model_dir / 'SOURCE.json'
            if path.is_file():
                metadata[name] = json.loads(path.read_text())
        metadata['object_assets'] = {}
        for name in ('board', 'peg'):
            for key in ('visual_mesh', 'collision_manifest'):
                value = config[name].get(key)
                if value:
                    asset = WORKSPACE_ROOT / value
                    metadata['object_assets'][f'{name}/{key}'] = dict(
                        path=str(asset), sha256=hashlib.sha256(asset.read_bytes()).hexdigest())
        writer = TorqueEpisodeWriter(folder / f'{stem}.h5', metadata)

        def observe_torque(action):
            sample = sampler.sample(action)
            sample.update(
                action=np.asarray(action, dtype=np.float64).copy(),
                ctrl=sim.data.ctrl.copy(), phase=expert.phase,
                image_index=np.int64(writer.torque_samples // SAMPLES_PER_IMAGE),
                action_index=np.int64(expert.steps),
            )
            writer.append_torque(sample)

        sim.torque_callback = observe_torque
        expert.writer = writer
        result = expert.run()
        result.update(seed=episode_id, backend=backend, model=scene_path.name,
                      recorded_frames=writer.steps, torque_samples=writer.torque_samples)
        path = writer.finalize(success=result['success'], failure_reason=result['failure_reason'],
                               final_metrics=result['metrics'], execution_result=copy.deepcopy(result))
        validation = validate_torque_episode(path)
        if not validation['valid']:
            raise RuntimeError(f'Recorded data failed validation: {validation["errors"]}')
        generate_head_video(path, folder / f'{stem}.mp4')
        if not result['success']:
            raise RuntimeError(f'Expert failed: {result["failure_reason"]}')
        return validation
    finally:
        if writer is not None:
            writer.close()
        sim.close()


def collect_one(position, base, manifest, manifest_hash, fingerprints, output, backend):
    started = time.monotonic()
    episode_id = int(position['episode_id'])
    stem = f'episode_{episode_id:06d}'
    try:
        config = episode_config(base, manifest, position, manifest_hash, fingerprints)
        folder = Path(output) / stem
        validation = existing_episode(folder, stem, config)
        resumed = validation is not None
        if validation is None:
            validation = record_episode(config, episode_id, output, backend)
        expected = {f'{stem}.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
        if {path.name for path in folder.iterdir()} != expected:
            raise RuntimeError('Episode does not contain exactly HDF5, YAML, XML and head MP4')
        return dict(
            episode=stem, position=position['name'], success=True, resumed=resumed,
            frames=validation['steps'], torque_samples=validation['torque_samples'],
            simulation_seconds=validation['duration_s'],
            bytes=sum(path.stat().st_size for path in folder.iterdir()),
            wall_seconds=time.monotonic() - started,
        )
    except Exception as exc:
        return dict(episode=stem, position=position['name'], success=False,
                    error=f'{type(exc).__name__}: {exc}', wall_seconds=time.monotonic() - started)


def _write_frozen(path, payload):
    if path.exists():
        if not path.is_file() or path.is_symlink() or path.read_bytes() != payload:
            raise FileExistsError(f'Existing provenance file differs: {path}')
        return
    with path.open('xb') as stream:
        stream.write(payload)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'insert_plug_torque/bags')
    parser.add_argument('--report', type=Path, help='Progress JSON, default: dataset parent/collection_report.json')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--backend', choices=['egl', 'osmesa'], default='osmesa')
    parser.add_argument('--episode', type=int, action='append', help='Collect only these IDs in 1..100')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    manifest_path = args.manifest.expanduser().resolve()
    manifest_bytes = manifest_path.read_bytes()
    manifest = yaml.safe_load(manifest_bytes)
    manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
    base_path = WORKSPACE_ROOT / manifest['base_config']
    base = yaml.safe_load(base_path.read_text())
    validate_manifest(manifest, base, base_path)
    fingerprints = source_fingerprints()
    rows = manifest['positions']
    if args.episode:
        requested = set(args.episode)
        if not requested <= set(range(1, 101)):
            parser.error('Episode IDs must be in 1..100')
        rows = [row for row in rows if row['episode_id'] in requested]
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    allowed = {f'episode_{i:06d}' for i in range(1, 101)}
    if any(not path.is_dir() or path.is_symlink() or path.name not in allowed for path in output.iterdir()):
        raise ValueError('Output contains unexpected entries; preserving them without changes')
    _write_frozen(output.parent / 'layout_manifest.yaml', manifest_bytes)
    _write_frozen(output.parent / 'collection_source_sha256.json',
                  (json.dumps(fingerprints, indent=2, sort_keys=True)+'\n').encode())
    report_path = (args.report.expanduser().resolve() if args.report else output.parent / 'collection_report.json')
    if report_path.is_relative_to(output):
        parser.error('Progress report must be outside bags/')
    report_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(
        started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
        dataset='insert_plug_torque_v1', split='ind', output=str(output),
        manifest=str(manifest_path), manifest_sha256=manifest_hash, source_sha256=fingerprints,
        workers=args.workers, backend=args.backend, image_hz=IMAGE_HZ, torque_hz=TORQUE_HZ,
        requested_episodes=len(rows), complete=False, results=[],
    )

    def save():
        report.update(successes=sum(row['success'] for row in report['results']),
                      frames=sum(row.get('frames', 0) for row in report['results']),
                      torque_samples=sum(row.get('torque_samples', 0) for row in report['results']),
                      wall_seconds=time.monotonic() - started)
        temporary = report_path.with_name(report_path.name + '.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temporary.replace(report_path)

    save()
    print(json.dumps(dict(event='collection_start', episodes=len(rows), output=str(output))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(collect_one, row, base, manifest, manifest_hash, fingerprints, output, args.backend)
                   for row in rows]
        for future in as_completed(futures):
            result = future.result()
            report['results'].append(result)
            save()
            print(json.dumps(dict(event='episode_complete', **result)), flush=True)
    report['results'].sort(key=lambda row: row['episode'])
    report['complete'] = len(report['results']) == len(rows) and all(row['success'] for row in report['results'])
    if manifest_path.read_bytes() != manifest_bytes or source_fingerprints() != fingerprints:
        raise RuntimeError('Collection source or layout manifest changed during execution')
    save()
    print(json.dumps(dict(event='collection_finished', complete=report['complete'],
                          successes=report['successes'], frames=report['frames'],
                          torque_samples=report['torque_samples'])), flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
