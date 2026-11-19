"""Collect one successful RGB-D episode per validated v2 ring layout.

Episodes are placed directly below bags/episode_000001, without pos folders.
Existing complete episodes can be resumed only with exactly matching configs.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
import copy
from datetime import datetime
import hashlib
import io
import json
import multiprocessing
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from mujoco_tools.paths import DATA_DIR, WORKSPACE_ROOT
from .collect_positions import existing_episode
from ..environment.ring_layouts import ring_contains


def episode_config(base, manifest, position, manifest_hash):
    config = copy.deepcopy(base)
    for side in ('source', 'target'):
        config['board'][f'{side}_xy'] = list(position[f'{side}_xy'])
    config['recording']['sample_hz'] = float(manifest['recording_hz'])
    config['task']['control_hz'] = float(manifest['control_hz'])
    config['cameras']['depth']['enabled'] = True
    config['collection'] = dict(
        dataset_version='insert_plug_v2', split='ind', episode_id=position['episode_id'],
        position_id=position['name'], episodes_per_position=1,
        layout_fixed_within_episode=True, layout_manifest_sha256=manifest_hash,
        sampling_seed=manifest['selection_seed'], regions=manifest['regions'],
        source_stratum=position['source_stratum'], target_stratum=position['target_stratum'],
        source_and_target_unique_across_episodes=True,
    )
    return config


def validate_manifest(manifest, base, base_path):
    if not all(manifest.get(key) is True for key in ('geometry_frozen', 'selection_complete', 'physics_validated')):
        raise ValueError('Freeze geometry and complete every physical preflight before collection')
    if hashlib.sha256(Path(base_path).read_bytes()).hexdigest() != manifest['base_config_sha256']:
        raise ValueError('Base scene configuration changed after validation')
    rows = manifest['positions']
    if len(rows) != 100 or sorted(row['episode_id'] for row in rows) != list(range(1, 101)):
        raise ValueError('Expected exactly episode IDs 1 through 100')
    if len({row['name'] for row in rows}) != 100:
        raise ValueError('Position names must be unique')
    for side in ('source', 'target'):
        points = np.asarray([row[f'{side}_xy'] for row in rows])
        if len(np.unique(points, axis=0)) != 100:
            raise ValueError(f'{side} positions are not unique')
        if not ring_contains(points, base['board'][f'{side}_xy']).all():
            raise ValueError(f'{side} point lies outside the InD ring')
    if not all(row.get('physics_check', {}).get('success') is True for row in rows):
        raise ValueError('One or more layouts lack successful expert preflight evidence')
    if manifest['recording_hz'] != 10 or manifest['control_hz'] != 10 or not manifest['depth_enabled']:
        raise ValueError('This dataset requires synchronized 10 Hz RGB-D and actions')
    if base['cameras']['width'] != 256 or base['cameras']['height'] != 256:
        raise ValueError('Preserve the current native 256 x 256 cameras')


def collect_one(position, base, manifest, manifest_hash, output, backend):
    from examples.insert_plug.cli import collect
    from examples.insert_plug.data.dataset import validate_episode

    started = time.monotonic()
    config = episode_config(base, manifest, position, manifest_hash)
    episode_id = int(position['episode_id'])
    stem = f'episode_{episode_id:06d}'
    directory = Path(output) / stem
    log = io.StringIO()
    try:
        resumed = existing_episode(directory, stem, config) is not None
        if not resumed:
            with TemporaryDirectory(prefix=f'insert-plug-v2-{stem}-') as temporary:
                config_path = Path(temporary) / 'config.yaml'
                config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf8')
                args = SimpleNamespace(
                    command='collect', config=str(config_path), output=str(output), episodes=1,
                    seed=episode_id, randomize_mm=0., backend=backend, width=None,
                    control_hz=None, sample_hz=10., no_images=False, viewer=False,
                    realtime=False, quiet=True, video=False,
                )
                with redirect_stdout(log):
                    status = collect(args)
                if status != 0:
                    raise RuntimeError(f'Expert recording returned status {status}')
        report = validate_episode(directory / f'{stem}.h5')
        if not report['valid'] or not report['success']:
            raise RuntimeError('Recording failed dataset validation or physical task success')
        expected = {f'{stem}.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
        if {path.name for path in directory.iterdir()} != expected:
            raise RuntimeError('Episode must contain exactly HDF5, YAML, XML and head MP4')
        return dict(episode=stem, position=position['name'], success=True, resumed=resumed,
                    frames=report['steps'], simulation_seconds=report['metadata']['execution_result']['steps'] / 10,
                    bytes=sum(path.stat().st_size for path in directory.iterdir()),
                    wall_seconds=time.monotonic() - started)
    except Exception as error:
        return dict(episode=stem, position=position['name'], success=False,
                    error=f'{type(error).__name__}: {error}', diagnostic=log.getvalue()[-6000:],
                    wall_seconds=time.monotonic() - started)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--output', type=Path, default=DATA_DIR / 'insert_plug_v2/bags')
    parser.add_argument('--report', type=Path, default=WORKSPACE_ROOT / 'outputs/_checks/insert_plug_v2_collection/collection_progress.json')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--backend', choices=['egl', 'osmesa'], default='egl')
    parser.add_argument('--episode', type=int, action='append', help='Collect only these IDs, e.g. --episode 1')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    manifest = yaml.safe_load(args.manifest.read_text())
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    base_path = WORKSPACE_ROOT / manifest['base_config']
    base = yaml.safe_load(base_path.read_text())
    validate_manifest(manifest, base, base_path)
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
    args.report.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    report = dict(started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  dataset='insert_plug_v2', output=str(output), manifest=str(args.manifest.resolve()),
                  manifest_sha256=manifest_hash, workers=args.workers, backend=args.backend,
                  requested_episodes=len(rows), complete=False, results=[])

    def save():
        report.update(successes=sum(row['success'] for row in report['results']),
                      frames=sum(row.get('frames', 0) for row in report['results']),
                      wall_seconds=time.monotonic() - started)
        temporary = args.report.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.report)

    save()
    print(json.dumps(dict(event='collection_start', episodes=len(rows), output=str(output))), flush=True)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(collect_one, row, base, manifest, manifest_hash, output, args.backend)
                   for row in rows]
        for future in as_completed(futures):
            result = future.result()
            report['results'].append(result)
            save()
            print(json.dumps(dict(event='episode_complete', **result)), flush=True)
    report['results'].sort(key=lambda row: row['episode'])
    report['complete'] = len(report['results']) == len(rows) and all(row['success'] for row in report['results'])
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != manifest_hash:
        raise RuntimeError('Layout manifest changed during collection')
    save()
    print(json.dumps(dict(event='collection_finished', complete=report['complete'],
                          successes=report['successes'], frames=report['frames'])), flush=True)
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
