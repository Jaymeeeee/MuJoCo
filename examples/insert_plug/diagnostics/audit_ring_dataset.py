"""Read-only audit of the flat, 100-layout insert_plug_v2 RGB-D dataset.

The optional replay executes saved actions through normal MuJoCo physics; it
does not call the expert, restore intermediate states, or render new videos.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import hashlib
import json
import multiprocessing
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import yaml

from mujoco_tools.paths import DATA_DIR, WORKSPACE_ROOT
from .audit_batch import audit_episode, require


CENTRES = {'source': np.array([.45, -.07]), 'target': np.array([.45, .07])}
OUTER_HALF_M = .055
INNER_HALF_M = .02
EPISODE_COUNT = 100


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require_ring_position(xy, side):
    """The closed central square, including its boundary, is never training."""
    point = np.asarray(xy, dtype=float)
    require(point.shape == (2,) and np.isfinite(point).all(), f'Invalid {side} XY')
    distance = float(np.max(np.abs(point - CENTRES[side])))
    require(distance <= OUTER_HALF_M + 1e-12, f'{side} lies outside the 11 cm outer square')
    require(distance > INNER_HALF_M + 1e-12, f'{side} lies in/on the held-out 4 cm central square')
    return tuple(np.round(point, 12))


def validate_positions(manifest):
    rows = manifest['positions']
    require(len(rows) == EPISODE_COUNT, 'Manifest must contain exactly 100 layouts')
    require({row['episode_id'] for row in rows} == set(range(1, EPISODE_COUNT + 1)),
            'Episode IDs must be exactly 1..100')
    require({row['name'] for row in rows} == {f'pos{i}' for i in range(EPISODE_COUNT)},
            'Position names must be exactly pos0..pos99')
    seen = {side: set() for side in CENTRES}
    for row in rows:
        require(type(row['episode_id']) is int, 'Episode ID must be an integer')
        for side in CENTRES:
            key = require_ring_position(row[f'{side}_xy'], side)
            require(key not in seen[side], f'Duplicate {side} coordinate in manifest')
            seen[side].add(key)
    return sorted(rows, key=lambda row: row['episode_id'])


def validate_stratification(rows):
    """Reconstruct square-ring strata from XY rather than trust their labels."""
    seen = {side: set() for side in CENTRES}
    combinations = np.zeros((4, 4), dtype=int)
    for row in rows:
        faces = {}
        for side in CENTRES:
            dx, dy = np.asarray(row[f'{side}_xy']) - CENTRES[side]
            radius = max(abs(dx), abs(dy))
            face = (0 if dx > 0 else 2) if abs(dx) >= abs(dy) else (1 if dy > 0 else 3)
            tangent = (dy, -dx, -dy, dx)[face] / radius
            radial_bin = min(4, int(np.floor(5 * (radius**2 - INNER_HALF_M**2)
                                               / (OUTER_HALF_M**2 - INNER_HALF_M**2))))
            tangent_bin = min(4, int(np.floor(5 * (tangent + 1) / 2)))
            actual = (face, radial_bin, tangent_bin)
            expected = row[f'{side}_stratum']
            require(actual == tuple(expected[key] for key in ('face', 'radial_bin', 'tangent_bin')),
                    f'{side} stratum label does not match its actual XY')
            require(actual not in seen[side], f'{side} has repeated spatial sampling strata')
            seen[side].add(actual)
            faces[side] = face
        combinations[faces['source'], faces['target']] += 1
    expected_strata = {(face, radial, tangent) for face in range(4)
                       for radial in range(5) for tangent in range(5)}
    require(all(values == expected_strata for values in seen.values()),
            'Each ring must cover all 100 equal-area strata exactly once')
    require(np.isin(combinations, [6, 7]).all()
            and (combinations.sum(axis=0) == 25).all() and (combinations.sum(axis=1) == 25).all(),
            'Source/target face combinations must be balanced at 6 or 7 pairs each')
    return dict(equal_area_strata_per_ring=100, strata_area_cm2=1.05,
                source_target_face_counts=combinations.tolist())


def validate_inventory(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), 'Dataset root must be a regular directory')
    entries = list(root.iterdir())
    expected = {f'episode_{i:06d}' for i in range(1, EPISODE_COUNT + 1)}
    require({entry.name for entry in entries} == expected,
            'Dataset must contain exactly episode_000001 through episode_000100')
    require(all(entry.is_dir() and not entry.is_symlink() for entry in entries),
            'Episode entries must be regular directories')


def full_video_decode(path, expected_frames, width=256, height=256):
    import av

    count = 0
    with av.open(str(path), mode='r') as container:
        require(len(container.streams.video) == 1, 'Expected one head-video stream')
        stream = container.streams.video[0]
        require(stream.average_rate == 10, 'Head video must run at 10 fps')
        for frame in container.decode(stream):
            require((frame.width, frame.height) == (width, height), 'Decoded video size differs')
            require(frame.to_ndarray(format='rgb24').shape == (height, width, 3),
                    'Decoded frame is not RGB')
            count += 1
    require(count == expected_frames, 'Fully decoded video frame count differs from HDF5')
    return count


def check_saved_scene(folder, config, row):
    import mujoco
    from ..environment.layouts import head_visibility
    from ..environment.simulation import Simulation

    stem = folder.name
    sim = Simulation(folder / f'{stem}.xml', config, render=False)
    try:
        for side in CENTRES:
            require(np.allclose(sim.site(f'{side}_hole')[:2], row[f'{side}_xy'], atol=1e-10, rtol=0),
                    f'Compiled XML {side} hole differs from the frozen manifest')
        views = {'initial': head_visibility(sim, 5.0)}
        with h5py.File(folder / f'{stem}.h5', 'r') as handle:
            sim.data.qpos[:] = handle['simulation/qpos'][-1]
            sim.data.qvel[:] = handle['simulation/qvel'][-1]
            sim.data.time = float(handle['timestamp'][-1])
        mujoco.mj_forward(sim.model, sim.data)
        views['last_recorded_frame'] = head_visibility(sim, 5.0)
        require(all(view['all_inside'] for view in views.values()),
                'A saved object mesh leaves the head-camera frustum')
        return dict(required_margin_px=5.0, occlusion_checked=False,
                    states=list(views), minimum_edge_margin_px=min(
                        part['edge_margin_px'] for view in views.values() for part in view['objects'].values()))
    finally:
        sim.close()


def audit_recording(root, row, *, max_duration_s=None, replay=False):
    folder = Path(root) / f"episode_{row['episode_id']:06d}"
    result = dict(episode=folder.name, position=row['name'], valid=False, errors=[])
    try:
        record = audit_episode(folder, sample_hz=10.0, control_hz=10.0, max_duration_s=max_duration_s)
        config = yaml.safe_load((folder / f'{folder.name}.yaml').read_text())
        require(config['cameras']['width'] == 256 and config['cameras']['height'] == 256,
                'Expected native 256x256 RGB-D cameras')
        require(record['collision_checks_enabled'] is False and record['unsafe_contacts'] is None,
                'Contact-threshold diagnostics must remain disabled/unknown')
        for side in CENTRES:
            require_ring_position(record[f'{side}_xy'], side)
            require(np.allclose(record[f'{side}_xy'], row[f'{side}_xy'], atol=1e-10, rtol=0),
                    f'Recorded {side} differs from the frozen manifest')
        result['recording'] = record
        result['head_visibility'] = check_saved_scene(folder, config, row)
        result['fully_decoded_video_frames'] = full_video_decode(
            folder / f'{folder.name}.mp4', record['frames'])
        before = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                  for path in sorted(folder.iterdir())}
        result['files'] = before
        if replay:
            from .check_action_replay import replay_episode

            outcome = replay_episode(folder / f'{folder.name}.h5', action_dtype='float32',
                                     contact_diagnostics=False)
            outcome['position'] = row['name']
            result['action_replay'] = outcome
            require(outcome['success'] is True and outcome['runtime_error'] is None,
                    'Saved-action replay did not complete successfully')
            require(outcome['max_joint_deviation_from_source_rad'] <= 1e-10
                    and outcome['max_peg_deviation_from_source_m'] <= 1e-10,
                    'Saved-action replay differs from recorded joint/peg states')
            require(outcome['final']['final_stability_physics_samples'] == 251,
                    'Replay is missing final-quarter-second stability samples')
            require(outcome['contact_clean'] is None and outcome['contacts']['enabled'] is False,
                    'Disabled replay contact checks must not claim zero violations')
            require(all(sha256(folder / name) == entry['sha256'] for name, entry in before.items()),
                    'Source recording changed during replay')
        result['valid'] = True
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def summarize(results):
    recordings = [row['recording'] for row in results if 'recording' in row]
    replays = [row['action_replay'] for row in results if 'action_replay' in row]
    durations = [row['simulation_duration_s'] for row in recordings]
    return dict(episodes=len(results), passed=sum(row['valid'] for row in results),
                total_frames=sum(row['frames'] for row in recordings),
                fully_decoded_video_frames=sum(row.get('fully_decoded_video_frames', 0) for row in results),
                total_bytes=sum(row['bytes'] for row in recordings),
                duration_min_s=min(durations) if durations else None,
                duration_max_s=max(durations) if durations else None,
                duration_mean_s=float(np.mean(durations)) if durations else None,
                episodes_at_least_20_seconds=sum(value >= 20 for value in durations),
                action_replay_episodes=len(replays),
                action_replay_passed=sum(row['success'] for row in replays),
                replay_max_joint_deviation_rad=max((row['max_joint_deviation_from_source_rad']
                                                    for row in replays), default=None),
                replay_max_peg_deviation_m=max((row['max_peg_deviation_from_source_m']
                                              for row in replays), default=None),
                collision_checked_episodes=sum(row['collision_checks_enabled'] for row in recordings),
                unsafe_contact_count=None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DATA_DIR / 'insert_plug_v2/bags')
    parser.add_argument('--manifest', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--output', type=Path, default=WORKSPACE_ROOT / 'outputs/_checks/insert_plug_v2_collection/raw_audit.json')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--max-duration-s', type=float, help='Optional strict upper bound; no limit by default')
    parser.add_argument('--replay', action='store_true', help='Also replay every saved action through physics')
    args = parser.parse_args(argv)
    if args.workers < 1 or (args.max_duration_s is not None
                           and (not np.isfinite(args.max_duration_s) or args.max_duration_s <= 0)):
        parser.error('Workers and duration limit must be positive')
    if args.output.exists():
        parser.error('Audit report already exists; choose a new output path')
    root = args.root.resolve()
    output = args.output.resolve()
    if output.is_relative_to(root):
        parser.error('Audit output must be outside the read-only dataset root')
    output.parent.mkdir(parents=True, exist_ok=True)
    report = dict(check='insert_plug_v2_ring_dataset', root=str(root), manifest=str(args.manifest.resolve()),
                  started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  expected_episodes=EPISODE_COUNT, replay_requested=args.replay,
                  max_duration_s=args.max_duration_s, complete=False, valid=False, errors=[], results=[])

    def save():
        report['summary'] = summarize(report['results'])
        temporary = output.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
        temporary.replace(output)

    try:
        manifest = yaml.safe_load(args.manifest.read_text())
        report['manifest_sha256'] = sha256(args.manifest)
        rows = validate_positions(manifest)
        stratification = validate_stratification(rows)
        validate_inventory(root)
        base_path = Path(manifest['base_config'])
        if not base_path.is_absolute():
            base_path = WORKSPACE_ROOT / base_path
        require(sha256(base_path) == manifest['base_config_sha256'],
                'Base configuration differs from the frozen manifest')
        report['base_config_sha256'] = manifest['base_config_sha256']
        report['layout_validation'] = dict(unique_source_positions=EPISODE_COUNT,
                                           unique_target_positions=EPISODE_COUNT,
                                           all_positions_in_ring=True, central_boundary_excluded=True,
                                           **stratification)
        save()
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=multiprocessing.get_context('spawn')) as executor:
            futures = [executor.submit(audit_recording, root, row, max_duration_s=args.max_duration_s,
                                       replay=args.replay) for row in rows]
            for future in as_completed(futures):
                result = future.result()
                report['results'].append(result)
                if not result['valid']:
                    report['errors'].append(dict(episode=result['episode'], errors=result['errors']))
                save()
                print(json.dumps(dict(episode=result['episode'], valid=result['valid'],
                                      completed=len(report['results']), errors=result['errors'])), flush=True)
        report['results'].sort(key=lambda row: row['episode'])
        report['complete'] = len(report['results']) == EPISODE_COUNT
        report['valid'] = report['complete'] and not report['errors']
    except Exception as error:
        report['errors'].append(f'{type(error).__name__}: {error}')
    save()
    print(json.dumps(dict(report=str(output), complete=report['complete'], valid=report['valid'],
                          **report['summary'])), flush=True)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
