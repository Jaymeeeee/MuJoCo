"""Freeze fresh v2 InD/OOD evaluation geometry without outcome-based selection.

Each split contains 100 new source/target pairs and one episode per pair. InD
uses the training ring's equal-area stratification and balanced face pairing.
OOD independently samples both central squares uniformly. All sampled layouts
remain in the evaluation, including layouts on which physics or policies fail.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from mujoco_tools.paths import WORKSPACE_ROOT
from .ring_layouts import region_spec, ring_contains, sample_layouts
from .scene import load_config


COUNT = 100
DEFAULT_SEEDS = {'ind': 2026111201, 'ood': 2026111202}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _display_path(path):
    path = Path(path).expanduser().resolve()
    return str(path.relative_to(WORKSPACE_ROOT)) if path.is_relative_to(WORKSPACE_ROOT) else str(path)


def _points(rows, side):
    try:
        points = np.asarray([row[f'{side}_xy'] for row in rows], dtype=float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f'Invalid {side} coordinates') from error
    if points.shape != (COUNT, 2) or not np.isfinite(points).all():
        raise ValueError(f'{side} must contain 100 finite XY points')
    if len(np.unique(points, axis=0)) != COUNT:
        raise ValueError(f'{side} coordinates must be unique')
    return points


def square_contains(points, bounds):
    """Membership of the closed central OOD square, including its boundary."""
    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError('Points must have final dimension two')
    limits = np.asarray([bounds['x'], bounds['y']], dtype=float)
    return (np.isfinite(values).all(axis=-1)
            & np.all((values >= limits[:, 0]) & (values <= limits[:, 1]), axis=-1))


def _training_sources(base_config, base_config_path, training_manifest_path):
    base_path = Path(base_config_path).expanduser().resolve()
    training_path = Path(training_manifest_path).expanduser().resolve()
    if yaml.safe_load(base_path.read_text(encoding='utf8')) != base_config:
        raise ValueError('base_config must exactly match base_config_path')
    training = yaml.safe_load(training_path.read_text(encoding='utf8'))
    if (not isinstance(training, dict) or training.get('format_version') != 2
            or training.get('dataset_version') != 'insert_plug_v2' or training.get('split') != 'ind'):
        raise ValueError('Expected the insert_plug_v2 training ring manifest')
    if training.get('base_config_sha256') != _sha256(base_path):
        raise ValueError('Base configuration hash differs from the training manifest')
    if not training.get('geometry_frozen') or not training.get('selection_complete'):
        raise ValueError('Training geometry must already be frozen and complete')
    if training.get('expected_episodes') != COUNT or training.get('episodes_per_position') != 1:
        raise ValueError('Training manifest must describe 100 one-episode layouts')
    sampling = training.get('sampling', {})
    if (sampling.get('method') != 'equal_area_square_ring_stratification'
            or sampling.get('faces') != 4 or sampling.get('radial_squared_bins') != 5
            or sampling.get('tangent_bins') != 5 or sampling.get('points_per_region') != COUNT
            or sampling.get('points_per_stratum') != 1):
        raise ValueError('Training sampling specification differs from the v2 ring design')
    training_seed = sampling.get('seed')
    if type(training_seed) is not int or training_seed < 0 or training.get('selection_seed') != training_seed:
        raise ValueError('Training manifest must record one consistent nonnegative sampling seed')
    rows = training.get('positions', [])
    if [row.get('name') for row in rows] != [f'pos{i}' for i in range(COUNT)]:
        raise ValueError('Training positions must be pos0 through pos99 in order')
    for side in ('source', 'target'):
        center = base_config['board'][f'{side}_xy']
        expected = region_spec(center)
        if training.get(f'base_{side}_xy') != list(center) or training.get('regions', {}).get(side) != expected:
            raise ValueError(f'{side} region bounds differ from the training/base configuration')
        if not ring_contains(_points(rows, side), center).all():
            raise ValueError(f'Training {side} points must lie in the declared ring')
    return training


def _sample_positions(base_config, regions, split, seed):
    if split == 'ind':
        rows = sample_layouts(base_config, seed)
    elif split == 'ood':
        source_seed, target_seed = np.random.SeedSequence(seed).spawn(2)
        points = {}
        for side, stream in (('source', source_seed), ('target', target_seed)):
            bounds = regions[side]['ood_bounds_m']
            limits = np.asarray([bounds['x'], bounds['y']], dtype=float)
            points[side] = np.random.default_rng(stream).uniform(limits[:, 0], limits[:, 1], (COUNT, 2))
        rows = [dict(source_xy=points['source'][i].tolist(), target_xy=points['target'][i].tolist())
                for i in range(COUNT)]
    else:
        raise ValueError("split must be 'ind' or 'ood'")
    for index, row in enumerate(rows):
        row.update(name=f'pos{index}', seed=index, episode_id=index, episode_index=index, split=split)
    return rows


def _sampling_spec(training, split, seed):
    if split == 'ind':
        sampling = deepcopy(training['sampling'])
        sampling['seed'] = seed
    else:
        sampling = dict(method='independent_uniform_central_squares', seed=seed,
                        points_per_region=COUNT,
                        pairing='independent source and target streams, paired by index',
                        random_streams='numpy SeedSequence(seed).spawn(2)',
                        coordinate_rounding=False)
    sampling.update(policy_outcomes_used=False, physics_outcomes_used=False,
                    failed_layouts_automatically_replaced=False)
    return sampling


def _build_manifest(base_config, base_config_path, training_manifest_path, training, split, seed):
    if split not in DEFAULT_SEEDS:
        raise ValueError("split must be 'ind' or 'ood'")
    if type(seed) is not int or seed < 0:
        raise ValueError('Sampling seed must be a nonnegative integer')
    if seed == training['sampling']['seed']:
        raise ValueError('Evaluation sampling seed must differ from the training seed')
    return dict(
        format_version=2, dataset_version='insert_plug_v2', split=split,
        layout_set_id=f'insert_plug_v2_{split}_evaluation_seed{seed}',
        description=('Fresh training-ring source/target pairs; one trial per pair.' if split == 'ind'
                     else 'Independent source/target pairs in both central 4 cm held-out squares; one trial per pair.'),
        names_are_local_to_manifest=True,
        base_config=_display_path(base_config_path), base_config_sha256=_sha256(base_config_path),
        base_source_xy=list(base_config['board']['source_xy']),
        base_target_xy=list(base_config['board']['target_xy']),
        training_manifest=dict(path=_display_path(training_manifest_path), sha256=_sha256(training_manifest_path)),
        training_sampling_seed=training['sampling']['seed'],
        regions=deepcopy(training['regions']),
        region_membership=('outer square minus closed central square' if split == 'ind' else 'closed central square'),
        sampling=_sampling_spec(training, split, seed), selection_seed=seed,
        selection_uses_policy_outcomes=False, selection_uses_physics_outcomes=False,
        failed_layouts_automatically_replaced=False,
        geometry_frozen=True, selection_complete=True, physics_validated=False, occlusion_checked=False,
        selection_complete_note='All presampled geometry is frozen; no expert or policy feasibility filter.',
        episodes_per_position=1, expected_episodes=COUNT, first_episode_id=0,
        episode_seeds=list(range(COUNT)), seed_semantics='One evaluation episode seed per layout, independent of geometry sampling seed.',
        control_hz=float(base_config['task']['control_hz']),
        recording_hz=float(base_config['recording']['sample_hz']), depth_enabled=True,
        positions=_sample_positions(base_config, training['regions'], split, seed),
    )


def _validate_points(manifest, training):
    rows = manifest['positions']
    for side in ('source', 'target'):
        points = _points(rows, side)
        region = manifest['regions'][side]
        inside = (ring_contains(points, region['center_xy']) if manifest['split'] == 'ind'
                  else square_contains(points, region['ood_bounds_m']))
        if not inside.all():
            raise ValueError(f'Evaluation {side} points fall outside the {manifest["split"]} region')
        old_points = {tuple(point) for point in _points(training['positions'], side)}
        if any(tuple(point) in old_points for point in points):
            # Do not silently replace a colliding draw: that would change the
            # frozen sample. Reject the manifest as a whole instead.
            raise ValueError(f'Evaluation {side} coordinates repeat training coordinates')


def make_manifest(base_config, base_config_path, training_manifest_path, *, split, seed=None):
    """Generate one deterministic, validated split in memory without writing."""
    if split not in DEFAULT_SEEDS:
        raise ValueError("split must be 'ind' or 'ood'")
    seed = DEFAULT_SEEDS[split] if seed is None else seed
    training = _training_sources(base_config, base_config_path, training_manifest_path)
    manifest = _build_manifest(base_config, base_config_path, training_manifest_path, training, split, seed)
    _validate_points(manifest, training)
    return manifest


def validate_manifest(manifest, base_config, base_config_path, training_manifest_path):
    """Check source hashes, regions, novelty, and exact seeded regeneration.

    Regeneration verifies all positions/labels and sampling metadata. It also
    prevents a filtered or manually substituted layout set from being accepted
    under an unchanged seed. The input document is never modified.
    """
    training = _training_sources(base_config, base_config_path, training_manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError('Evaluation manifest must be a mapping')
    expected = _build_manifest(base_config, base_config_path, training_manifest_path, training,
                               manifest.get('split'), manifest.get('selection_seed'))
    if manifest != expected:
        raise ValueError('Evaluation manifest differs from its sources or exact seeded sample')
    _validate_points(manifest, training)
    return dict(valid=True, split=manifest['split'], positions=COUNT, episodes=COUNT,
                novel_source_points=COUNT, novel_target_points=COUNT,
                exact_seeded_regeneration=True, training_manifest_sha256=_sha256(training_manifest_path))


def write_manifest(path, manifest):
    """Exclusively write a manifest returned by make_manifest; never overwrite."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug.yaml')
    parser.add_argument('--training-manifest', type=Path,
                        default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--split', choices=tuple(DEFAULT_SEEDS), required=True)
    parser.add_argument('--seed', type=int, help='Default: a fixed fresh seed for the selected split')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = make_manifest(load_config(args.config), args.config, args.training_manifest,
                             split=args.split, seed=args.seed)
    path = write_manifest(args.output, manifest)
    print(json.dumps(dict(output=str(path), split=args.split, layouts=COUNT,
                          sampling_seed=manifest['selection_seed'], physics_validated=False)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
