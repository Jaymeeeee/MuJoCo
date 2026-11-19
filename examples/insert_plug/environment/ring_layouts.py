"""Freeze 100 equal-area square-ring layouts before expert validation.

The central square, including its boundary, is held out from training. This
module samples geometry only: it neither runs physics nor replaces failed
layouts. Run the collection preflight before setting selection_complete.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from mujoco_tools.paths import WORKSPACE_ROOT
from .scene import load_config


OUTER_SIDE_M = .11
OOD_SIDE_M = .04
BINS = 5
FACE_NAMES = ('positive_x', 'positive_y', 'negative_x', 'negative_y')


def region_spec(center_xy):
    center = np.asarray(center_xy, dtype=float)
    if center.shape != (2,) or not np.isfinite(center).all():
        raise ValueError('Region centre must contain two finite coordinates')
    bounds = lambda half: {axis: np.round([value - half, value + half], 12).tolist()
                           for axis, value in zip(('x', 'y'), center)}
    return dict(center_xy=center.tolist(), outer_side_m=OUTER_SIDE_M,
                ood_side_m=OOD_SIDE_M, outer_bounds_m=bounds(OUTER_SIDE_M / 2),
                ood_bounds_m=bounds(OOD_SIDE_M / 2),
                ood_boundary_excluded_from_training=True,
                ind_area_m2=OUTER_SIDE_M ** 2 - OOD_SIDE_M ** 2)


def ring_contains(points, center_xy):
    """Return membership in the outer square minus the closed OOD square."""
    values = np.asarray(points, dtype=float)
    if values.shape[-1:] != (2,):
        raise ValueError('Points must have final dimension two')
    spec = region_spec(center_xy)
    outer = np.array(list(spec['outer_bounds_m'].values()))
    inner = np.array(list(spec['ood_bounds_m'].values()))
    inside_outer = np.all((values >= outer[:, 0]) & (values <= outer[:, 1]), axis=-1)
    inside_hole = np.all((values >= inner[:, 0]) & (values <= inner[:, 1]), axis=-1)
    return np.isfinite(values).all(axis=-1) & inside_outer & ~inside_hole


def sample_ring(center_xy, rng):
    """Sample once in each of 100 equal-area strata; return points and labels.

    Each of four face trapezoids has coordinates (r, t*r), rotated by 90
    degrees. Its Jacobian is r, so uniform r**2 and t give uniform XY area.
    Splitting each coordinate into five bins yields 25 equal-area strata per
    face, each with area 0.000105 m². No rounding to a spatial grid is applied.
    """
    center = np.asarray(region_spec(center_xy)['center_xy'])
    radial_edges = np.linspace((OOD_SIDE_M / 2) ** 2, (OUTER_SIDE_M / 2) ** 2, BINS + 1)
    tangent_edges = np.linspace(-1., 1., BINS + 1)
    rotations = (np.eye(2), np.array([[0., -1.], [1., 0.]]),
                 -np.eye(2), np.array([[0., 1.], [-1., 0.]]))
    points, strata = [], []
    for face, rotation in enumerate(rotations):
        for radial in range(BINS):
            for tangent in range(BINS):
                # A draw exactly on a stratum edge has measure zero. Reject
                # numerical endpoint draws rather than putting a point in OOD.
                while True:
                    u, v = rng.random(2)
                    if not (0 < u < 1 and 0 < v < 1):
                        continue
                    r = np.sqrt(radial_edges[radial] + u * np.diff(radial_edges)[radial])
                    t = tangent_edges[tangent] + v * np.diff(tangent_edges)[tangent]
                    point = center + rotation @ np.array([r, t * r])
                    if ring_contains(point, center):
                        break
                points.append(point.tolist())
                strata.append(dict(face=face, face_name=FACE_NAMES[face],
                                   radial_bin=radial, tangent_bin=tangent))
    return points, strata


def sample_layouts(base_config, seed=20261112):
    """Pair independent ring samples with 6/7 examples per source-target face."""
    source_seed, target_seed, pairing_seed = np.random.SeedSequence(seed).spawn(3)
    source, source_strata = sample_ring(base_config['board']['source_xy'], np.random.default_rng(source_seed))
    target, target_strata = sample_ring(base_config['board']['target_xy'], np.random.default_rng(target_seed))
    rng = np.random.default_rng(pairing_seed)
    counts = np.full((4, 4), 6, dtype=int)
    counts[np.arange(4), rng.permutation(4)] += 1
    available = [list(rng.permutation(np.arange(face * 25, (face + 1) * 25))) for face in range(4)]
    pairs = []
    for source_face in range(4):
        source_indices = rng.permutation(np.arange(source_face * 25, (source_face + 1) * 25))
        target_faces = rng.permutation(np.repeat(np.arange(4), counts[source_face]))
        for source_index, target_face in zip(source_indices, target_faces):
            target_index = available[int(target_face)].pop()
            pairs.append((int(source_index), int(target_index)))
    rows = []
    for index in rng.permutation(len(pairs)):
        source_index, target_index = pairs[index]
        episode = len(rows) + 1
        rows.append(dict(name=f'pos{episode - 1}', episode_id=episode,
                         source_xy=source[source_index], target_xy=target[target_index],
                         source_stratum=source_strata[source_index],
                         target_stratum=target_strata[target_index]))
    return rows


def make_manifest(base_config, base_config_path, seed=20261112):
    """Construct an unvalidated geometry manifest without writing any files."""
    path = Path(base_config_path).resolve()
    if yaml.safe_load(path.read_text(encoding='utf8')) != base_config:
        raise ValueError('base_config must exactly match base_config_path')
    display_path = str(path.relative_to(WORKSPACE_ROOT)) if path.is_relative_to(WORKSPACE_ROOT) else str(path)
    return dict(
        format_version=2, dataset_version='insert_plug_v2', split='ind',
        base_config=display_path, base_config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        base_source_xy=list(base_config['board']['source_xy']),
        base_target_xy=list(base_config['board']['target_xy']),
        regions={side: region_spec(base_config['board'][f'{side}_xy']) for side in ('source', 'target')},
        sampling=dict(method='equal_area_square_ring_stratification', seed=int(seed),
                      faces=4, radial_squared_bins=BINS, tangent_bins=BINS,
                      points_per_region=100, points_per_stratum=1,
                      stratum_area_m2=(OUTER_SIDE_M ** 2 - OOD_SIDE_M ** 2) / 100,
                      pairing='randomized one-to-one; each face pair receives 6 or 7 layouts',
                      coordinate_rounding=False, policy_outcomes_used=False,
                      failed_layouts_automatically_replaced=False),
        selection_seed=int(seed), episodes_per_position=1, expected_episodes=100,
        first_episode_id=1, recording_hz=float(base_config['recording']['sample_hz']),
        control_hz=float(base_config['task']['control_hz']), depth_enabled=True,
        required_edge_margin_px=5.0, geometry_frozen=True, selection_complete=False,
        physics_validated=False, occlusion_checked=False,
        positions=sample_layouts(base_config, seed),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug.yaml')
    parser.add_argument('--output', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--seed', type=int, default=20261112)
    args = parser.parse_args(argv)
    manifest = make_manifest(load_config(args.config), args.config, args.seed)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf8') as stream:
        yaml.safe_dump(manifest, stream, sort_keys=False)
    print(json.dumps(dict(output=str(output), layouts=len(manifest['positions']),
                          geometry_frozen=True, physics_validated=False)))


if __name__ == '__main__':
    main()
