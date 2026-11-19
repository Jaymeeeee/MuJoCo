"""Check the declared ring coverage and the train/OOD separation."""
from collections import Counter

import numpy as np
import pytest
import yaml

from examples.insert_plug.environment.ring_layouts import (
    make_manifest, region_spec, ring_contains, sample_layouts,
)


BASE = dict(board=dict(source_xy=[.45, -.07], target_xy=[.45, .07]),
            recording=dict(sample_hz=10), task=dict(control_hz=10))


def test_ood_square_including_boundary_is_excluded():
    centre = [.45, -.07]
    assert not ring_contains([[.45, -.07], [.43, -.09], [.47, -.05], [.45, -.05]], centre).any()
    assert ring_contains([[.45, -.049], [.429, -.07], [.395, -.125], [.505, -.015]], centre).all()
    assert not ring_contains([[.394, -.07], [.45, -.126], [np.nan, -.07]], centre).any()
    assert region_spec(centre)['outer_bounds_m'] == dict(x=[.395, .505], y=[-.125, -.015])


@pytest.mark.parametrize('seed', [0, 17, 20261112])
def test_unique_points_cover_every_equal_area_stratum(seed):
    rows = sample_layouts(BASE, seed)
    assert [row['episode_id'] for row in rows] == list(range(1, 101))
    for side in ('source', 'target'):
        centre = np.asarray(BASE['board'][f'{side}_xy'])
        points = np.array([row[f'{side}_xy'] for row in rows])
        assert len(np.unique(points, axis=0)) == 100
        assert ring_contains(points, centre).all()
        # Reconstruct strata from coordinates, independently of saved labels.
        observed = set()
        for point, row in zip(points, rows):
            x, y = point - centre
            face = (0 if x > 0 else 2) if abs(x) > abs(y) else (1 if y > 0 else 3)
            r = max(abs(x), abs(y))
            t = (y / r, -x / r, -y / r, x / r)[face]
            radial = int((r * r - .02 ** 2) / (.055 ** 2 - .02 ** 2) * 5)
            tangent = int((t + 1) / 2 * 5)
            observed.add((face, radial, tangent))
            label = row[f'{side}_stratum']
            assert (label['face'], label['radial_bin'], label['tangent_bin']) == (face, radial, tangent)
        assert observed == {(face, radial, tangent) for face in range(4)
                            for radial in range(5) for tangent in range(5)}


def test_reproducible_balanced_pairing_is_not_fixed_translation():
    rows = sample_layouts(BASE)
    assert rows == sample_layouts(BASE)
    assert rows != sample_layouts(BASE, 42)
    counts = Counter((r['source_stratum']['face'], r['target_stratum']['face']) for r in rows)
    assert len(counts) == 16 and set(counts.values()) == {6, 7}
    for face in range(4):
        assert sum(n for (s, t), n in counts.items() if s == face) == 25
        assert sum(n for (s, t), n in counts.items() if t == face) == 25
    delta = np.array([np.array(r['target_xy']) - r['source_xy'] for r in rows])
    assert np.ptp(delta, axis=0).min() > .07


def test_manifest_freezes_geometry_without_claiming_expert_validation(tmp_path):
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(BASE))
    document = make_manifest(BASE, path)
    assert document['geometry_frozen'] and not document['selection_complete']
    assert not document['physics_validated']
    assert not document['sampling']['policy_outcomes_used']
    assert document['regions']['target']['ood_bounds_m'] == dict(x=[.43, .47], y=[.05, .09])
    assert len(document['positions']) == document['expected_episodes'] == 100
    assert len(list(tmp_path.iterdir())) == 1
