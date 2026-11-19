"""Reject central-region leakage, repeated layouts and mislabeled flat batches."""
import copy

import numpy as np
import pytest

from examples.insert_plug.diagnostics.audit_ring_dataset import (
    require_ring_position, validate_inventory, validate_positions, validate_stratification,
)


def manifest():
    return {'positions': [dict(name=f'pos{i}', episode_id=i + 1,
                               source_xy=[.48, -.11 + i * .0008],
                               target_xy=[.48, .03 + i * .0008]) for i in range(100)]}


@pytest.mark.parametrize('xy', [[.45, -.07], [.47, -.07], [.43, -.09], [.46, -.08]])
def test_central_interior_and_boundary_are_not_training(xy):
    with pytest.raises(ValueError, match='held-out'):
        require_ring_position(xy, 'source')


@pytest.mark.parametrize('xy', [[.395, -.125], [.505, -.015], [.45, -.10], [.48, -.07]])
def test_outer_ring_and_outer_boundary_are_allowed(xy):
    assert require_ring_position(xy, 'source') == tuple(np.round(xy, 12))


@pytest.mark.parametrize('xy', [[.505001, -.07], [.45, -.125001], [float('nan'), -.07]])
def test_out_of_range_or_nonfinite_coordinates_are_rejected(xy):
    with pytest.raises(ValueError):
        require_ring_position(xy, 'source')


def test_manifest_requires_unique_source_and_target_coordinates():
    valid = manifest()
    assert len(validate_positions(valid)) == 100
    for key in ('source_xy', 'target_xy'):
        duplicate = copy.deepcopy(valid)
        duplicate['positions'][99][key] = duplicate['positions'][0][key]
        with pytest.raises(ValueError, match='Duplicate'):
            validate_positions(duplicate)


def test_episode_numbering_starts_at_one_and_preserves_all_hundred():
    bad = manifest()
    bad['positions'][0]['episode_id'] = 0
    with pytest.raises(ValueError, match='1..100'):
        validate_positions(bad)


def test_flat_inventory_rejects_missing_episode_extras_and_symlinks(tmp_path):
    for index in range(1, 101):
        (tmp_path / f'episode_{index:06d}').mkdir()
    validate_inventory(tmp_path)
    (tmp_path / 'episode_000100').rmdir()
    with pytest.raises(ValueError, match='exactly'):
        validate_inventory(tmp_path)
    (tmp_path / 'episode_000100').symlink_to(tmp_path / 'episode_000099', target_is_directory=True)
    with pytest.raises(ValueError, match='regular directories'):
        validate_inventory(tmp_path)


def stratified_rows():
    positions = []
    for face in range(4):
        for radial in range(5):
            r = np.sqrt(.02**2 + (radial + .5) / 5 * (.055**2 - .02**2))
            for tangential in range(5):
                t = -1 + (tangential + .5) * 2 / 5
                offsets = [(r, t*r), (-t*r, r), (-r, -t*r), (t*r, -r)]
                positions.append((np.array(offsets[face]),
                                  dict(face=face, radial_bin=radial, tangent_bin=tangential)))
    queues = [list(range(face*25, (face+1)*25)) for face in range(4)]
    result = []
    for face in range(4):
        target_faces = [other for other in range(4) for _ in range(7 if other == face else 6)]
        for local, target_face in enumerate(target_faces):
            source = positions[face * 25 + local]
            target = positions[queues[target_face].pop(0)]
            result.append(dict(source_xy=(source[0] + [.45, -.07]).tolist(),
                               target_xy=(target[0] + [.45, .07]).tolist(),
                               source_stratum=source[1], target_stratum=target[1]))
    return result


def test_independent_geometry_recovers_strata_and_balanced_pairing():
    report = validate_stratification(stratified_rows())
    assert report['source_target_face_counts'] == [[7,6,6,6], [6,7,6,6], [6,6,7,6], [6,6,6,7]]


def test_false_stratum_metadata_and_fixed_translation_pairing_are_rejected():
    rows = stratified_rows()
    altered = copy.deepcopy(rows)
    altered[0]['source_stratum']['radial_bin'] = 4
    with pytest.raises(ValueError, match='label'):
        validate_stratification(altered)
    for row in rows:
        row['target_xy'] = (np.array(row['source_xy']) + [0, .14]).tolist()
        row['target_stratum'] = row['source_stratum'].copy()
    with pytest.raises(ValueError, match='balanced'):
        validate_stratification(rows)
