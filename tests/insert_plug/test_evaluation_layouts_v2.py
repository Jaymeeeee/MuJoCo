"""Independent checks of fresh frozen v2 ring and central-square evaluations."""
from collections import Counter
import hashlib

import numpy as np
import pytest
import yaml

from examples.insert_plug.environment import evaluation_layouts_v2 as layouts
from examples.insert_plug.environment.ring_layouts import make_manifest as make_training_manifest


BASE = dict(board=dict(source_xy=[.45, -.07], target_xy=[.45, .07]),
            recording=dict(sample_hz=10), task=dict(control_hz=10))


@pytest.fixture
def sources(tmp_path):
    config_path = tmp_path / 'base.yaml'
    config_path.write_text(yaml.safe_dump(BASE))
    training = make_training_manifest(BASE, config_path)
    training['selection_complete'] = True
    training_path = tmp_path / 'training.yaml'
    training_path.write_text(yaml.safe_dump(training, sort_keys=False))
    return config_path, training_path, training


@pytest.mark.parametrize('split', ['ind', 'ood'])
def test_deterministic_new_points_one_trial_each_and_yaml_round_trip(sources, tmp_path, split):
    config_path, training_path, training = sources
    original = training_path.read_bytes()
    manifest = layouts.make_manifest(BASE, config_path, training_path, split=split)
    assert manifest == layouts.make_manifest(BASE, config_path, training_path, split=split)
    assert [row['name'] for row in manifest['positions']] == [f'pos{i}' for i in range(100)]
    for field in ('seed', 'episode_id', 'episode_index'):
        assert [row[field] for row in manifest['positions']] == list(range(100))
    assert manifest['expected_episodes'] == 100 and manifest['episodes_per_position'] == 1
    assert manifest['selection_seed'] != training['selection_seed']
    assert manifest['geometry_frozen'] and manifest['selection_complete']
    assert not manifest['physics_validated'] and not manifest['selection_uses_policy_outcomes']
    assert not manifest['selection_uses_physics_outcomes']
    assert not manifest['failed_layouts_automatically_replaced']
    assert manifest['training_manifest']['sha256'] == hashlib.sha256(original).hexdigest()
    for side in ('source', 'target'):
        points = {tuple(row[f'{side}_xy']) for row in manifest['positions']}
        old = {tuple(row[f'{side}_xy']) for row in training['positions']}
        assert len(points) == 100 and points.isdisjoint(old)
    destination = layouts.write_manifest(tmp_path / f'{split}.yaml', manifest)
    reloaded = yaml.safe_load(destination.read_text())
    assert reloaded == manifest
    assert layouts.validate_manifest(reloaded, BASE, config_path, training_path)['exact_seeded_regeneration']
    with pytest.raises(FileExistsError):
        layouts.write_manifest(destination, manifest)
    assert training_path.read_bytes() == original


def test_ind_coordinates_cover_training_strata_and_balanced_face_pairs(sources):
    config_path, training_path, _ = sources
    manifest = layouts.make_manifest(BASE, config_path, training_path, split='ind')
    rows = manifest['positions']
    faces = {}
    for side in ('source', 'target'):
        seen = set()
        faces[side] = []
        for row in rows:
            x, y = np.asarray(row[f'{side}_xy']) - BASE['board'][f'{side}_xy']
            radius = max(abs(x), abs(y))
            assert .02 < radius <= .055
            face = (0 if x > 0 else 2) if abs(x) > abs(y) else (1 if y > 0 else 3)
            tangent_coordinate = (y / radius, -x / radius, -y / radius, x / radius)[face]
            radial_bin = int((radius ** 2 - .02 ** 2) / (.055 ** 2 - .02 ** 2) * 5)
            tangent_bin = int((tangent_coordinate + 1) / 2 * 5)
            seen.add((face, radial_bin, tangent_bin))
            faces[side].append(face)
            label = row[f'{side}_stratum']
            assert (label['face'], label['radial_bin'], label['tangent_bin']) == (face, radial_bin, tangent_bin)
        assert seen == {(face, radial, tangent) for face in range(4) for radial in range(5) for tangent in range(5)}
    counts = Counter(zip(faces['source'], faces['target']))
    assert len(counts) == 16 and set(counts.values()) == {6, 7}


def test_ood_draws_use_independent_uniform_source_target_streams_and_closed_bounds(sources):
    config_path, training_path, _ = sources
    seed = 7351
    manifest = layouts.make_manifest(BASE, config_path, training_path, split='ood', seed=seed)
    streams = np.random.SeedSequence(seed).spawn(2)
    offsets = []
    for side, stream in zip(('source', 'target'), streams):
        bounds = manifest['regions'][side]['ood_bounds_m']
        lower = np.array([bounds['x'][0], bounds['y'][0]])
        upper = np.array([bounds['x'][1], bounds['y'][1]])
        points = np.asarray([row[f'{side}_xy'] for row in manifest['positions']])
        np.testing.assert_array_equal(points, np.random.default_rng(stream).uniform(lower, upper, (100, 2)))
        assert ((points >= lower) & (points <= upper)).all()
        offsets.append(points - BASE['board'][f'{side}_xy'])
        assert layouts.square_contains([lower, upper], bounds).all()
        assert not layouts.square_contains([lower - .000001, [np.nan, 0]], bounds).any()
    assert not np.array_equal(*offsets)


@pytest.mark.parametrize('split', ['ind', 'ood'])
def test_fresh_seed_changes_geometry_and_training_seed_is_rejected(sources, split):
    config_path, training_path, training = sources
    a = layouts.make_manifest(BASE, config_path, training_path, split=split, seed=1001)
    b = layouts.make_manifest(BASE, config_path, training_path, split=split, seed=1002)
    assert a['positions'] != b['positions']
    with pytest.raises(ValueError, match='differ from the training seed'):
        layouts.make_manifest(BASE, config_path, training_path, split=split, seed=training['selection_seed'])


@pytest.mark.parametrize('change', ['position', 'seed', 'regions', 'training_hash', 'outcomes', 'duplicates'])
def test_validation_rejects_edited_or_substituted_frozen_samples(sources, change):
    config_path, training_path, _ = sources
    document = layouts.make_manifest(BASE, config_path, training_path, split='ind')
    if change == 'position':
        document['positions'][0]['source_xy'][0] += .00001
    elif change == 'seed':
        document['positions'][0]['seed'] = 200
    elif change == 'regions':
        document['regions']['target']['ood_side_m'] = .05
    elif change == 'training_hash':
        document['training_manifest']['sha256'] = '0' * 64
    elif change == 'outcomes':
        document['sampling']['failed_layouts_automatically_replaced'] = True
    else:
        document['positions'][1]['source_xy'] = list(document['positions'][0]['source_xy'])
    with pytest.raises(ValueError, match='differs from its sources or exact seeded sample'):
        layouts.validate_manifest(document, BASE, config_path, training_path)


def test_source_changes_are_rejected_without_silently_redefining_regions(sources):
    config_path, training_path, training = sources
    config_path.write_text(config_path.read_text() + '\n')
    with pytest.raises(ValueError, match='Base configuration hash'):
        layouts.make_manifest(BASE, config_path, training_path, split='ind')
    training['base_config_sha256'] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    training['regions']['source']['outer_bounds_m']['x'][0] -= .01
    training_path.write_text(yaml.safe_dump(training))
    with pytest.raises(ValueError, match='region bounds differ'):
        layouts.make_manifest(BASE, config_path, training_path, split='ood')


def test_cli_writes_only_requested_new_file(sources, tmp_path, capsys):
    config_path, training_path, _ = sources
    destination = tmp_path / 'output' / 'ood.yaml'
    arguments = ['--config', str(config_path), '--training-manifest', str(training_path),
                 '--split', 'ood', '--output', str(destination)]
    assert layouts.main(arguments) == 0
    assert '"layouts": 100' in capsys.readouterr().out
    preserved = destination.read_bytes()
    with pytest.raises(FileExistsError):
        layouts.main(arguments)
    assert destination.read_bytes() == preserved
