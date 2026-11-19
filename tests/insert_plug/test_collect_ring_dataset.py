"""Collection storage and resume checks without running an expert/rendering."""
import copy
import hashlib
from pathlib import Path

import pytest
import yaml

from examples.insert_plug import cli
from examples.insert_plug.data import collect_ring_dataset as collector
from examples.insert_plug.data import dataset
from examples.insert_plug.environment.ring_layouts import make_manifest
from examples.insert_plug.environment.scene import load_config


@pytest.fixture
def collection_inputs(tmp_path, monkeypatch):
    base = load_config()
    base_path = tmp_path / 'base.yaml'
    base_path.write_text(yaml.safe_dump(base))
    manifest = make_manifest(base, base_path)
    manifest.update(selection_complete=True, physics_validated=True)
    for row in manifest['positions']:
        row['physics_check'] = dict(success=True)
    calls = []

    def fake_collect(args):
        config = yaml.safe_load(Path(args.config).read_text())
        calls.append((copy.deepcopy(args), config))
        stem = f'episode_{args.seed:06d}'
        folder = Path(args.output) / stem
        folder.mkdir()
        for suffix in ('h5', 'xml', 'mp4'):
            (folder / f'{stem}.{suffix}').write_bytes(b'unchanged fixture')
        (folder / f'{stem}.yaml').write_text(yaml.safe_dump(config))
        return 0

    monkeypatch.setattr(cli, 'collect', fake_collect)
    monkeypatch.setattr(dataset, 'validate_episode', lambda path: dict(
        valid=True, success=True, steps=191,
        metadata=dict(execution_result=dict(steps=191))))
    output = tmp_path / 'bags'
    output.mkdir()
    return base, base_path, manifest, output, calls


def test_collect_writes_one_flat_episode_with_preserved_settings(collection_inputs):
    base, base_path, manifest, output, calls = collection_inputs
    original = copy.deepcopy(base)
    collector.validate_manifest(manifest, base, base_path)
    result = collector.collect_one(manifest['positions'][0], base, manifest, 'frozen', output, 'egl')
    assert result['success'] and not result['resumed']
    assert [path.name for path in output.iterdir()] == ['episode_000001']
    folder = output / 'episode_000001'
    assert {path.name for path in folder.iterdir()} == {
        f'episode_000001.{suffix}' for suffix in ('h5', 'yaml', 'xml', 'mp4')}
    args, saved = calls[0]
    assert args.seed == 1 and args.randomize_mm == 0 and not args.no_images
    assert saved['task']['control_hz'] == saved['recording']['sample_hz'] == 10
    assert saved['cameras']['depth']['enabled'] is True
    assert saved['collection']['layout_manifest_sha256'] == 'frozen'
    assert saved['board']['source_xy'] == manifest['positions'][0]['source_xy']
    assert base == original


def test_resume_requires_same_frozen_manifest_and_does_not_overwrite(collection_inputs):
    base, _, manifest, output, calls = collection_inputs
    row = manifest['positions'][-1]
    assert collector.collect_one(row, base, manifest, 'frozen', output, 'egl')['success']
    folder = output / 'episode_000100'
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    resumed = collector.collect_one(row, base, manifest, 'frozen', output, 'egl')
    assert resumed['success'] and resumed['resumed'] and len(calls) == 1
    changed = collector.collect_one(row, base, manifest, 'different', output, 'egl')
    assert not changed['success'] and 'configuration differs' in changed['error']
    assert {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()} == before
    assert len(calls) == 1


def test_incomplete_existing_episode_is_preserved(collection_inputs):
    base, _, manifest, output, calls = collection_inputs
    folder = output / 'episode_000001'
    folder.mkdir()
    (folder / 'episode_000001.h5').write_bytes(b'keep partial recording')
    result = collector.collect_one(manifest['positions'][0], base, manifest, 'frozen', output, 'egl')
    assert not result['success'] and 'Incomplete/extraneous' in result['error']
    assert (folder / 'episode_000001.h5').read_bytes() == b'keep partial recording'
    assert not calls


def test_old_dataset_structure_is_rejected_before_collection(collection_inputs, tmp_path):
    base, base_path, manifest, output, calls = collection_inputs
    manifest['base_config'] = str(base_path)
    manifest_path = tmp_path / 'manifest.yaml'
    manifest_path.write_text(yaml.safe_dump(manifest))
    historical = output / 'pos0'
    historical.mkdir()
    protected = historical / 'record.h5'
    protected.write_bytes(b'old data')
    with pytest.raises(ValueError, match='unexpected entries'):
        collector.main(['--manifest', str(manifest_path), '--output', str(output),
                        '--report', str(tmp_path / 'report.json')])
    assert protected.read_bytes() == b'old data' and not calls
