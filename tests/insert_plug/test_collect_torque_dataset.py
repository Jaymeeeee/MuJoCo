"""Multirate integrity, failure preservation, and torque collection provenance."""
import copy
import hashlib
import json

import h5py
import numpy as np
import pytest
import yaml

from examples.insert_plug.data import collect_torque_dataset as collector
from examples.insert_plug.data.torque_recording import (
    REQUIRED_TORQUE_SHAPES, TorqueEpisodeWriter, validate_torque_episode,
)
from examples.insert_plug.environment.scene import load_config


def image_record(index):
    return dict(
        timestamp=index/10, phase='move',
        joint_position=np.full(7, index*.01), joint_velocity=np.zeros(7),
        gripper_width=np.array([.03]), action=np.r_[np.full(7, index*.02), .03],
        qpos=np.full(10, index*.01), qvel=np.zeros(9), ctrl=np.full(8, index*.02),
        peg_pose=np.r_[np.zeros(3), 1., 0., 0., 0.],
        source_pose=np.r_[np.zeros(3), 1., 0., 0., 0.],
        target_pose=np.r_[np.zeros(3), 1., 0., 0., 0.],
        images={key: np.zeros((4, 4, 3), np.uint8) for key in ('head', 'wrist')},
        depth={key: np.ones((4, 4), np.float32) for key in ('head', 'wrist')},
    )


def torque_record(tick):
    low = image_record(tick//100)
    row = {key: np.zeros(shape) for key, shape in REQUIRED_TORQUE_SHAPES.items()}
    row.update(timestamp=tick/1000, phase=low['phase'],
               action_index=np.int64(tick//100), image_index=np.int64(tick//100),
               action=low['action'], qpos=low['qpos'], qvel=low['qvel'], ctrl=low['ctrl'],
               joint_position=low['joint_position'], joint_velocity=low['joint_velocity'],
               tcp_pose=np.r_[np.zeros(3), 1., 0., 0., 0.])
    return row


def metadata():
    return dict(sample_hz=10., control_hz=10., physics_timestep=.001,
                sample_origin_s=0., sampling_timing='nearest_physics_tick', depth_enabled=True,
                torque_recording=dict(sample_hz=1000.))


def write_fixture(path, frames=2, extra_metadata=None):
    with TorqueEpisodeWriter(path, {**metadata(), **(extra_metadata or {})}, buffer_size=37) as writer:
        for tick in range(frames*100):
            writer.append_torque(torque_record(tick))
            if tick % 100 == 0:
                writer.append(image_record(tick//100))
        writer.finalize(True)


def test_multirate_stream_preserves_every_tick_and_image_alignment(tmp_path):
    path = tmp_path/'episode.h5'
    write_fixture(path)
    result = validate_torque_episode(path)
    assert result['valid'], result['errors']
    assert result['steps'] == 2 and result['torque_samples'] == 200
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle['alignment/image_to_torque_index'], [0, 100])
        assert handle['torque/timestamp'][-1] == .199
        assert handle['torque/joint_torque_measured_sim'].shape == (200, 7)
        assert handle['observations/images/head'].shape == (2, 4, 4, 3)
        assert handle['torque'].attrs['samples'] == 200


@pytest.mark.parametrize('corruption', ['gap', 'action', 'state', 'alignment', 'nonfinite'])
def test_validator_rejects_timing_and_cross_stream_corruption(tmp_path, corruption):
    path = tmp_path/'episode.h5'
    write_fixture(path)
    with h5py.File(path, 'r+') as handle:
        if corruption == 'gap':
            handle['torque/timestamp'][5] += .001
        elif corruption == 'action':
            handle['torque/action'][55, 0] = 5
        elif corruption == 'state':
            handle['torque/qpos'][100, 0] = 5
        elif corruption == 'alignment':
            handle['alignment/image_to_torque_index'][1] = 99
        else:
            handle['torque/joint_external_torque_estimated'][4, 0] = np.nan
    assert not validate_torque_episode(path)['valid']


def test_writer_refuses_missing_ticks_and_preserves_partial(tmp_path):
    path = tmp_path/'episode.h5'
    with TorqueEpisodeWriter(path, metadata()) as writer:
        writer.append_torque(torque_record(0))
        writer.append(image_record(0))
        with pytest.raises(ValueError, match='Missed or repeated'):
            writer.append_torque(torque_record(2))
        with pytest.raises(ValueError, match='100 torque'):
            writer.finalize(True)
    assert not path.exists()
    partial = path.with_name('episode.h5.partial')
    with h5py.File(partial) as handle:
        assert not handle.attrs['complete']
        assert handle['torque/timestamp'].shape == (1,)
    with pytest.raises((FileExistsError, OSError)):
        TorqueEpisodeWriter(path, metadata())


def test_configuration_retains_physics_control_and_records_provenance():
    base = load_config()
    original = copy.deepcopy(base)
    manifest = yaml.safe_load((collector.WORKSPACE_ROOT/'config/insert_plug_positions_v2.yaml').read_text())
    row = manifest['positions'][0]
    cfg = collector.episode_config(base, manifest, row, 'manifest_hash', {'module': 'source_hash'})
    assert base == original
    assert cfg['physics'] == base['physics']
    assert cfg['task']['control_hz'] == cfg['recording']['sample_hz'] == 10
    assert cfg['recording']['torque_sample_hz'] == 1000
    assert cfg['cameras']['depth']['enabled']
    assert cfg['collection']['freshly_executed']
    assert not cfg['collection']['force_feedback_enabled']
    assert cfg['collection']['source_sha256'] == {'module': 'source_hash'}
    assert cfg['collection']['layout_manifest_sha256'] == 'manifest_hash'
    assert cfg['board']['source_xy'] == row['source_xy']


def test_resume_requires_complete_multirate_data_and_exact_configuration(tmp_path):
    folder = tmp_path/'episode_000001'
    folder.mkdir()
    stem = folder.name
    config = {'recording': {'torque_sample_hz': 1000}}
    (folder/f'{stem}.yaml').write_text(yaml.safe_dump(config))
    model = folder/f'{stem}.xml'
    model.write_text('<mujoco/>')
    (folder/f'{stem}.mp4').write_bytes(b'video fixture')
    path = folder/f'{stem}.h5'
    write_fixture(path, extra_metadata=dict(config=config, model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()))
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    assert collector.existing_episode(folder, stem, config)['valid']
    with pytest.raises(RuntimeError, match='configuration differs'):
        collector.existing_episode(folder, stem, {'recording': {'torque_sample_hz': 500}})
    assert before == {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
    with h5py.File(path, 'r+') as handle:
        handle['torque/timestamp'][5] += .1
    with pytest.raises(RuntimeError, match='validation failed'):
        collector.existing_episode(folder, stem, config)


def test_existing_partial_episode_and_provenance_are_never_overwritten(tmp_path):
    folder = tmp_path/'episode_000001'
    folder.mkdir()
    partial = folder/'episode_000001.h5.partial'
    partial.write_bytes(b'preserve')
    with pytest.raises(RuntimeError, match='Incomplete/extraneous'):
        collector.existing_episode(folder, folder.name, {})
    assert partial.read_bytes() == b'preserve'
    frozen = tmp_path/'manifest.yaml'
    collector._write_frozen(frozen, b'first')
    collector._write_frozen(frozen, b'first')
    with pytest.raises(FileExistsError, match='differs'):
        collector._write_frozen(frozen, b'second')
    assert frozen.read_bytes() == b'first'
