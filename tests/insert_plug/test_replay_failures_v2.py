"""Passive diagnostics must preserve physics and reject changed evidence."""
from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from examples.insert_plug.policy import replay_failures_v2 as diagnostic


def test_physical_comparison_is_exact_and_includes_final_state_and_action():
    original = {key: None for key in diagnostic.PHYSICAL_RESULT_KEYS}
    original.update(last_action=[1., 2.], final_metrics={'radial_error_m': .0001},
                    wall_seconds=1., policy_selection_wall_seconds={'mean': .1})
    replay = deepcopy(original)
    replay['wall_seconds'] = 99.
    replay['policy_selection_wall_seconds'] = {'mean': .8}
    assert diagnostic.exact_result_comparison(original, replay)['equal']
    replay['final_metrics']['radial_error_m'] += 1e-15
    comparison = diagnostic.exact_result_comparison(original, replay)
    assert not comparison['equal']
    assert comparison['differences'][0]['field'] == 'final_metrics'
    replay = deepcopy(original)
    replay['last_action'][0] = np.nextafter(1., 2.)
    assert not diagnostic.exact_result_comparison(original, replay)['equal']


def test_missing_physical_field_cannot_match():
    value = {key: None for key in diagnostic.PHYSICAL_RESULT_KEYS}
    other = dict(value)
    del other['physics_checks']
    assert not diagnostic.exact_result_comparison(value, other)['equal']


def test_protected_file_drift_rejected(tmp_path):
    path = tmp_path/'reference.json'
    path.write_text('{"original":true}')
    config = dict(protected_files={str(path): diagnostic.fingerprint(path)})
    diagnostic.verify_provenance(config)
    path.write_text('{"original":false}')
    with pytest.raises(ValueError, match='Protected'):
        diagnostic.verify_provenance(config)


@pytest.fixture
def encoded_video(tmp_path):
    import imageio.v2 as imageio
    path = tmp_path/'source.mp4'
    frames = []
    for i in range(3):
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        frame[4+i:14+i, 6:17] = [50+i*20, 180, 220]
        frames.append(frame)
    with imageio.get_writer(str(path), fps=10, codec='libx264', quality=8,
                            macro_block_size=1, ffmpeg_log_level='error') as writer:
        for frame in frames:
            writer.append_data(frame)
    return path, frames


def test_identical_video_bytes_prove_all_frames(encoded_video, tmp_path):
    original, _ = encoded_video
    copied = tmp_path/'copy.mp4'
    shutil.copyfile(original, copied)
    result = diagnostic.compare_videos(original, copied, 3)
    assert result['equal'] and result['frames'] == 3
    assert result['method'] == 'identical_complete_video_bytes'


def test_different_container_same_decoded_frames_is_accepted(encoded_video, tmp_path):
    import imageio_ffmpeg
    original, _ = encoded_video
    changed = tmp_path/'different_container.mp4'
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-v', 'error', '-i', str(original),
                    '-c', 'copy', '-metadata', 'title=distinct container metadata', str(changed)], check=True)
    assert original.read_bytes() != changed.read_bytes()
    result = diagnostic.compare_videos(original, changed, 3)
    assert result['equal'] and result['frames'] == 3
    assert result['method'] == 'decoded_rgb_bytes'


def test_changed_video_frame_is_rejected(encoded_video, tmp_path):
    import imageio.v2 as imageio
    original, frames = encoded_video
    changed = tmp_path/'changed_frame.mp4'
    frames[1] = frames[1].copy()
    frames[1][15:25, 15:25] = [250, 100, 10]
    with imageio.get_writer(str(changed), fps=10, codec='libx264', quality=8,
                            macro_block_size=1, ffmpeg_log_level='error') as writer:
        for frame in frames:
            writer.append_data(frame)
    result = diagnostic.compare_videos(original, changed, 3)
    assert not result['equal'] and result['mismatched_frames'] > 0
    assert result['max_absolute_pixel_difference'] > 0


def test_invalid_worker_count_rejected():
    with pytest.raises(SystemExit):
        diagnostic.parse_args(['--workers', '0'])


def test_snapshot_observer_and_independent_fk_preserve_live_physics_exactly():
    pytest.importorskip('mujoco')
    from examples.insert_plug.environment.simulation import Simulation
    from examples.insert_plug.policy.rollout import PhysicalSuccessMonitor

    root = diagnostic.WORKSPACE_ROOT
    episode = root/diagnostic.DEFAULT_REFERENCES['act_ind']/'episodes/episode_000000'
    if not (episode/'scene.xml').is_file():
        pytest.skip('This integration check uses the preserved local v2 scene')
    config = json.loads((episode/'result.json').read_text())['config']
    ordinary = Simulation(episode/'scene.xml', config, render=False)
    observed = diagnostic.simulation_class()(episode/'scene.xml', config, render=False, diagnostic_duration=.4)
    try:
        monitors = [PhysicalSuccessMonitor(sim) for sim in (ordinary, observed)]
        for sim, monitor in zip((ordinary, observed), monitors):
            sim.safety_callback = monitor.sample
        for index in range(4):
            action = np.r_[config['robot']['home_qpos'], .075].astype(np.float32)
            action[0] += index*.002
            for sim in (ordinary, observed):
                sim.step(action)
            for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'ctrl', 'qfrc_constraint'):
                assert np.array_equal(getattr(ordinary.data, field), getattr(observed.data, field)), field
            assert ordinary.metrics() == observed.metrics()
        arrays = observed.trace.arrays()
        assert len(arrays['physics_time_s']) == 401
        assert arrays['physics_time_s'][0] == 0
        assert np.array_equal(arrays['physics_action_index'][1:], np.repeat(np.arange(4), 100))
        assert arrays['action'].shape == (4, 8)
        assert arrays['target_tcp_xyz'].shape == (4, 3)
        assert arrays['target_tcp_rotation'].shape == (4, 3, 3)
        assert arrays['dual_contact'].shape == (401, 2)
        assert arrays['peg_collision_vertices'].shape[1] == 3
        count = observed.trace.count
        monitors[1].metrics()
        monitors[1].metrics()
        assert observed.trace.count == count
        assert np.array_equal(ordinary.data.qpos, observed.data.qpos)
    finally:
        ordinary.close()
        observed.close()
