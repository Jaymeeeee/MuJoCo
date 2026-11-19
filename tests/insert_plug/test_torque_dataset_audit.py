"""Detect corrupted clocks, alignment, force frames and noncausal estimates."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from examples.insert_plug.diagnostics.audit_torque_dataset import (
    SIGNAL_SHAPES, HighRateStateComparison, audit_alignment, audit_dynamics_components, audit_filter,
    audit_provenance, rotations_from_wxyz,
)


def streams():
    count, n = 3, 300
    high = {key: np.zeros((n, *shape), dtype=np.float64)
            for key, shape in SIGNAL_SHAPES.items() if key != 'phase'}
    high['timestamp'] = np.arange(n) / 1000
    high['action_index'] = np.arange(n) // 100
    high['image_index'] = high['action_index'].copy()
    high['phase'] = np.repeat(['approach', 'grasp', 'extract'], 100)
    high['qpos'] = np.arange(n * 9).reshape(n, 9) * 1e-5
    high['qvel'] = np.arange(n * 8).reshape(n, 8) * 1e-6
    high['ctrl'] = np.repeat(np.arange(count * 8).reshape(count, 8) * 1e-3, 100, axis=0)
    high['action'] = high['ctrl'].copy()
    high['tcp_pose'][:, 3] = np.sqrt(.5)
    high['tcp_pose'][:, 6] = np.sqrt(.5)
    # A +90-degree world rotation maps world +X into local -Y.
    high['ee_external_wrench_world_estimated'][:, [0, 3]] = [2, .3]
    high['ee_external_wrench_tcp_estimated'][:, [1, 4]] = [-2, -.3]
    high['finger_normal_force'][:] = [4, 5]
    high['finger_contact_force_world'][:, :, 0] = [4, -5]
    high['finger_contact_force_tcp'][:, :, 1] = [-4, 5]
    low = {key: high[key][::100].copy() for key in
           ('timestamp', 'action', 'phase', 'qpos', 'qvel', 'ctrl', 'joint_position', 'joint_velocity')}
    return high, low, np.arange(count) * 100


def test_all_physics_ticks_align_with_image_states_and_active_targets():
    high, low, index = streams()
    result = audit_alignment(high, low, index)
    assert result['physics_samples'] == 300
    assert result['image_frames'] == 3
    assert result['last_timestamp_s'] == .299


@pytest.mark.parametrize('corruption,match', [
    ('missing_tick', 'dimensions'), ('future_image', 'image_index'),
    ('control_between_images', 'Held control target'), ('stale_state', 'qpos alignment'),
    ('negative_normal', 'negative'), ('wrong_frame', 'wrench rotation'),
    ('wrong_finger_frame', 'finger force rotation'), ('nonfinite', 'Non-finite'),
    ('phase_shift', 'phases'), ('misaligned_index', 'indices'),
])
def test_corruptions_are_rejected(corruption, match):
    high, low, index = streams()
    if corruption == 'missing_tick':
        high['timestamp'] = high['timestamp'][:-1]
    elif corruption == 'future_image':
        high['image_index'][99] = 1
    elif corruption == 'control_between_images':
        high['action'][50, 0] += .001
    elif corruption == 'stale_state':
        high['qpos'][100, 0] += 1e-6
    elif corruption == 'negative_normal':
        high['finger_normal_force'][2, 0] = -.1
    elif corruption == 'wrong_frame':
        high['ee_external_wrench_tcp_estimated'][20, 0] = 2
    elif corruption == 'wrong_finger_frame':
        high['finger_contact_force_tcp'][20, 0, 1] *= -1
    elif corruption == 'nonfinite':
        high['joint_actuator_torque'][30, 2] = np.nan
    elif corruption == 'phase_shift':
        high['phase'][99] = 'grasp'
    elif corruption == 'misaligned_index':
        index[1] = 101
    with pytest.raises(ValueError, match=match):
        audit_alignment(high, low, index)


def test_force_filter_has_no_response_before_force_step():
    raw = np.zeros((20, 7))
    raw[10:, 0] = 4
    alpha = 1 - np.exp(-2*np.pi*30*.001)
    filtered = np.zeros_like(raw)
    filtered[10:, 0] = 4 * (1 - (1-alpha)**np.arange(1, 11))
    assert audit_filter(raw, filtered, 30)['maximum_error_nm'] < 1e-12
    filtered[9, 0] = filtered[10, 0]
    with pytest.raises(ValueError, match='Causal'):
        audit_filter(raw, filtered, 30)


def test_filter_initialization_preserves_first_measured_value():
    raw = np.full((10, 7), 3.5)
    assert audit_filter(raw, raw, 30)['maximum_error_nm'] == 0
    filtered = raw.copy()
    filtered[0] = 0
    with pytest.raises(ValueError, match='Causal'):
        audit_filter(raw, filtered, 30)


def test_quaternion_rotation_rejects_nonunit_orientation():
    with pytest.raises(ValueError, match='unit'):
        rotations_from_wxyz([[2, 0, 0, 0]])


def dynamics_fixture():
    high, _, _ = streams()
    n = len(high['timestamp'])
    high['qacc'] = np.full((n, 8), .5)
    for key, source in [('joint_position', 'qpos'), ('joint_velocity', 'qvel'),
                        ('joint_acceleration', 'qacc')]:
        high[key] = high[source][:, :7]
    high['joint_actuator_torque'] = np.ones((n, 7))
    high['joint_actuator_torque'][:, 6] = -.75
    high['joint_passive_torque'] = np.full((n, 7), 2.)
    high['joint_internal_friction_torque'] = np.full((n, 7), .25)
    high['joint_rotor_inertia_torque'] = np.full((n, 7), .5)
    high['joint_torque_measured_sim'] = np.full((n, 7), 2.75)
    high['joint_torque_measured_sim'][:, 6] = 1.
    high['joint_model_torque'] = np.ones((n, 7))
    raw = np.full((n, 7), 1.75)
    raw[:, 6] = 0
    high['joint_external_torque_raw'] = raw.copy()
    high['joint_external_torque_estimated'] = raw.copy()
    high['joint_contact_torque_gt'] = raw.copy()
    for name in ['joint_self_contact_torque_gt', 'joint_noncontact_constraint_torque', 'joint_applied_external_torque']:
        high[name] = np.zeros((n, 7))
    high['external_filter_alpha'] = np.full(n, 1-np.exp(-2*np.pi*30*.001))
    high['external_filter_alpha'][0] = 1.
    diagonal = np.array([.3, .25, .2, 1., .8, .9])
    jac = np.c_[np.diag(diagonal), np.zeros(6)]
    high['tcp_jacobian_world'] = np.repeat(jac[None], n, axis=0)
    scales = np.array([1, 1, 1, .1, .1, .1])
    # Diagonal system has six independent scalar ridge regressions.
    wrench = raw[:, :6] * scales**2 * diagonal / ((diagonal*scales)**2 + 1e-8)
    high['ee_external_wrench_world_estimated'] = wrench
    high['wrench_jacobian_singular_values'] = np.repeat(np.sort(diagonal*scales)[::-1][None], n, axis=0)
    high['wrench_joint_residual'] = raw - wrench @ jac
    high['wrench_estimate_well_conditioned'] = np.ones(n, dtype=bool)
    high['joint_actuator_saturated'] = np.zeros((n, 7), dtype=bool)
    high['finger_contact_count'] = np.ones((n, 2), dtype=np.int32)
    high['arm_external_contact_count'] = np.zeros(n, dtype=np.int32)
    high['tcp_rotation'] = rotations_from_wxyz(high['tcp_pose'][:, 3:])
    high['contact_wrench_at_tcp_world_gt'] = np.zeros((n, 6))
    high['contact_wrench_at_tcp_tcp_gt'] = np.zeros((n, 6))
    method = dict(schema_version='torque-signals-v1',
                  external_sign='robot_on_environment; opposite of environment_on_robot',
                  pose_order=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'], finger_order=['left', 'right'],
                  virtual_measurement={'kind': 'ideal_generalized_link_side_drive_proxy'},
                  external_estimator={'method': 'exact_model_acceleration_dynamics_residual_then_causal_first_order_lowpass',
                                      'filter_cutoff_hz': 30},
                  wrench_estimator={'characteristic_length_m': .1, 'damping_m': 1e-4},
                  arm_qpos_addresses=list(range(7)), arm_dof_addresses=list(range(7)), dof_armature=[1.]*8)
    return high, method


def test_dynamics_reconstruction_matches_analytic_diagonal_wrench_system():
    high, method = dynamics_fixture()
    result = audit_dynamics_components(high, method)
    assert result['ill_conditioned_samples'] == 0
    assert result['errors']['wrench_inverse'] < 1e-12
    assert result['errors']['external_balance_nm'] == 0


@pytest.mark.parametrize('field,match', [
    ('joint_torque_measured_sim', 'drive components'),
    ('joint_model_torque', 'Dynamics residual'),
    ('joint_external_torque_estimated', 'Causal'),
    ('joint_contact_torque_gt', 'force balance'),
    ('ee_external_wrench_world_estimated', 'DLS'),
    ('wrench_jacobian_singular_values', 'singular values'),
    ('joint_rotor_inertia_torque', 'drive components'),
])
def test_torque_sign_and_estimation_corruption_are_detected(field, match):
    high, method = dynamics_fixture()
    high[field][40, 0] += .1
    with pytest.raises(ValueError, match=match):
        audit_dynamics_components(high, method)


def test_frozen_manifest_and_source_bytes_must_match(tmp_path):
    root = tmp_path / 'bags'
    root.mkdir()
    manifest = tmp_path / 'layout_manifest.yaml'
    manifest.write_text('positions: []\n')
    source = tmp_path / 'source_snapshot' / 'sampler.py'
    source.parent.mkdir()
    source.write_text('sensor = 1\n')
    fingerprints = {'sampler.py': hashlib.sha256(source.read_bytes()).hexdigest()}
    (tmp_path / 'collection_source_sha256.json').write_text(json.dumps(fingerprints))
    assert audit_provenance(root, manifest, require_snapshot=True)['frozen_source_files_checked'] == 1
    source.write_text('sensor = 2\n')
    with pytest.raises(ValueError, match='hash differs'):
        audit_provenance(root, manifest, require_snapshot=True)


def test_provenance_rejects_manifest_substitution_and_source_path_escape(tmp_path):
    root = tmp_path / 'bags'
    root.mkdir()
    manifest = tmp_path / 'layout_manifest.yaml'
    manifest.write_text('positions: []\n')
    alternate = tmp_path / 'alternate.yaml'
    alternate.write_text('positions: [other]\n')
    hashes = tmp_path / 'collection_source_sha256.json'
    hashes.write_text(json.dumps({'../outside.py': '0'*64}))
    with pytest.raises(ValueError, match='manifests differ'):
        audit_provenance(root, alternate)
    with pytest.raises(ValueError, match='escapes'):
        audit_provenance(root, manifest)


def test_every_interimage_physics_state_must_match_replay():
    high, _, _ = streams()
    comparison = HighRateStateComparison(high)
    for index in range(300):
        state = SimpleNamespace(qpos=high['qpos'][index], qvel=high['qvel'][index],
                                ctrl=high['ctrl'][index], time=high['timestamp'][index])
        comparison.observe(state)
    assert comparison.result()['samples'] == 300
    assert all(error == 0 for error in comparison.result()['maximum_errors'].values())


def test_high_rate_replay_detects_corruption_between_image_frames():
    high, _, _ = streams()
    comparison = HighRateStateComparison(high)
    for index in range(50):
        comparison.observe(SimpleNamespace(qpos=high['qpos'][index], qvel=high['qvel'][index],
                                           ctrl=high['ctrl'][index], time=high['timestamp'][index]))
    altered = high['qpos'][50].copy()
    altered[0] += 1e-4
    with pytest.raises(ValueError, match='tick 50: qpos'):
        comparison.observe(SimpleNamespace(qpos=altered, qvel=high['qvel'][50],
                                           ctrl=high['ctrl'][50], time=high['timestamp'][50]))


def test_partial_high_rate_replay_is_not_reported_as_complete():
    high, _, _ = streams()
    with pytest.raises(ValueError, match='every high-rate state'):
        HighRateStateComparison(high).result()
