"""Read-only verification of 1 kHz torque / 10 Hz RGB-D insertion recordings.

The numerical checks are intentionally independent of the recording sampler.
Optional action replay integrates the saved controls without calling the expert.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py
import numpy as np
import yaml

from mujoco_tools.paths import WORKSPACE_ROOT
from .audit_batch import audit_episode, require
from .audit_ring_dataset import (
    check_saved_scene, full_video_decode, require_ring_position, sha256,
    validate_positions, validate_stratification,
)


SIGNAL_SHAPES = {
    'timestamp': (), 'joint_position': (7,), 'joint_velocity': (7,),
    'joint_acceleration': (7,), 'tcp_pose': (7,),
    'joint_torque_measured_sim': (7,), 'joint_external_torque_raw': (7,),
    'joint_external_torque_estimated': (7,),
    'ee_external_wrench_world_estimated': (6,),
    'ee_external_wrench_tcp_estimated': (6,),
    'joint_actuator_torque': (7,), 'joint_pd_torque_unclipped': (7,),
    'finger_normal_force': (2,), 'finger_contact_force_world': (2, 3),
    'finger_contact_force_tcp': (2, 3), 'action': (8,),
    'image_index': (), 'action_index': (), 'phase': (),
}


def rotations_from_wxyz(quaternions):
    """Quaternion-to-world rotation, without sharing recorder conversion code."""
    quaternions = np.asarray(quaternions, dtype=float)
    require(quaternions.ndim == 2 and quaternions.shape[1] == 4,
            'Expected wxyz quaternion vectors')
    require(np.allclose(np.linalg.norm(quaternions, axis=1), 1, atol=1e-9, rtol=0),
            'TCP quaternion is not unit length')
    w, x, y, z = quaternions.T
    result = np.empty((len(w), 3, 3))
    result[:, 0, 0] = 1 - 2 * (y*y + z*z)
    result[:, 0, 1] = 2 * (x*y - z*w)
    result[:, 0, 2] = 2 * (x*z + y*w)
    result[:, 1, 0] = 2 * (x*y + z*w)
    result[:, 1, 1] = 1 - 2 * (x*x + z*z)
    result[:, 1, 2] = 2 * (y*z - x*w)
    result[:, 2, 0] = 2 * (x*z - y*w)
    result[:, 2, 1] = 2 * (y*z + x*w)
    result[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return result


def maximum_error(actual, expected):
    return float(np.max(np.abs(np.asarray(actual) - np.asarray(expected)), initial=0))


def check_equal(actual, expected, name, *, tolerance=1e-10):
    actual, expected = np.asarray(actual), np.asarray(expected)
    require(actual.shape == expected.shape, f'{name}: shape mismatch')
    error = maximum_error(actual, expected)
    require(error <= tolerance, f'{name}: maximum error {error:.6g} exceeds {tolerance:.6g}')
    return error


def audit_alignment(high, low, image_to_torque_index):
    """Verify every high-rate state/action against its physical time axis."""
    low_time = np.asarray(low['timestamp'])
    count = len(low_time)
    n = count * 100
    require(count > 0, 'No RGB-D samples')
    for name, shape in SIGNAL_SHAPES.items():
        require(name in high, f'Missing torque/{name}')
        require(np.shape(high[name]) == (n, *shape), f'Wrong torque/{name} dimensions')
    for name, value in high.items():
        array = np.asarray(value)
        require(array.shape[:1] == (n,), f'torque/{name} length differs from 1 kHz timeline')
        if np.issubdtype(array.dtype, np.number):
            require(np.isfinite(array).all(), f'Non-finite torque/{name}')
    check_equal(low_time, np.arange(count) / 10, '10 Hz timestamps', tolerance=1e-7)
    check_equal(high['timestamp'], np.arange(n) / 1000, '1 kHz timestamps', tolerance=1e-7)
    indices = np.arange(count, dtype=np.int64) * 100
    require(np.asarray(image_to_torque_index).dtype.kind in 'iu', 'Image indices must be integers')
    require(np.array_equal(image_to_torque_index, indices), 'Image-to-torque indices do not match physical ticks')
    repeat_index = np.arange(n, dtype=np.int64) // 100
    for name in ('action_index', 'image_index'):
        require(np.asarray(high[name]).dtype.kind in 'iu', f'{name} must use integer dtype')
        require(np.array_equal(high[name], repeat_index), f'{name} does not identify the active 100 ms interval')
    errors = {}
    for high_key, low_key in (
        ('qpos', 'qpos'), ('qvel', 'qvel'), ('ctrl', 'ctrl'),
        ('joint_position', 'joint_position'), ('joint_velocity', 'joint_velocity'),
    ):
        require(high_key in high, f'Missing torque/{high_key}')
        errors[high_key] = check_equal(np.asarray(high[high_key])[indices], low[low_key],
                                      f'RGB-D/torque {high_key} alignment')
    errors['action'] = check_equal(high['action'], np.asarray(low['action'])[repeat_index],
                                   'Held control target', tolerance=0)
    require(np.array_equal(np.asarray(high['phase']).astype(str),
                           np.asarray(low['phase']).astype(str)[repeat_index]),
            'High-rate phases disagree with the actual control intervals')
    require(np.all(np.asarray(high['finger_normal_force']) >= 0), 'Finger normal-force sum is negative')
    rotations = rotations_from_wxyz(np.asarray(high['tcp_pose'])[:, 3:])
    world = np.asarray(high['ee_external_wrench_world_estimated'])
    expected = np.concatenate((np.einsum('nji,nj->ni', rotations, world[:, :3]),
                               np.einsum('nji,nj->ni', rotations, world[:, 3:])), axis=1)
    errors['wrench_frame_rotation'] = check_equal(high['ee_external_wrench_tcp_estimated'], expected,
                                                 'TCP external wrench rotation', tolerance=1e-8)
    expected_fingers = np.einsum('nji,nkj->nki', rotations, high['finger_contact_force_world'])
    errors['finger_frame_rotation'] = check_equal(high['finger_contact_force_tcp'], expected_fingers,
                                                 'TCP finger force rotation', tolerance=1e-8)
    return dict(physics_samples=n, image_frames=count, samples_per_control_interval=100,
                first_timestamp_s=float(high['timestamp'][0]),
                last_timestamp_s=float(high['timestamp'][-1]),
                duration_s=count / 10, errors=errors)


def audit_filter(raw, filtered, cutoff_hz, dt=.001):
    """Recompute the causal filter from its complete recorded input history."""
    raw, filtered = np.asarray(raw), np.asarray(filtered)
    require(raw.shape == filtered.shape and raw.ndim == 2 and len(raw), 'Invalid external-torque filter arrays')
    require(np.isfinite(cutoff_hz) and 0 < cutoff_hz < .5/dt, 'Invalid filter cutoff')
    alpha = -np.expm1(-2 * np.pi * cutoff_hz * dt)
    expected = np.empty_like(raw, dtype=np.float64)
    expected[0] = raw[0]
    for index in range(1, len(raw)):
        expected[index] = expected[index - 1] + alpha * (raw[index] - expected[index - 1])
    error = check_equal(filtered, expected, 'Causal external-torque filter', tolerance=1e-9)
    return dict(cutoff_hz=float(cutoff_hz), alpha=float(alpha), maximum_error_nm=error)


def audit_dynamics_components(high, method):
    """Verify all recorded algebra, causal estimates and wrench inversions."""
    require(method.get('schema_version') == 'torque-signals-v1', 'Unknown signal method schema')
    require(method.get('external_sign') == 'robot_on_environment; opposite of environment_on_robot',
            'External-force sign convention is ambiguous')
    require(method.get('pose_order') == ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'], 'Unknown pose convention')
    require(method.get('finger_order') == ['left', 'right'], 'Finger identity convention is missing')
    virtual = method['virtual_measurement']
    require(virtual['kind'] == 'ideal_generalized_link_side_drive_proxy',
            'Virtual torque proxy must not be labelled a calibrated Franka sensor')
    estimator = method['external_estimator']
    require(estimator['method'] == 'exact_model_acceleration_dynamics_residual_then_causal_first_order_lowpass',
            'Unknown external-torque estimator')
    errors = {}
    proxy = (high['joint_actuator_torque'] + high['joint_passive_torque']
             + high['joint_internal_friction_torque'] - high['joint_rotor_inertia_torque'])
    errors['virtual_torque_nm'] = check_equal(high['joint_torque_measured_sim'], proxy,
                                            'Virtual link-side drive components')
    raw = proxy - high['joint_model_torque']
    errors['raw_residual_nm'] = check_equal(high['joint_external_torque_raw'], raw, 'Dynamics residual')
    truth = (high['joint_contact_torque_gt'] + high['joint_self_contact_torque_gt']
             + high['joint_noncontact_constraint_torque'] + high['joint_applied_external_torque'])
    # MuJoCo's iterative acceleration/constraint solution need not satisfy force
    # balance to machine precision. Independently replaying the first preflight's
    # worst sample reproduced its 4.617e-5 Nm error exactly as
    # qfrc_smooth + qfrc_constraint - M*qacc, not a missing contact component.
    errors['external_balance_nm'] = check_equal(raw, truth, 'External torque force balance', tolerance=1e-3)
    filter_report = audit_filter(raw, high['joint_external_torque_estimated'],
                                 estimator['filter_cutoff_hz'])
    times = high['timestamp']
    expected_alpha = np.r_[1., -np.expm1(-2 * np.pi * estimator['filter_cutoff_hz'] * np.diff(times))]
    errors['filter_alpha'] = check_equal(high['external_filter_alpha'], expected_alpha, 'Filter alpha')
    jac = high['tcp_jacobian_world']
    require(jac.shape == (len(raw), 6, 7), 'TCP Jacobian has incorrect dimensions')
    settings = method['wrench_estimator']
    length = float(settings['characteristic_length_m'])
    damping = float(settings['damping_m'])
    require(length > 0 and damping > 0, 'Wrench inversion regularization must be positive')
    scales = np.array([1, 1, 1, length, length, length])
    matrix = jac.transpose(0, 2, 1) * scales
    # Normal equations offer an independent calculation of the sampler's SVD
    # formula; regularization bounds its numerical conditioning.
    transpose = matrix.transpose(0, 2, 1)
    normal = transpose @ matrix + damping**2 * np.eye(6)
    rhs = np.einsum('nij,nj->ni', transpose, high['joint_external_torque_estimated'])
    coordinates = np.linalg.solve(normal, rhs[..., None])[..., 0]
    reconstructed = coordinates * scales
    errors['wrench_inverse'] = check_equal(high['ee_external_wrench_world_estimated'], reconstructed,
                                          'Independent DLS wrench inverse', tolerance=1e-6)
    singular = np.linalg.svd(matrix, compute_uv=False)
    errors['singular_values_m'] = check_equal(high['wrench_jacobian_singular_values'], singular,
                                             'Jacobian singular values')
    require(np.array_equal(high['wrench_estimate_well_conditioned'], singular[:, -1] > 10*damping),
            'Wrench conditioning flag disagrees with singular values')
    reconstructed_joint = np.einsum('nji,nj->ni', jac, high['ee_external_wrench_world_estimated'])
    residual = high['joint_external_torque_estimated'] - reconstructed_joint
    errors['wrench_joint_residual_nm'] = check_equal(high['wrench_joint_residual'], residual,
                                                   'Joint/wrench projection residual', tolerance=1e-9)
    rotations = rotations_from_wxyz(high['tcp_pose'][:, 3:])
    errors['rotation_matrix'] = check_equal(high['tcp_rotation'], rotations, 'Stored TCP rotation')
    ground_wrench = high['contact_wrench_at_tcp_world_gt']
    expected_tcp = np.concatenate((np.einsum('nji,nj->ni', rotations, ground_wrench[:, :3]),
                                   np.einsum('nji,nj->ni', rotations, ground_wrench[:, 3:])), axis=1)
    errors['contact_wrench_rotation'] = check_equal(high['contact_wrench_at_tcp_tcp_gt'], expected_tcp,
                                                   'Contact ground-truth wrench rotation', tolerance=1e-8)
    arm_q = np.asarray(method['arm_qpos_addresses'], dtype=int)
    arm_v = np.asarray(method['arm_dof_addresses'], dtype=int)
    for name, all_name, columns in [('joint_position', 'qpos', arm_q),
                                   ('joint_velocity', 'qvel', arm_v),
                                   ('joint_acceleration', 'qacc', arm_v)]:
        errors[name] = check_equal(high[name], high[all_name][:, columns], f'{name} DOF mapping')
    rotor = np.asarray(method['dof_armature'])[arm_v] * high['joint_acceleration']
    errors['rotor_inertia_torque_nm'] = check_equal(high['joint_rotor_inertia_torque'], rotor, 'Rotor inertia correction')
    require(np.array_equal(high['joint_actuator_saturated'].dtype, np.dtype(bool)),
            'Actuator saturation flags must be boolean')
    require(np.asarray(high['finger_contact_count']).dtype.kind in 'iu', 'Contact counts must be integers')
    require(np.all(high['finger_contact_count'] >= 0), 'Contact counts cannot be negative')
    return dict(errors=errors, filter=filter_report, solver_balance_tolerance_nm=1e-3,
                ill_conditioned_samples=int((~high['wrench_estimate_well_conditioned']).sum()),
                actuator_saturation_samples=int(np.any(high['joint_actuator_saturated'], axis=1).sum()),
                arm_contact_samples=int(np.count_nonzero(high['arm_external_contact_count'])),
                maximum_external_torque_nm=float(np.abs(raw).max()),
                maximum_filtered_external_torque_nm=float(np.abs(high['joint_external_torque_estimated']).max()),
                maximum_finger_normal_force_n=float(high['finger_normal_force'].max()),
                filtered_to_instantaneous_truth_max_nm=maximum_error(high['joint_external_torque_estimated'], truth),
                filtered_truth_error_note='Expected causal low-pass lag; no zero-lag equality is asserted')


def audit_model_snapshots(xml_path, high, method):
    """Rebuild geometry/dynamics at sampled states using only the saved XML."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    joint_ids = np.array([model.joint(name).id for name in method['joint_names']])
    arm = model.jnt_dofadr[joint_ids]
    qadr = model.jnt_qposadr[joint_ids]
    aids = np.array(method['arm_actuator_ids'])
    require(np.array_equal(arm, method['arm_dof_addresses'])
            and np.array_equal(qadr, method['arm_qpos_addresses']), 'Saved model joint addresses differ')
    tcp = model.site('tcp').id
    samples = set(np.linspace(0, len(high['timestamp']) - 1, 31).astype(int).tolist())
    samples.add(int(np.argmax(np.max(high['finger_normal_force'], axis=1))))
    samples.add(int(np.argmax(np.max(np.abs(high['joint_external_torque_raw']), axis=1))))
    mass = np.empty((model.nv, model.nv))
    jacp, jacr = np.empty((3, model.nv)), np.empty((3, model.nv))
    errors = {key: 0. for key in ('mass_model_torque_nm', 'jacobian', 'tcp_position_m',
                                 'tcp_rotation', 'passive_torque_nm', 'actuator_torque_nm',
                                 'requested_pd_torque_nm')}
    for index in sorted(samples):
        data.qpos[:] = high['qpos'][index]
        data.qvel[:] = high['qvel'][index]
        data.ctrl[:] = high['ctrl'][index]
        data.time = float(high['timestamp'][index])
        mujoco.mj_forward(model, data)
        try:
            mujoco.mj_fullM(model, data, mass)
        except TypeError:  # MuJoCo 3.3 and earlier Python signature.
            mujoco.mj_fullM(model, mass, data.qM)
        model_torque = (mass @ high['qacc'][index] + data.qfrc_bias
                        - model.dof_armature * high['qacc'][index])[arm]
        values = {'mass_model_torque_nm': (high['joint_model_torque'][index], model_torque),
                  'tcp_position_m': (high['tcp_pose'][index, :3], data.site_xpos[tcp]),
                  'tcp_rotation': (high['tcp_rotation'][index], data.site_xmat[tcp].reshape(3, 3)),
                  'passive_torque_nm': (high['joint_passive_torque'][index], data.qfrc_passive[arm]),
                  'actuator_torque_nm': (high['joint_actuator_torque'][index], data.qfrc_actuator[arm])}
        mujoco.mj_jacSite(model, data, jacp, jacr, tcp)
        values['jacobian'] = (high['tcp_jacobian_world'][index], np.vstack((jacp[:, arm], jacr[:, arm])))
        controls = data.ctrl[aids].copy()
        limited = model.actuator_ctrllimited[aids].astype(bool)
        limits = model.actuator_ctrlrange[aids]
        controls[limited] = np.clip(controls[limited], limits[limited, 0], limits[limited, 1])
        bias = model.actuator_biasprm[aids]
        expected_pd = (model.actuator_gainprm[aids, 0] * controls + bias[:, 0]
                       + bias[:, 1] * data.actuator_length[aids]
                       + bias[:, 2] * data.actuator_velocity[aids]) * model.actuator_gear[aids, 0]
        values['requested_pd_torque_nm'] = (high['joint_pd_torque_unclipped'][index], expected_pd)
        for name, (actual, expected) in values.items():
            error = check_equal(actual, expected, f'XML snapshot {index}: {name}', tolerance=1e-8)
            errors[name] = max(errors[name], error)
    return dict(samples=len(samples), sample_indices=sorted(samples), errors=errors,
                note='Fresh model geometry, mass matrix and controls; uses saved instantaneous accelerations')


class HighRateStateComparison:
    """Compare every pre-integration state, including ticks between images."""

    def __init__(self, high):
        self.high = high
        self.count = 0
        self.errors = {key: 0. for key in ('qpos', 'qvel', 'ctrl', 'timestamp')}

    def observe(self, data):
        index = self.count
        require(index < len(self.high['timestamp']), 'Replay produced extra high-rate states')
        values = {'qpos': data.qpos, 'qvel': data.qvel, 'ctrl': data.ctrl, 'timestamp': data.time}
        for name, actual in values.items():
            error = check_equal(actual, self.high[name][index],
                                f'1 kHz physics replay tick {index}: {name}', tolerance=1e-10)
            self.errors[name] = max(self.errors[name], error)
        self.count += 1

    def result(self):
        require(self.count == len(self.high['timestamp']), 'Replay did not visit every high-rate state')
        return dict(samples=self.count, maximum_errors=self.errors,
                    state_comparison='qpos/qvel/ctrl/time before every integration; no state restoration')


def replay_high_rate_states(xml_path, config, high, actions):
    from ..environment.simulation import Simulation

    sim = Simulation(xml_path, config, render=False)
    comparison = HighRateStateComparison(high)
    try:
        for action in actions:
            sim.step(action, before_substep=lambda active: comparison.observe(sim.data))
        return comparison.result()
    finally:
        sim.close()


def read_streams(handle):
    require('torque' in handle and isinstance(handle['torque'], h5py.Group), 'Missing torque group')
    require('alignment/image_to_torque_index' in handle, 'Missing image/torque alignment indices')
    high = {name: dataset.asstr()[:] if h5py.check_string_dtype(dataset.dtype) else dataset[:]
            for name, dataset in handle['torque'].items() if isinstance(dataset, h5py.Dataset)}
    low = {'timestamp': handle['timestamp'][:], 'action': handle['action'][:],
           'phase': handle['phase'].asstr()[:]}
    for name in ('qpos', 'qvel', 'ctrl'):
        low[name] = handle[f'simulation/{name}'][:]
    for name in ('joint_position', 'joint_velocity'):
        low[name] = handle[f'observations/{name}'][:]
    return high, low, handle['alignment/image_to_torque_index'][:]


def audit_recording(root, row, *, replay=False, manifest_sha256=None, source_sha256=None):
    """Audit one complete episode; preserve failures as explicit report entries."""
    folder = Path(root) / f"episode_{row['episode_id']:06d}"
    result = dict(episode=folder.name, position=row['name'], valid=False, errors=[])
    try:
        record = audit_episode(folder, sample_hz=10, control_hz=10)
        result['recording'] = record
        config = yaml.safe_load((folder / f'{folder.name}.yaml').read_text())
        collection = config.get('collection', {})
        require(collection.get('dataset_version') == 'insert_plug_torque_v1'
                and collection.get('split') == 'ind', 'Wrong collection identity or split')
        require(collection.get('freshly_executed') is True
                and collection.get('force_feedback_enabled') is False,
                'Expected fresh expert execution with passive torque logging')
        require(collection.get('episode_id') == row['episode_id']
                and collection.get('position_id') == row['name'], 'Episode/layout identities disagree')
        if manifest_sha256 is not None:
            require(collection.get('layout_manifest_sha256') == manifest_sha256,
                    'Episode points to a different layout manifest')
        if source_sha256 is not None:
            require(collection.get('source_sha256') == source_sha256,
                    'Episode source hashes differ from the frozen collection sources')
        require(config['cameras']['width'] == 256 and config['cameras']['height'] == 256,
                'Expected native 256 x 256 RGB-D')
        require(record['collision_checks_enabled'] is False and record['unsafe_contacts'] is None,
                'Ordinary contact must not be rejected by strict collision diagnostics')
        for side in ('source', 'target'):
            require_ring_position(record[f'{side}_xy'], side)
            check_equal(record[f'{side}_xy'], row[f'{side}_xy'], f'{side} frozen layout')
        h5_path = folder / f'{folder.name}.h5'
        with h5py.File(h5_path, 'r') as handle:
            metadata = json.loads(handle.attrs['metadata_json'])
            high, low, indices = read_streams(handle)
            result['torque'] = audit_alignment(high, low, indices)
            require(handle['torque'].attrs.get('schema_version') == 'insert-plug-torque-v1',
                    'Unknown high-rate schema')
            require(handle['torque'].attrs.get('sample_hz') == 1000, 'Incorrect torque sample frequency')
            require('torque_recording' in metadata, 'Missing high-rate metadata')
            require(int(handle['torque'].attrs['samples']) == len(high['timestamp']), 'Incorrect high-rate sample count')
            timing = metadata['torque_recording']
            require(timing.get('sample_hz') == 1000 and timing.get('image_hz') == 10
                    and timing.get('samples_per_image') == 100, 'Wrong declared high/low frequency relationship')
            require(timing.get('timing') == 'before_physics_integration_after_active_control_target_applied'
                    and timing.get('time_interval') == '[initial_time, final_time)',
                    'High-rate sample timing is ambiguous')
            require(timing.get('force_feedback_enabled') is False
                    and timing.get('simulation_truth_is_not_real_robot_measurement') is True,
                    'Missing force-stream provenance distinction')
            method = timing['method']
            for name in ('joint_torque_measured_sim', 'joint_external_torque_raw',
                         'joint_external_torque_estimated', 'joint_actuator_torque',
                         'joint_pd_torque_unclipped'):
                require(handle[f'torque/{name}'].attrs.get('units') == 'N*m', f'{name} units missing or wrong')
            for name in ('ee_external_wrench_world_estimated', 'ee_external_wrench_tcp_estimated'):
                require(handle[f'torque/{name}'].attrs.get('units') == '[N,N,N,N*m,N*m,N*m]',
                        f'{name} units missing or wrong')
            for name in ('finger_normal_force', 'finger_contact_force_world', 'finger_contact_force_tcp'):
                require(handle[f'torque/{name}'].attrs.get('units') == 'N', f'{name} units missing or wrong')
            result['signal_recomputation'] = audit_dynamics_components(high, method)
            result['model_snapshots'] = audit_model_snapshots(folder / f'{folder.name}.xml', high, method)
        result['head_visibility'] = check_saved_scene(folder, config, row)
        result['fully_decoded_video_frames'] = full_video_decode(folder / f'{folder.name}.mp4', record['frames'])
        result['files'] = {path.name: dict(bytes=path.stat().st_size, sha256=sha256(path))
                           for path in sorted(folder.iterdir())}
        if replay:
            from .check_action_replay import replay_episode
            outcome = replay_episode(h5_path, action_dtype='float32', contact_diagnostics=False)
            result['action_replay'] = outcome
            require(outcome['success'] is True and outcome['runtime_error'] is None,
                    'Saved-action physics replay failed')
            require(outcome['max_joint_deviation_from_source_rad'] <= 1e-10
                    and outcome['max_peg_deviation_from_source_m'] <= 1e-10,
                    'Recorded physical trajectory differs from uninstrumented action replay')
            require(outcome['final']['final_stability_physics_samples'] == 251,
                    'Replay final stability window is incomplete')
            result['high_rate_state_replay'] = replay_high_rate_states(
                folder / f'{folder.name}.xml', config, high, low['action'])
            require(all(sha256(folder / name) == value['sha256'] for name, value in result['files'].items()),
                    'Read-only replay changed a source recording')
        result['valid'] = True
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    return result


def audit_provenance(root, manifest_path, *, require_snapshot=False):
    """Check frozen manifest bytes and collection sources without importing them."""
    parent = Path(root).parent
    frozen_manifest = parent / 'layout_manifest.yaml'
    require(frozen_manifest.is_file() and not frozen_manifest.is_symlink(), 'Missing regular frozen layout manifest')
    require(frozen_manifest.read_bytes() == Path(manifest_path).read_bytes(),
            'Frozen and requested layout manifests differ')
    fingerprints_path = parent / 'collection_source_sha256.json'
    require(fingerprints_path.is_file() and not fingerprints_path.is_symlink(), 'Missing frozen collection source hashes')
    fingerprints = json.loads(fingerprints_path.read_text())
    require(isinstance(fingerprints, dict) and fingerprints, 'Source fingerprint map is empty')
    snapshot = parent / 'source_snapshot'
    if require_snapshot:
        require(snapshot.is_dir() and not snapshot.is_symlink(), 'Missing regular frozen source snapshot')
    checked = 0
    for name, expected in fingerprints.items():
        relative = Path(name)
        require(not relative.is_absolute() and '..' not in relative.parts,
                'Frozen source name escapes its snapshot')
        require(isinstance(expected, str) and len(expected) == 64, 'Invalid SHA256 in source map')
        if snapshot.exists():
            path = snapshot / relative
            require(path.is_file() and not path.is_symlink(), f'Missing frozen source: {name}')
            require(sha256(path) == expected, f'Frozen source hash differs: {name}')
            checked += 1
    return dict(layout_manifest_sha256=sha256(frozen_manifest), source_sha256=fingerprints,
                frozen_source_files_checked=checked,
                audit_source_sha256=sha256(Path(__file__)))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, default=WORKSPACE_ROOT / 'config/insert_plug_positions_v2.yaml')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--episode', type=int, action='append')
    parser.add_argument('--replay', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('workers must be positive')
    root, output = args.root.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists() or output.is_relative_to(root):
        parser.error('Choose a new audit output outside the read-only dataset root')
    manifest = yaml.safe_load(args.manifest.read_text())
    rows = validate_positions(manifest)
    strata = validate_stratification(rows)
    if args.episode:
        requested = set(args.episode)
        if not requested <= set(range(1, 101)):
            parser.error('episode IDs must lie in 1..100')
        rows = [row for row in rows if row['episode_id'] in requested]
    entries = list(root.iterdir())
    require(all(path.is_dir() and not path.is_symlink() for path in entries), 'Non-directory or symlink in bags')
    allowed = {f'episode_{index:06d}' for index in range(1, 101)}
    require({path.name for path in entries} <= allowed, 'Unexpected directory in bags')
    if not args.episode:
        require({path.name for path in entries} == allowed, 'Expected exactly 100 complete episode directories')
    provenance = audit_provenance(root, args.manifest, require_snapshot=not bool(args.episode))
    report = dict(check='insert_plug_torque_v1', root=str(root),
                  manifest=str(args.manifest.resolve()), manifest_sha256=sha256(args.manifest),
                  started_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                  requested_episodes=len(rows), replay=args.replay, stratification=strata,
                  provenance=provenance, valid=False, results=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=multiprocessing.get_context('spawn')) as pool:
        pending = [pool.submit(audit_recording, root, row, replay=args.replay,
                               manifest_sha256=provenance['layout_manifest_sha256'],
                               source_sha256=provenance['source_sha256']) for row in rows]
        for future in as_completed(pending):
            result = future.result()
            report['results'].append(result)
            print(json.dumps(dict(episode=result['episode'], valid=result['valid'], errors=result['errors'])), flush=True)
    report['results'].sort(key=lambda row: row['episode'])
    report.update(passed=sum(row['valid'] for row in report['results']),
                  image_frames=sum(row.get('recording', {}).get('frames', 0) for row in report['results']),
                  physics_samples=sum(row.get('torque', {}).get('physics_samples', 0) for row in report['results']),
                  fully_decoded_video_frames=sum(row.get('fully_decoded_video_frames', 0) for row in report['results']),
                  total_bytes=sum(row.get('recording', {}).get('bytes', 0) for row in report['results']))
    report['valid'] = report['passed'] == len(rows)
    signals = [row['signal_recomputation'] for row in report['results'] if 'signal_recomputation' in row]
    replays = [row['action_replay'] for row in report['results'] if 'action_replay' in row]
    report['numerical_summary'] = dict(
        maximum_force_balance_error_nm=max((row['errors']['external_balance_nm'] for row in signals), default=None),
        maximum_filter_reconstruction_error_nm=max((row['filter']['maximum_error_nm'] for row in signals), default=None),
        maximum_wrench_inverse_error=max((row['errors']['wrench_inverse'] for row in signals), default=None),
        ill_conditioned_samples=sum(row['ill_conditioned_samples'] for row in signals),
        actuator_saturation_samples=sum(row['actuator_saturation_samples'] for row in signals),
        action_replay_episodes=len(replays),
        maximum_replay_joint_deviation_rad=max((row['max_joint_deviation_from_source_rad'] for row in replays), default=None),
        maximum_replay_peg_deviation_m=max((row['max_peg_deviation_from_source_m'] for row in replays), default=None),
        high_rate_replayed_samples=sum(row.get('high_rate_state_replay', {}).get('samples', 0)
                                      for row in report['results']),
        high_rate_maximum_errors={key: max((row['high_rate_state_replay']['maximum_errors'][key]
                                          for row in report['results'] if 'high_rate_state_replay' in row), default=None)
                                  for key in ('qpos', 'qvel', 'ctrl', 'timestamp')},
    )
    report['completed_at'] = datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
    with output.open('x') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in ('valid', 'passed', 'image_frames', 'physics_samples')}), flush=True)
    return 0 if report['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
