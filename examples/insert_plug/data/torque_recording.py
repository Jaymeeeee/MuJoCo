"""Independent 1 kHz force/torque stream beside the ordinary 10 Hz images.

The image schema remains insert-plug-v3. High-rate samples occupy ``torque/``
and use the half-open interval [initial_time, final_time): exactly one sample
immediately before every physics integration, after its active target is set.
"""
from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np

from .dataset import EpisodeWriter, validate_episode


TORQUE_SCHEMA = 'insert-plug-torque-v1'
TORQUE_HZ = 1000.0
IMAGE_HZ = 10.0
SAMPLES_PER_IMAGE = 100
REQUIRED_TORQUE_SHAPES = {
    'joint_torque_measured_sim': (7,),
    'joint_external_torque_raw': (7,),
    'joint_external_torque_estimated': (7,),
    'ee_external_wrench_world_estimated': (6,),
    'ee_external_wrench_tcp_estimated': (6,),
    'joint_actuator_torque': (7,),
    'joint_pd_torque_unclipped': (7,),
    'finger_normal_force': (2,),
    'finger_contact_force_world': (2, 3),
    'finger_contact_force_tcp': (2, 3),
    'joint_position': (7,),
    'joint_velocity': (7,),
    'joint_acceleration': (7,),
    'tcp_pose': (7,),
    'action': (8,),
    'timestamp': (),
    'image_index': (),
    'action_index': (),
    'phase': (),
}


class TorqueEpisodeWriter(EpisodeWriter):
    """Buffer numeric streams separately; retain original image writer behavior."""

    def __init__(self, path, metadata, *, buffer_size=1000):
        if buffer_size < 1:
            raise ValueError('buffer_size must be positive')
        super().__init__(path, metadata)
        self._torque_group = self._file.create_group('torque')
        self._torque_group.attrs.update(
            schema_version=TORQUE_SCHEMA, sample_hz=TORQUE_HZ,
            time_interval='[initial_time, final_time)',
            timing='before_physics_integration_after_active_control_target_applied',
            samples=0,
        )
        self._torque_datasets = {}
        self._torque_shapes = {}
        self._torque_buffer = []
        self._buffer_size = int(buffer_size)
        self.torque_samples = 0
        self._torque_origin = float(metadata.get('sample_origin_s', 0.0))

    def append_torque(self, record):
        if self._closed:
            raise RuntimeError('EpisodeWriter is closed')
        if not isinstance(record, dict) or not REQUIRED_TORQUE_SHAPES.keys() <= record.keys():
            raise ValueError('Torque sample is missing required fields')
        expected_time = self._torque_origin + self.torque_samples / TORQUE_HZ
        if not np.isclose(record['timestamp'], expected_time, rtol=0, atol=1e-8):
            raise ValueError(f'Missed or repeated torque sample: expected {expected_time}')
        expected_index = self.torque_samples // SAMPLES_PER_IMAGE
        for key in ('image_index', 'action_index'):
            if int(record[key]) != expected_index or record[key] != int(record[key]):
                raise ValueError(f'{key} does not match high-rate sample time')
        values = {}
        for key, value in record.items():
            if not key or '/' in key:
                raise ValueError('Torque field names must be nonempty and flat')
            if key == 'phase':
                value = str(value)
                if not value or '\x00' in value:
                    raise ValueError('phase must be nonempty and contain no NUL')
            else:
                value = np.asarray(value)
                if value.dtype.kind not in 'biuf' or not np.isfinite(value).all():
                    raise ValueError(f'Torque field {key} must be finite numeric values')
                value = value.copy()
            shape = np.shape(value)
            if key in REQUIRED_TORQUE_SHAPES and shape != REQUIRED_TORQUE_SHAPES[key]:
                raise ValueError(f'Torque field {key} has shape {shape}')
            values[key] = value
        shapes = {key: np.shape(value) for key, value in values.items()}
        if self._torque_shapes and shapes != self._torque_shapes:
            raise ValueError('Torque fields or shapes changed during recording')
        if not self._torque_shapes:
            self._torque_shapes = shapes
            for key, value in values.items():
                dtype = h5py.string_dtype('utf-8') if key == 'phase' else np.asarray(value).dtype
                self._torque_datasets[key] = self._torque_group.create_dataset(
                    key, shape=(0, *shapes[key]), maxshape=(None, *shapes[key]),
                    chunks=(min(self._buffer_size, 1000), *shapes[key]),
                    dtype=dtype, compression='lzf')
            scalar_units = {
                'timestamp': 's', 'joint_position': 'rad', 'joint_position_target': 'rad',
                'joint_velocity': 'rad/s', 'joint_acceleration': 'rad/s^2',
                'gripper_width': 'm', 'wrench_joint_residual': 'N*m',
                'wrench_jacobian_singular_values': 'm',
                'external_filter_alpha': '1', 'tcp_rotation': '1',
                'wrench_estimate_well_conditioned': 'boolean',
                'joint_actuator_saturated': 'boolean',
                'image_index': 'index', 'action_index': 'index',
                'finger_contact_count': 'count', 'robot_external_contact_count': 'count',
                'qpos': 'MuJoCo generalized position (mixed SI/quaternion)',
                'qvel': 'MuJoCo generalized velocity (mixed m/s,rad/s)',
                'qacc': 'MuJoCo generalized acceleration (mixed m/s^2,rad/s^2)',
                'ctrl': 'MuJoCo actuator control (per-transmission units)',
                'action': '[rad,rad,rad,rad,rad,rad,rad,m]',
                'tcp_pose': '[m,m,m,1,1,1,1]',
                'tcp_jacobian_world': '[linear:m/rad;angular:rad/rad]',
            }
            for key in values:
                if key in scalar_units:
                    units = scalar_units[key]
                elif key.startswith('joint_') and 'torque' in key:
                    units = 'N*m'
                elif key.startswith(('ee_external_wrench_', 'contact_wrench_')):
                    units = '[N,N,N,N*m,N*m,N*m]'
                elif key.startswith('finger_') and 'force' in key:
                    units = 'N'
                else:
                    continue
                self._torque_datasets[key].attrs['units'] = units
        self._torque_buffer.append(values)
        self.torque_samples += 1
        if len(self._torque_buffer) >= self._buffer_size:
            self.flush_torque()

    def flush_torque(self):
        if not self._torque_buffer:
            return
        count = len(self._torque_buffer)
        start = self.torque_samples - count
        try:
            for key, dataset in self._torque_datasets.items():
                dataset.resize(self.torque_samples, axis=0)
                dataset[start:self.torque_samples] = [row[key] for row in self._torque_buffer]
        except Exception:
            for dataset in self._torque_datasets.values():
                dataset.resize(start, axis=0)
            self._file.flush()
            raise
        self._torque_buffer.clear()
        self._torque_group.attrs['samples'] = self.torque_samples
        self._file.flush()

    def finalize(self, success, failure_reason='', **extra_metadata):
        self.flush_torque()
        if self.torque_samples != self.steps * SAMPLES_PER_IMAGE:
            raise ValueError('Expected exactly 100 torque samples per image/control frame')
        indices = np.arange(self.steps, dtype=np.int64) * SAMPLES_PER_IMAGE
        self._file.create_dataset('alignment/image_to_torque_index', data=indices)
        self._file['alignment'].attrs['meaning'] = 'identical timestamp/state and active action'
        for low, high in (
            ('timestamp', 'timestamp'), ('action', 'action'),
            ('simulation/qpos', 'qpos'), ('simulation/qvel', 'qvel'),
            ('simulation/ctrl', 'ctrl'),
        ):
            if not np.array_equal(self._file[low][...], self._torque_datasets[high][indices]):
                raise ValueError(f'Image/torque alignment failed for {low}')
        extra_metadata['torque_samples'] = self.torque_samples
        extra_metadata['torque_schema_version'] = TORQUE_SCHEMA
        return super().finalize(success, failure_reason, **extra_metadata)

    def close(self):
        if not self._closed:
            self.flush_torque()
        super().close()


def validate_torque_episode(path: str | Path):
    """Validate both time axes and their shared states/actions without rendering."""
    result = validate_episode(path)
    errors = result['errors']
    try:
        with h5py.File(path, 'r') as handle:
            group = handle['torque']
            if group.attrs.get('schema_version') != TORQUE_SCHEMA or group.attrs.get('sample_hz') != TORQUE_HZ:
                errors.append('Unsupported torque schema or sample rate')
            count = int(group.attrs.get('samples', -1))
            result['torque_samples'] = count
            expected_count = result['steps'] * SAMPLES_PER_IMAGE
            if count != expected_count or count <= 0:
                errors.append('Torque/image frame counts do not have the required 100:1 ratio')
            times = group['timestamp'][...]
            origin = float(result['metadata'].get('sample_origin_s', 0))
            if times.shape != (count,) or not np.allclose(times, origin + np.arange(count)/TORQUE_HZ, rtol=0, atol=1e-8):
                errors.append('Torque timestamps are not contiguous 1 kHz samples')
            if result['metadata'].get('torque_samples') != count:
                errors.append('Torque metadata count differs')
            for key, shape in REQUIRED_TORQUE_SHAPES.items():
                if key not in group or group[key].shape != (count, *shape):
                    errors.append(f'Missing or wrong torque dataset shape: {key}')
            for key, dataset in group.items():
                if not isinstance(dataset, h5py.Dataset) or dataset.shape[0] != count:
                    errors.append(f'Inconsistent torque stream: {key}')
                    continue
                if key == 'phase':
                    if any(not value for value in dataset.asstr()[...]):
                        errors.append('Empty torque phase')
                elif not np.isfinite(dataset[...]).all():
                    errors.append(f'Nonfinite torque stream: {key}')
            indices = handle['alignment/image_to_torque_index'][...]
            expected_indices = np.arange(result['steps'], dtype=np.int64) * SAMPLES_PER_IMAGE
            if not np.array_equal(indices, expected_indices):
                errors.append('Incorrect image-to-torque indices')
            for low, high in (
                ('timestamp', 'timestamp'), ('action', 'action'),
                ('simulation/qpos', 'qpos'), ('simulation/qvel', 'qvel'), ('simulation/ctrl', 'ctrl'),
                ('observations/joint_position', 'joint_position'),
                ('observations/joint_velocity', 'joint_velocity'),
            ):
                if not np.array_equal(handle[low][...], group[high][indices]):
                    errors.append(f'Low/high frequency state mismatch: {low}')
            action_indices = np.arange(count, dtype=np.int64) // SAMPLES_PER_IMAGE
            for name in ('action_index', 'image_index'):
                if not np.array_equal(group[name][...], action_indices):
                    errors.append(f'Incorrect torque/{name}')
            if not np.array_equal(group['action'][...], handle['action'][...][action_indices]):
                errors.append('High-rate actions differ from active 10 Hz targets')
            if not np.array_equal(group['phase'].asstr()[...], handle['phase'].asstr()[...][action_indices]):
                errors.append('High-rate phases differ from image phases')
            if np.any(group['finger_normal_force'][...] < 0):
                errors.append('Negative summed normal contact force')
            result['duration_s'] = count / TORQUE_HZ
    except (OSError, KeyError, TypeError, ValueError, RuntimeError, IndexError) as exc:
        errors.append(f'Cannot validate torque stream: {exc}')
    result['valid'] = not errors
    return result
