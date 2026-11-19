"""Paired RGB-D patch interventions on frozen Diffusion observations.

These are action sensitivities to a specified replacement, not Transformer
attention or signed/causal pixel attribution. No masked action is executed.
Use a dedicated offline adapter: every prediction resets its policy/processors.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from collections.abc import Mapping

import numpy as np


@dataclass(frozen=True)
class PairedChunkPrediction:
    normalized: np.ndarray
    physical: np.ndarray
    final_rng: dict


def capture_rng_state(torch, *, include_cuda=None):
    """Copy CPU and all initialized CUDA generators without drawing noise.

    CPU-only work does not initialize CUDA. Once a CUDA policy has been loaded,
    all device states are included. ``include_cuda=True`` explicitly requests
    them, for callers that deliberately initialize CUDA beforehand.
    """
    if include_cuda is None:
        include_cuda = torch.cuda.is_initialized()
    return {'cpu': torch.get_rng_state().clone(),
            'cuda': [state.clone() for state in torch.cuda.get_rng_state_all()]
            if include_cuda else []}


def restore_rng_state(torch, state):
    """Restore the complete generator state saved before one real prediction."""
    if not isinstance(state, Mapping) or set(state) != {'cpu', 'cuda'}:
        raise ValueError('RNG state must contain exactly cpu and cuda')
    values = [state['cpu'], *state['cuda']] if isinstance(state['cuda'], (list, tuple)) else []
    if not values or any(not isinstance(value, torch.Tensor) or value.device.type != 'cpu'
                         or value.dtype != torch.uint8 or value.ndim != 1 for value in values):
        raise ValueError('RNG states must be one-dimensional CPU uint8 tensors')
    if state['cuda'] and (not torch.cuda.is_available()
                          or len(state['cuda']) != torch.cuda.device_count()):
        raise ValueError('Saved CUDA RNG states must cover every visible CUDA device')
    if torch.cuda.is_initialized() and not state['cuda']:
        raise ValueError('Initialized CUDA generators are missing from the RNG state')
    torch.set_rng_state(state['cpu'])
    if state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def rng_states_equal(torch, first, second):
    return (len(first['cuda']) == len(second['cuda'])
            and torch.equal(first['cpu'], second['cpu'])
            and all(torch.equal(a, b) for a, b in zip(first['cuda'], second['cuda'], strict=True)))


def observation_fingerprint(observation):
    """Hash every source array, its name, shape and dtype to detect mutation."""
    digest = hashlib.sha256()
    for key in sorted(observation):
        array = np.asarray(observation[key])
        if array.dtype.hasobject:
            raise ValueError('Observation arrays must not contain Python objects')
        for part in (str(key), str(array.shape), array.dtype.str):
            digest.update(part.encode('utf-8'))
            digest.update(b'\0')
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def predict_paired_chunk(adapter, observation, rng_state):
    """Predict one batch-one chunk using exactly the supplied full noise stream.

    Reset happens BEFORE RNG restoration. For n_obs_steps=1 the empty-queue
    path requires a history axis on state as well as on both RGB-D streams.
    Each predicted action is postprocessed as [1,8], matching select_action.
    Initial diffusion noise AND every scheduler noise draw are therefore paired.
    Returned arrays are owned float32 [n_action_steps,8] arrays.
    """
    torch = adapter._torch
    if getattr(adapter, 'policy_type', None) != 'diffusion':
        raise ValueError('Paired chunks require a Diffusion adapter')
    if adapter.n_obs_steps != 1:
        raise ValueError('Frozen-observation inference requires n_obs_steps=1')
    if tuple(adapter.action_shape) != (8,):
        raise ValueError('Sensitivity units require seven joint angles and one aperture action')
    if getattr(adapter.policy, 'training', False):
        raise ValueError('Paired inference requires the policy in evaluation mode')
    steps = adapter.n_action_steps
    if type(steps) is not int or steps < 1:
        raise ValueError('n_action_steps must be a positive integer')
    before = observation_fingerprint(observation)
    adapter.reset()
    restore_rng_state(torch, rng_state)
    queues = getattr(adapter.policy, '_queues', {})
    if any(len(queue) for queue in queues.values()):
        raise ValueError('Reset must clear all Diffusion observation and action queues')
    with torch.inference_mode():
        prepared = adapter.preprocessor(adapter._prepare_observation(observation))
        batch = {}
        for key, feature in adapter.input_features.items():
            value = prepared.get(key)
            expected = (1, *feature['shape'])
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
                raise ValueError(f'Preprocessed {key} must have batch-one shape {expected}')
            batch[key] = value.unsqueeze(1)
        prediction = adapter.policy.predict_action_chunk(batch)
        expected = (1, steps, 8)
        if not isinstance(prediction, torch.Tensor) or tuple(prediction.shape) != expected:
            raise ValueError(f'Diffusion chunk shape must be {expected}')
        if not bool(torch.isfinite(prediction).all()):
            raise ValueError('Diffusion produced nonfinite normalized actions')
        normalized = prediction[0].detach().to(device='cpu', dtype=torch.float32).numpy().copy()
        physical = []
        for query in range(steps):
            result = adapter.postprocessor(prediction[:, query, :].clone())
            if not isinstance(result, torch.Tensor) or tuple(result.shape) != (1, 8):
                raise ValueError('Postprocessed action must have batch-one shape (1, 8)')
            physical.append(result[0].detach().to(device='cpu', dtype=torch.float32).numpy().copy())
        physical = np.stack(physical)
    if not np.isfinite(physical).all():
        raise ValueError('Diffusion produced nonfinite physical actions')
    if observation_fingerprint(observation) != before:
        raise RuntimeError('Paired prediction mutated a source observation')
    return PairedChunkPrediction(normalized, physical, capture_rng_state(torch))


def _rgb_depth(observation, camera, depth_camera_map):
    if camera not in depth_camera_map or depth_camera_map[camera] not in observation:
        raise ValueError('Each RGB camera must have its declared paired depth input')
    image = np.asarray(observation[camera])
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError('Camera image must be HWC RGB uint8')
    depth_key = depth_camera_map[camera]
    depth = np.asarray(observation[depth_key])
    height, width = image.shape[:2]
    if depth.dtype != np.float32 or depth.shape not in ((height, width), (height, width, 1), (1, height, width)):
        raise ValueError('Paired depth must be aligned HW/HWC1/CHW float32 metres')
    if not np.isfinite(depth).all() or np.any(depth < 0):
        raise ValueError('Paired depth must be finite and nonnegative')
    return image, depth_key, depth


def mask_patch(observation, camera, row, column, grid_size=8, *, depth_camera_map):
    """Jointly replace one RGB-D cell by full-frame RGB mean/depth median.

    RGB uses the rounded per-channel mean of the original uint8 image. Depth
    uses the median of all original depth pixels, in metres, including zeros.
    Other pixels, the other camera and state retain their original values.
    The original mapping/arrays are never modified.
    """
    image, depth_key, depth = _rgb_depth(observation, camera, depth_camera_map)
    if isinstance(grid_size, bool) or not isinstance(grid_size, (int, np.integer)) or grid_size < 1:
        raise ValueError('grid_size must be a positive integer')
    height, width = image.shape[:2]
    if height % grid_size or width % grid_size:
        raise ValueError('Image dimensions must be divisible by grid_size')
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer))
           or not 0 <= value < grid_size for value in (row, column)):
        raise ValueError('Patch row and column must be integer grid indices')
    y = slice(row * height // grid_size, (row + 1) * height // grid_size)
    x = slice(column * width // grid_size, (column + 1) * width // grid_size)
    masked_rgb = image.copy()
    masked_rgb[y, x] = np.rint(image.mean(axis=(0, 1), dtype=np.float64)).astype(np.uint8)
    masked_depth = depth.copy()
    median = np.median(depth).astype(np.float32)
    if depth.ndim == 2:
        masked_depth[y, x] = median
    elif depth.shape[0] == 1 and depth.shape[1:] == (height, width):
        masked_depth[:, y, x] = median
    else:
        masked_depth[y, x, :] = median
    return {**observation, camera: masked_rgb, depth_key: masked_depth}


def sensitivity_metrics(physical_predictions, baseline_physical):
    """Separate joint-angle degrees from gripper-aperture millimetres."""
    prediction = np.asarray(physical_predictions, dtype=np.float64)
    baseline = np.asarray(baseline_physical, dtype=np.float64)
    if baseline.ndim != 2 or baseline.shape[-1] != 8 or prediction.shape[-2:] != baseline.shape:
        raise ValueError('Predictions must end in the baseline [time,8] shape')
    if not np.isfinite(prediction).all() or not np.isfinite(baseline).all():
        raise ValueError('Sensitivity inputs must be finite')
    delta = prediction - baseline
    arm_deg = np.rad2deg(delta[..., :7])
    gripper_mm = delta[..., 7] * 1000.0
    return {
        'arm_rms_deg': np.sqrt(np.mean(arm_deg**2, axis=(-2, -1))),
        'gripper_rms_mm': np.sqrt(np.mean(gripper_mm**2, axis=-1)),
        'arm_rms_deg_per_query': np.sqrt(np.mean(arm_deg**2, axis=-1)),
        'gripper_abs_mm_per_query': np.abs(gripper_mm),
    }


def compute_sensitivity(adapter, observation, reference_actions, pre_rng, post_rng,
                        grid_size=8, *, progress=None):
    """Scan both cameras' cells at one recorded prediction observation.

    Baseline physical actions must equal the recorded executed prefix bitwise;
    the last trajectory chunk can have fewer than n_action_steps actions.
    Every variant restores pre_rng and must end at post_rng. This fixes all
    DDPM randomness while keeping batch size one and action postprocessing
    identical. The caller's RNG is restored even on failure. Adapter queues
    and processors are reset, so this function is for offline analysis only.
    """
    torch = adapter._torch
    outer_rng = capture_rng_state(torch)
    source_hash = observation_fingerprint(observation)
    cameras = tuple(key for key in adapter.input_features if key in adapter.depth_camera_map)
    if len(cameras) != 2 or len(set(adapter.depth_camera_map.values())) != 2:
        raise ValueError('This analysis requires two distinct paired RGB-D cameras')
    # Validate the mask contract before spending a baseline model inference.
    for camera in cameras:
        mask_patch(observation, camera, 0, 0, grid_size, depth_camera_map=adapter.depth_camera_map)
    reference = np.asarray(reference_actions)
    if reference.dtype != np.float32 or reference.ndim != 2 or reference.shape[1:] != (8,) \
            or not 1 <= len(reference) <= adapter.n_action_steps or not np.isfinite(reference).all():
        raise ValueError('Recorded executed prefix must be finite float32 [1..n_action_steps,8]')
    try:
        baseline = predict_paired_chunk(adapter, observation, pre_rng)
        if baseline.physical[:len(reference)].tobytes() != reference.tobytes():
            maximum = float(np.max(np.abs(baseline.physical[:len(reference)] - reference)))
            raise RuntimeError(f'Baseline does not match recorded executed actions bitwise (max abs {maximum})')
        if not rng_states_equal(torch, baseline.final_rng, post_rng):
            raise RuntimeError('Baseline final RNG does not match recorded post-prediction RNG')
        shape = (2, grid_size, grid_size, adapter.n_action_steps, 8)
        normalized = np.empty(shape, dtype=np.float32)
        physical = np.empty(shape, dtype=np.float32)
        count = 0
        total = 2 * grid_size**2
        for camera_index, camera in enumerate(cameras):
            for row in range(grid_size):
                for column in range(grid_size):
                    masked = mask_patch(observation, camera, row, column, grid_size,
                                        depth_camera_map=adapter.depth_camera_map)
                    prediction = predict_paired_chunk(adapter, masked, pre_rng)
                    if not rng_states_equal(torch, prediction.final_rng, post_rng):
                        raise RuntimeError(f'Patch {camera}/{row}/{column} changed the paired RNG stream')
                    normalized[camera_index, row, column] = prediction.normalized
                    physical[camera_index, row, column] = prediction.physical
                    count += 1
                    if progress is not None:
                        progress(count, total)
        if observation_fingerprint(observation) != source_hash:
            raise RuntimeError('Patch analysis mutated a source observation')
        fill_values = {}
        for camera in cameras:
            image, depth_key, depth = _rgb_depth(observation, camera, adapter.depth_camera_map)
            fill_values[camera] = {
                'rgb_uint8': np.rint(image.mean(axis=(0, 1), dtype=np.float64)).astype(np.uint8).tolist(),
                'depth_key': depth_key, 'depth_m': float(np.median(depth)),
            }
        return {
            **sensitivity_metrics(physical, baseline.physical),
            'physical_predictions': physical, 'normalized_predictions': normalized,
            'baseline_physical': baseline.physical, 'baseline_normalized': baseline.normalized,
            'checks': {'baseline_executed_prefix_bitwise_equal': True,
                       'reference_actions_checked': len(reference), 'all_rng_matched': True,
                       'rng_checks': count + 1, 'original_observation_unchanged': True,
                       'observation_sha256': source_hash},
            'protocol': {'cameras': list(cameras), 'grid_size': int(grid_size),
                         'variants': count, 'predicted_actions': adapter.n_action_steps,
                         'replacement': 'joint RGB-D: full-frame rounded RGB mean and depth median (including zeros)',
                         'fill_values': fill_values,
                         'randomness': 'same recorded CPU and all CUDA RNG states before every batch-one prediction',
                         'held_fixed': 'proprioceptive state, other RGB-D camera, unmasked pixels',
                         'metric': 'RMS difference across all predicted queries; seven arm joints in degrees, aperture in mm separately',
                         'interpretation': 'fixed-observation replacement sensitivity for one diffusion noise realization; no masked actions executed'},
        }
    finally:
        restore_rng_state(torch, outer_rng)
