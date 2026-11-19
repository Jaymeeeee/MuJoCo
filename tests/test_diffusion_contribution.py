"""Frozen RGB-D interventions pair the complete stochastic diffusion stream."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from mujoco_tools.diffusion_contribution import (
    capture_rng_state, compute_sensitivity, mask_patch, observation_fingerprint,
    predict_paired_chunk, restore_rng_state, rng_states_equal, sensitivity_metrics,
)


HEAD = 'observation.images.head'
WRIST = 'observation.images.wrist'
DEPTH_MAP = {HEAD: 'observation.depth.head', WRIST: 'observation.depth.wrist'}


def observation():
    image = np.arange(8 * 8 * 3, dtype=np.uint8).reshape(8, 8, 3)
    depth = np.arange(64, dtype=np.float32).reshape(8, 8) / 100 + .1
    return {HEAD: image, WRIST: image[::-1].copy(),
            DEPTH_MAP[HEAD]: depth, DEPTH_MAP[WRIST]: depth[::-1].copy() * 2,
            'observation.state': np.arange(8, dtype=np.float32) / 10}


@pytest.fixture
def adapter():
    torch = pytest.importorskip('torch')
    scheduler_type = pytest.importorskip('diffusers').DDPMScheduler

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scheduler = scheduler_type(num_train_timesteps=4, clip_sample=False)
            self._queues = {'observation.state': [], 'observation.images': [], 'action': []}
            self.calls = 0
            self.samples = []

        def reset(self):
            for queue in self._queues.values():
                queue.clear()

        def predict_action_chunk(self, batch):
            assert torch.is_inference_mode_enabled()
            assert batch['observation.state'].shape == (1, 1, 8)
            for camera, depth in DEPTH_MAP.items():
                assert batch[camera].shape == (1, 1, 3, 8, 8)
                assert batch[depth].shape == (1, 1, 1, 8, 8)
            self.calls += 1
            condition = batch['observation.state'][0, 0].clone()
            for index, (camera, depth) in enumerate(DEPTH_MAP.items()):
                condition += (index + 1) * (batch[camera].mean() + batch[depth].mean())
            sample = torch.randn((1, 64, 8))
            self.samples.append(sample.clone())
            self.scheduler.set_timesteps(4)
            for timestep in self.scheduler.timesteps:
                epsilon = .1 * sample + .03 * condition[None, None, :]
                sample = self.scheduler.step(epsilon, timestep, sample).prev_sample
            return sample[:, :32]

    features = {HEAD: {'shape': (3, 8, 8)}, WRIST: {'shape': (3, 8, 8)},
                DEPTH_MAP[HEAD]: {'shape': (1, 8, 8)}, DEPTH_MAP[WRIST]: {'shape': (1, 8, 8)},
                'observation.state': {'shape': (8,)}}

    def prepare(obs):
        out = {}
        for key, feature in features.items():
            value = np.array(obs[key], copy=True)
            if key in DEPTH_MAP:
                value = value.transpose(2, 0, 1).astype(np.float32) / 255
            elif key in DEPTH_MAP.values():
                if value.ndim == 2:
                    value = value[None]
                elif value.shape[-1] == 1:
                    value = value.transpose(2, 0, 1)
            out[key] = torch.from_numpy(value.astype(np.float32))
        return out

    def preprocess(batch):
        return {**{key: value.unsqueeze(0) for key, value in batch.items()}, 'action': None}

    postprocess_shapes = []

    def postprocess(value):
        postprocess_shapes.append(tuple(value.shape))
        assert value.shape == (1, 8)
        # Exercise ownership despite an in-place postprocessor.
        return value.mul_(torch.tensor([.5] * 7 + [.002])).add_(torch.tensor([0.] * 7 + [.02]))

    policy = Policy().eval()

    def reset():
        policy.reset()
        # Restoration must happen after reset, even if a processor draws noise.
        torch.rand(3)

    return SimpleNamespace(_torch=torch, policy=policy, policy_type='diffusion',
                           n_obs_steps=1, n_action_steps=32, action_shape=(8,),
                           input_features=features, depth_camera_map=DEPTH_MAP,
                           reset=reset, _prepare_observation=prepare,
                           preprocessor=preprocess, postprocessor=postprocess,
                           postprocess_shapes=postprocess_shapes)


def test_complete_ddpm_noise_pairing_and_batch_one_postprocessing(adapter):
    torch = adapter._torch
    torch.manual_seed(17)
    before = capture_rng_state(torch)
    obs = observation()
    old_hash = observation_fingerprint(obs)
    first = predict_paired_chunk(adapter, obs, before)
    # Arbitrary draws between calls and draws in reset must not affect pairing.
    torch.randn(117)
    second = predict_paired_chunk(adapter, obs, before)
    np.testing.assert_array_equal(first.normalized, second.normalized)
    np.testing.assert_array_equal(first.physical, second.physical)
    assert first.normalized.shape == first.physical.shape == (32, 8)
    assert first.physical.dtype == np.float32
    assert rng_states_equal(torch, first.final_rng, second.final_rng)
    assert not rng_states_equal(torch, before, first.final_rng)
    assert all(shape == (1, 8) for shape in adapter.postprocess_shapes)
    assert len(adapter.postprocess_shapes) == 64
    assert not np.shares_memory(first.normalized, first.physical)
    assert old_hash == observation_fingerprint(obs)
    expected = first.normalized * np.array([.5] * 7 + [.002], np.float32)
    expected += np.array([0.] * 7 + [.02], np.float32)
    np.testing.assert_array_equal(first.physical, expected)


def test_initial_noise_alone_does_not_pair_ddpm_scheduler_noise(adapter):
    torch = adapter._torch
    scheduler = adapter.policy.scheduler
    torch.manual_seed(23)
    initial = torch.randn(1, 64, 8)
    saved = capture_rng_state(torch)

    def denoise():
        sample = initial.clone()
        scheduler.set_timesteps(4)
        for timestep in scheduler.timesteps:
            sample = scheduler.step(sample * .1, timestep, sample).prev_sample
        return sample

    first = denoise()
    unpaired = denoise()
    restore_rng_state(torch, saved)
    paired = denoise()
    assert not torch.equal(first, unpaired)
    assert torch.equal(first, paired)


@pytest.mark.parametrize('layout', ['hw', 'hwc', 'chw'])
def test_joint_patch_fill_and_other_inputs_are_exactly_preserved(layout):
    obs = observation()
    if layout == 'hwc':
        obs[DEPTH_MAP[HEAD]] = obs[DEPTH_MAP[HEAD]][..., None]
    elif layout == 'chw':
        obs[DEPTH_MAP[HEAD]] = obs[DEPTH_MAP[HEAD]][None]
    original = deepcopy(obs)
    masked = mask_patch(obs, HEAD, 0, 1, 2, depth_camera_map=DEPTH_MAP)
    mask = np.zeros((8, 8), bool)
    mask[:4, 4:] = True
    mean = np.rint(obs[HEAD].mean((0, 1))).astype(np.uint8)
    np.testing.assert_array_equal(masked[HEAD][mask], np.broadcast_to(mean, (16, 3)))
    np.testing.assert_array_equal(masked[HEAD][~mask], obs[HEAD][~mask])
    assert np.all(masked[DEPTH_MAP[HEAD]].squeeze()[mask] == np.median(obs[DEPTH_MAP[HEAD]]))
    np.testing.assert_array_equal(masked[DEPTH_MAP[HEAD]].squeeze()[~mask],
                                  obs[DEPTH_MAP[HEAD]].squeeze()[~mask])
    for key in obs:
        np.testing.assert_array_equal(obs[key], original[key])
        if key not in (HEAD, DEPTH_MAP[HEAD]):
            np.testing.assert_array_equal(masked[key], original[key])
    assert masked[HEAD] is not obs[HEAD]
    assert masked[DEPTH_MAP[HEAD]] is not obs[DEPTH_MAP[HEAD]]


def test_scan_checks_actual_prefix_and_all_rng_states_and_restores_outer_rng(adapter):
    torch = adapter._torch
    obs = observation()
    torch.manual_seed(5)
    pre = capture_rng_state(torch)
    baseline = predict_paired_chunk(adapter, obs, pre)
    torch.rand(11)
    outer = capture_rng_state(torch)
    calls = []
    result = compute_sensitivity(adapter, obs, baseline.physical[:24], pre,
                                 baseline.final_rng, grid_size=2,
                                 progress=lambda completed, total: calls.append((completed, total)))
    assert rng_states_equal(torch, outer, capture_rng_state(torch))
    assert result['checks']['reference_actions_checked'] == 24
    assert result['checks']['baseline_executed_prefix_bitwise_equal']
    assert result['checks']['all_rng_matched']
    assert result['checks']['rng_checks'] == 9
    assert result['physical_predictions'].shape == (2, 2, 2, 32, 8)
    assert result['arm_rms_deg'].shape == result['gripper_rms_mm'].shape == (2, 2, 2)
    assert result['arm_rms_deg_per_query'].shape == (2, 2, 2, 32)
    assert result['gripper_abs_mm_per_query'].shape == (2, 2, 2, 32)
    assert np.all(result['arm_rms_deg'] > 0)
    assert result['protocol']['cameras'] == [HEAD, WRIST]
    assert calls == [(index, 8) for index in range(1, 9)]
    assert all(torch.equal(adapter.policy.samples[0], sample) for sample in adapter.policy.samples)
    np.testing.assert_array_equal(result['baseline_physical'], baseline.physical)
    for key, value in sensitivity_metrics(result['physical_predictions'], baseline.physical).items():
        np.testing.assert_array_equal(result[key], value)


def test_angle_and_aperture_metrics_use_separate_units_and_all_queries():
    baseline = np.zeros((2, 8), np.float32)
    prediction = np.zeros((2, 3, 3, 2, 8), np.float32)
    prediction[..., 0, :7] = np.deg2rad(2)
    prediction[..., 1, :7] = np.deg2rad(4)
    prediction[..., 0, 7] = .003
    prediction[..., 1, 7] = -.004
    result = sensitivity_metrics(prediction, baseline)
    np.testing.assert_allclose(result['arm_rms_deg'], np.sqrt(10), rtol=1e-7)
    np.testing.assert_allclose(result['gripper_rms_mm'], np.sqrt(12.5), rtol=1e-7)
    np.testing.assert_allclose(result['arm_rms_deg_per_query'][..., 0], 2, rtol=1e-7)
    np.testing.assert_allclose(result['gripper_abs_mm_per_query'][..., 1], 4, rtol=1e-7)


@pytest.mark.parametrize('failure', ['action', 'rng', 'variant_rng'])
def test_disagreement_fails_loudly_and_restores_outer_rng(adapter, failure):
    torch = adapter._torch
    pre = capture_rng_state(torch)
    baseline = predict_paired_chunk(adapter, observation(), pre)
    reference = baseline.physical.copy()
    post = baseline.final_rng
    if failure == 'action':
        reference[0, 0] += .01
        message = 'Baseline does not match'
    elif failure == 'rng':
        torch.rand(1)
        post = capture_rng_state(torch)
        message = 'Baseline final RNG'
    else:
        original = adapter.policy.predict_action_chunk
        calls = 0

        def changing_draws(batch):
            nonlocal calls
            calls += 1
            result = original(batch)
            if calls > 1:
                torch.rand(1)
            return result

        adapter.policy.predict_action_chunk = changing_draws
        message = 'changed the paired RNG stream'
    outer = capture_rng_state(torch)
    with pytest.raises(RuntimeError, match=message):
        compute_sensitivity(adapter, observation(), reference, pre, post, grid_size=2)
    assert rng_states_equal(torch, outer, capture_rng_state(torch))


def test_mutating_source_adapter_is_rejected(adapter):
    obs = observation()
    original = adapter._prepare_observation

    def mutate(source):
        source['observation.state'][0] += 1
        return original(source)

    adapter._prepare_observation = mutate
    with pytest.raises(RuntimeError, match='mutated a source observation'):
        predict_paired_chunk(adapter, obs, capture_rng_state(adapter._torch))


def test_baseline_comparison_is_bitwise_even_for_signed_zero(adapter):
    torch = adapter._torch
    adapter.policy.predict_action_chunk = lambda batch: torch.zeros((1, 32, 8))
    pre = capture_rng_state(torch)
    baseline = predict_paired_chunk(adapter, observation(), pre)
    reference = baseline.physical.copy()
    reference[0, 0] = -0.0
    assert np.array_equal(reference, baseline.physical)
    assert reference.tobytes() != baseline.physical.tobytes()
    with pytest.raises(RuntimeError, match='bitwise'):
        compute_sensitivity(adapter, observation(), reference, pre, baseline.final_rng, grid_size=2)


@pytest.mark.parametrize('change,message', [
    ({'n_obs_steps': 2}, 'n_obs_steps=1'),
    ({'policy_type': 'act'}, 'Diffusion adapter'),
    ({'action_shape': (7,)}, 'seven joint angles'),
])
def test_incompatible_adapter_rejected(adapter, change, message):
    adapter.__dict__.update(change)
    with pytest.raises(ValueError, match=message):
        predict_paired_chunk(adapter, observation(), capture_rng_state(adapter._torch))


@pytest.mark.parametrize('row,column,grid', [(0, 0, 3), (-1, 0, 2), (0, 2, 2), (0, 0, True)])
def test_bad_patch_geometry_rejected(row, column, grid):
    with pytest.raises(ValueError):
        mask_patch(observation(), HEAD, row, column, grid, depth_camera_map=DEPTH_MAP)


def test_missing_or_invalid_depth_rejected():
    obs = observation()
    with pytest.raises(ValueError, match='declared paired depth'):
        mask_patch(obs, HEAD, 0, 0, 2, depth_camera_map={})
    obs[DEPTH_MAP[HEAD]][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite and nonnegative'):
        mask_patch(obs, HEAD, 0, 0, 2, depth_camera_map=DEPTH_MAP)


def test_invalid_rng_state_rejected(adapter):
    torch = adapter._torch
    with pytest.raises(ValueError, match='exactly cpu and cuda'):
        restore_rng_state(torch, {'cpu': torch.get_rng_state()})
    with pytest.raises(ValueError, match='CPU uint8'):
        restore_rng_state(torch, {'cpu': torch.zeros(16), 'cuda': []})
