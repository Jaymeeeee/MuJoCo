"""Visualize one OOD DiT rollout and paired RGB-D region interventions.

Actual action self-attention and visual spatial pooling are kept distinct.
Region interventions rerun predictions at fixed observations and shared DDPM
noise; their actions are never executed in the simulator.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import json
import os
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

from mujoco_tools.paths import WORKSPACE_ROOT, OUTPUTS_DIR, relocated_checkpoint_path, relocated_evaluation_path
from .evaluate import write_json
from .output_naming import experiment_timestamp
from .visualize_diffusion_attention import (
    _fingerprint, _rng_snapshot, _audit_video, verify_actions,
)


DEFAULT_REFERENCE = WORKSPACE_ROOT / (
    'outputs/insert_plug/reports/act_diffusion_v2_new_failure_diagnostic_20261112_223400/'
    'episodes/dit_ood/episode_000000/replay_result.json')


def _native(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _native(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_native(item) for item in value]
    return value


def _save_rng(path, before, after):
    arrays = {}
    for prefix, state in (('before', before), ('after', after)):
        arrays[f'{prefix}_cpu'] = state['cpu'].cpu().numpy()
        for index, item in enumerate(state['cuda']):
            arrays[f'{prefix}_cuda_{index}'] = item.cpu().numpy()
    np.savez_compressed(path, **arrays)


def _load_rng(path, torch):
    with np.load(path, allow_pickle=False) as data:
        result = []
        for prefix in ('before', 'after'):
            keys = sorted((k for k in data.files if k.startswith(prefix+'_cuda_')),
                          key=lambda name: int(name.rsplit('_', 1)[1]))
            result.append(dict(cpu=torch.from_numpy(data[f'{prefix}_cpu'].copy()),
                               cuda=[torch.from_numpy(data[key].copy()) for key in keys]))
    return result


class Recorder:
    """Observe the existing adapter without replacing its action selection."""

    def __init__(self, adapter, spatial, dit, output, torch):
        self.adapter, self.spatial, self.dit = adapter, spatial, dit
        self.output, self.torch = output, torch
        self.observations, self.actions, self.frames, self.chunks = [], [], [], []
        self.spatial_count, self.dit_count = spatial.capture_count, dit.capture_count
        (output/'chunks').mkdir()
        (output/'rng').mkdir()

    def __getattr__(self, name):
        return getattr(self.adapter, name)

    def reset(self):
        if self.frames:
            raise RuntimeError('Recorder supports one rollout')
        self.adapter.reset()
        self.spatial.clear(); self.dit.clear()
        self.spatial_count, self.dit_count = self.spatial.capture_count, self.dit.capture_count

    def select_action(self, observation):
        copied = {key: np.array(value, copy=True) for key, value in observation.items()}
        step = len(self.frames)
        new_prediction = step % self.adapter.n_action_steps == 0
        before = _rng_snapshot(self.torch) if new_prediction else None
        action = self.adapter.select_action(observation)
        new_spatial = self.spatial.capture_count-self.spatial_count
        new_dit = self.dit.capture_count-self.dit_count
        if new_spatial != int(new_prediction) or new_dit != int(new_prediction):
            raise RuntimeError('Actual encoder/DiT invocation counts disagree with the action queue')
        if new_prediction:
            visual, attention = self.spatial.latest, self.dit.latest
            if visual is None or attention is None:
                raise RuntimeError('Missing completed attention capture')
            index = len(self.chunks)
            arrays = {f'obs__{key}': value for key, value in copied.items()}
            arrays.update(dit_weights=attention.weights, dit_mean_weights=attention.mean_weights,
                dit_timesteps=attention.timesteps, sampling_timesteps=attention.sampling_timesteps,
                attention_gates=attention.attention_gates,
                reference_weights_max_abs_error=attention.reference_weights_max_abs_error,
                live_output_max_abs_error=attention.live_output_max_abs_error,
                layer_indices=attention.layer_indices)
            for camera, key in enumerate(visual.camera_weights):
                arrays[f'spatial_weights_{camera}'] = visual.camera_weights[key]
                arrays[f'spatial_mean_{camera}'] = visual.mean_weights[key]
                arrays[f'keypoints_{camera}'] = visual.keypoints[key]
            weights_path = f'chunks/chunk_{index:04d}.npz'
            rng_path = f'rng/chunk_{index:04d}.npz'
            np.savez_compressed(self.output/weights_path, **arrays)
            _save_rng(self.output/rng_path, before, _rng_snapshot(self.torch))
            self.chunks.append(dict(index=index, source_step=step, source_time_s=step/10.,
                arrays=weights_path, rng=rng_path, camera_keys=list(visual.camera_weights),
                grid_shapes=visual.grid_shapes, keypoint_counts=visual.keypoint_counts,
                keypoint_reconstruction_max_abs_error=visual.keypoint_reconstruction_max_abs_error,
                denoising_timesteps=attention.timesteps.tolist(),
                all_sampling_timesteps=attention.sampling_timesteps.tolist(),
                num_heads=attention.num_heads, action_horizon=attention.action_horizon,
                reference_weights_max_abs_error=float(attention.reference_weights_max_abs_error.max()),
                live_output_max_abs_error=float(attention.live_output_max_abs_error.max())))
            self.spatial_count, self.dit_count = self.spatial.capture_count, self.dit.capture_count
        source = self.chunks[-1]
        self.observations.append(copied)
        self.actions.append(np.array(action, copy=True))
        self.frames.append(dict(frame=step, time_s=step/10., chunk_index=source['index'],
            source_step=source['source_step'], source_time_s=source['source_time_s'],
            action_offset=step-source['source_step']))
        return action

    def save(self):
        if not self.frames:
            raise RuntimeError('No actions were recorded')
        arrays = {f'obs__{key}': np.stack([obs[key] for obs in self.observations])
                  for key in self.observations[0]}
        arrays.update(action=np.stack(self.actions), time_s=np.array([row['time_s'] for row in self.frames]),
            source_step=np.array([row['source_step'] for row in self.frames]),
            action_offset=np.array([row['action_offset'] for row in self.frames]))
        np.savez_compressed(self.output/'trace.npz', **arrays)


def render_attention(output, report):
    import imageio.v2 as imageio
    from .dit_attention_render import render_frame

    (output/'previews').mkdir(exist_ok=True)
    with np.load(output/'trace.npz', allow_pickle=False) as trace:
        images = {key: trace[f'obs__{key}'] for key in report['camera_keys']}
    previous, current = None, None
    with imageio.get_writer(str(output/'attention.mp4'), fps=10, codec='libx264', quality=8,
                            macro_block_size=1, ffmpeg_log_level='error') as writer:
        for row in report['frames']:
            chunk = report['chunks'][row['chunk_index']]
            if previous != chunk['index']:
                with np.load(output/chunk['arrays'], allow_pickle=False) as data:
                    current = {key: data[key].copy() for key in data.files
                               if key.startswith(('spatial_', 'keypoints_', 'obs__observation.images.', 'dit_mean', 'dit_time'))}
                previous = chunk['index']
            keys = report['camera_keys']
            if current['dit_timesteps'][-1] != 0:
                raise ValueError('Display requires the final denoising timestep t=0')
            frame = render_frame(live_images={k: images[k][row['frame']] for k in keys},
                source_images={k: current[f'obs__{k}'] for k in keys},
                spatial_maps={k: current[f'spatial_mean_{i}'] for i,k in enumerate(keys)},
                keypoints={k: current[f'keypoints_{i}'] for i,k in enumerate(keys)},
                grid_shapes=chunk['grid_shapes'], action_attention=current['dit_mean_weights'][-1,-1],
                action_offset=row['action_offset'], time_s=row['time_s'], source_time_s=row['source_time_s'],
                chunk_index=row['chunk_index'], spatial_color_max=.05, action_color_max=.05)
            writer.append_data(frame)
            if row['action_offset'] == 0:
                Image.fromarray(frame).save(output/'previews'/f"chunk_{chunk['index']:04d}.png")
            if row['frame'] == min(128, len(report['frames'])-1):
                Image.fromarray(frame).save(output/'attention_preview.png')
    return _audit_video(output/'attention.mp4', len(report['frames']))


def capture(*, reference_report=DEFAULT_REFERENCE, output=None, duration=60., device='cuda'):
    import torch
    from mujoco_tools.dit_attention import DiTAttentionTrace, RGBDSpatialAttentionTrace
    from mujoco_tools.lerobot_policy import LeRobotPolicy
    from ..environment.simulation import Simulation
    from .rollout import _seed_policy, run_episode, validate_contract

    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    started = experiment_timestamp()
    reference_path = relocated_evaluation_path(reference_report).resolve()
    reference = json.loads(reference_path.read_text())
    scene_report = relocated_evaluation_path(reference.get('original_report', reference_path))
    scene_path = scene_report.parent/'scene.xml'
    checkpoint = relocated_checkpoint_path(reference['policy']['checkpoint'])
    saved = json.loads((checkpoint/'config.json').read_text())
    if saved.get('type') != 'diffusion' or saved.get('denoiser_type') != 'dit':
        raise ValueError('An explicitly configured Diffusion DiT checkpoint is required')
    if saved['n_obs_steps'] != 1 or saved['horizon'] != 64 or not saved.get('depth_camera_map'):
        raise ValueError('This visualization requires the v2 RGB-D n_obs1/horizon64 contract')
    cfg, seed = reference['config'], reference['seed']
    for key, centre in (('source_xy', (.45,-.07)), ('target_xy', (.45,.07))):
        if np.any(np.abs(np.asarray(cfg['board'][key])-centre) > .02):
            raise ValueError('Chosen layout is not inside the v2 central-square OOD region')
    output = Path(output).resolve() if output else OUTPUTS_DIR/'insert_plug/attention'/(
        'diffusion_dit_insert_plug_v2_rgbd_ood_'+datetime.fromisoformat(started).strftime('%Y%m%d_%H%M%S'))
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite an existing visualization: {output}')
    protected = {str(path): _fingerprint(path) for path in
        [reference_path, scene_path, *sorted(p for p in checkpoint.rglob('*') if p.is_file()),
         WORKSPACE_ROOT/'src/mujoco_tools/lerobot_policy.py',
         WORKSPACE_ROOT/'src/mujoco_tools/diffusion_attention.py',
         WORKSPACE_ROOT/'third_party/lerobot/src/lerobot/policies/diffusion/modeling_dit.py',
         WORKSPACE_ROOT/'third_party/lerobot/src/lerobot/policies/diffusion/modeling_diffusion.py',
         Path(__file__), WORKSPACE_ROOT/'src/mujoco_tools/dit_attention.py',
         WORKSPACE_ROOT/'examples/insert_plug/policy/dit_attention_render.py']}
    output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(scene_path, output/'scene.xml')
    write_json(output/'started.json', dict(started_at=started, output=str(output), reference=str(reference_path),
        source_xy=cfg['board']['source_xy'], target_xy=cfg['board']['target_xy'], seed=seed,
        protocol='new_ood_attention_visualization', protected_files=protected))
    adapter = LeRobotPolicy(checkpoint, device=device, n_action_steps=32)
    validate_contract(cfg, adapter)
    _seed_policy(seed)
    sim = Simulation(output/'scene.xml', cfg, render=True)
    initial_rng = _rng_snapshot(torch)
    try:
        with RGBDSpatialAttentionTrace(adapter.policy.diffusion) as spatial, DiTAttentionTrace(adapter.policy.diffusion) as dit:
            recorder = Recorder(adapter, spatial, dit, output, torch)
            result = run_episode(sim, recorder, duration=duration, video_path=output/'rollout.mp4',
                                 stop_on_contact=False, contact_diagnostics=False)
            final_rng = _rng_snapshot(torch)
            policy_metadata = deepcopy(adapter.metadata())
    finally:
        sim.close()
    if result['runtime_error'] is not None:
        write_json(output/'capture_error.json', result)
        raise RuntimeError(result['runtime_error'])
    recorder.save()
    verification, replay_rng = verify_actions(adapter, recorder.observations, recorder.actions, seed=seed,
        expected_initial_rng=initial_rng, expected_final_rng=final_rng, seed_function=_seed_policy, torch=torch)
    _save_rng(output/'rng/rollout.npz', initial_rng, final_rng)
    unchanged = all(_fingerprint(path) == value for path,value in protected.items())
    report = dict(started_at=started, output=str(output), reference_report=str(reference_path),
        reference_usage='Only the original OOD layout/checkpoint/seed; this is a new visualization rollout.',
        policy=policy_metadata, config=cfg, seed=seed, rollout=result,
        camera_keys=list(adapter.policy.config.image_features), observation_keys=list(adapter.input_features),
        depth_camera_map=adapter.depth_camera_map, chunks=recorder.chunks, frames=recorder.frames,
        protected_files=protected, action_replay=verification, protected_files_unchanged=unchanged,
        interpretation=dict(action_attention='Reconstructed real Q/K self-attention over 64 action-time tokens, not image patches.',
            conditioning='Two RGB-D ResNet18/SpatialSoftmax encoders plus state condition all DiT blocks through AdaLN.',
            spatial_maps='Visual RGB-D spatial pooling, not action-specific image attention or causal attribution.',
            timing='Source images update only on real replanning; cached actions retain their original source image.',
            displayed_attention='Last layer, final denoising timestep0, mean of8 heads; all6 layers/8 heads and selected steps saved.'),
        validation=dict(valid=bool(verification['passed'] and unchanged)),
        trace='trace.npz', video='attention.mp4', physical_video='rollout.mp4', preview='attention_preview.png')
    if not report['validation']['valid']:
        write_json(output/'capture_report.json', _native(report))
        raise RuntimeError('Passive action/RNG parity or source preservation validation failed')
    report['videos'] = [render_attention(output, report), _audit_video(output/'rollout.mp4', result['video_frames'])]
    report['validation']['valid'] &= all(item['valid'] for item in report['videos'])
    write_json(output/'capture_report.json', _native(report))
    print(json.dumps(dict(event='capture_complete',output=str(output),frames=len(recorder.frames),
                          chunks=len(recorder.chunks),valid=report['validation']['valid'])),flush=True)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-report', default=str(DEFAULT_REFERENCE))
    parser.add_argument('--output')
    parser.add_argument('--duration', type=float, default=60.)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--backend', choices=('egl','osmesa'), default='egl')
    args = vars(parser.parse_args(argv))
    os.environ['MUJOCO_GL'] = args.pop('backend')
    capture(**args)


if __name__ == '__main__':
    main()
