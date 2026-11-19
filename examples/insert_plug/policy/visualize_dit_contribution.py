"""Measure RGB-D patch sensitivity along a saved OOD DiT attention rollout.

Only fixed-observation predictions are rerun. Their complete DDPM RNG streams
are paired with the real rollout, and none of the masked actions is executed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time

import numpy as np
from PIL import Image

from mujoco_tools.paths import WORKSPACE_ROOT, relocated_checkpoint_path
from .evaluate import write_json
from .output_naming import experiment_timestamp
from .visualize_dit_attention import _load_rng, _native
from .visualize_diffusion_attention import _fingerprint, _audit_video


def render_contribution(run, output, capture, records, scales):
    import imageio.v2 as imageio
    from .dit_attention_render import render_sensitivity_frame

    keys = capture['camera_keys']
    with np.load(run/'trace.npz', allow_pickle=False) as data:
        images = {key: data[f'obs__{key}'] for key in keys}
    (output/'previews').mkdir()
    previous = None
    with imageio.get_writer(str(output/'image_contribution.mp4'), fps=10, codec='libx264',
                            quality=8, macro_block_size=1, ffmpeg_log_level='error') as writer:
        for row in capture['frames']:
            index = row['chunk_index']
            if previous != index:
                with np.load(run/capture['chunks'][index]['arrays'], allow_pickle=False) as data:
                    source_images = {key: data[f'obs__{key}'].copy() for key in keys}
                with np.load(output/records[index]['arrays'], allow_pickle=False) as data:
                    arm = data['arm_rms_deg'].copy()
                    gripper = data['gripper_rms_mm'].copy()
                previous = index
            frame = render_sensitivity_frame(
                live_images={key: images[key][row['frame']] for key in keys},
                source_images=source_images,
                arm_maps={key: arm[i] for i, key in enumerate(keys)},
                gripper_maps={key: gripper[i] for i, key in enumerate(keys)},
                arm_vmax_deg=scales['arm_rms_deg'], gripper_vmax_mm=scales['gripper_rms_mm'],
                time_s=row['time_s'], source_time_s=row['source_time_s'], chunk_index=index,
                intervention_label='RGB 通道均值＋配对深度中位数遮挡；同一采样噪声')
            writer.append_data(frame)
            if row['action_offset'] == 0:
                Image.fromarray(frame).save(output/'previews'/f'chunk_{index:04d}.png')
            if row['frame'] == min(128, len(capture['frames'])-1):
                Image.fromarray(frame).save(output/'image_contribution_preview.png')
    return _audit_video(output/'image_contribution.mp4', len(capture['frames']))


def analyze(attention_run, *, grid_size=8, device='cuda'):
    import torch
    from mujoco_tools.diffusion_contribution import compute_sensitivity, sensitivity_metrics
    from mujoco_tools.lerobot_policy import LeRobotPolicy

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    run = Path(attention_run).expanduser().resolve()
    capture_path = run/'capture_report.json'
    capture = json.loads(capture_path.read_text())
    if not capture['validation']['valid'] or not capture['action_replay']['passed']:
        raise ValueError('Source rollout did not pass passive action/RNG validation')
    if capture['policy']['denoiser_type'] != 'dit' or capture['policy']['n_action_steps'] != 32:
        raise ValueError('Expected the saved DiT runtime with 32 action steps')
    if device != capture['policy']['device']:
        raise ValueError('Paired bitwise replay must use the original inference device')
    for path, fingerprint in capture['protected_files'].items():
        if _fingerprint(relocated_checkpoint_path(path)) != fingerprint:
            raise ValueError(f'Source/checkpoint has changed since capture: {path}')
    source_paths = [capture_path, run/'trace.npz', Path(__file__),
                   WORKSPACE_ROOT/'src/mujoco_tools/diffusion_contribution.py']
    for chunk in capture['chunks']:
        source_paths.extend((run/chunk['arrays'], run/chunk['rng']))
    protected = {str(path): _fingerprint(path) for path in source_paths}
    output = run/'contribution'
    output.mkdir(exist_ok=False)
    (output/'chunks').mkdir()
    (output/'analysis_source').mkdir()
    for path in source_paths[2:4]:
        shutil.copy2(path, output/'analysis_source'/path.name)
    write_json(output/'started.json', dict(started_at=experiment_timestamp(),
        attention_run=str(run), grid_size=grid_size, device=device, protected_files=protected))
    with np.load(run/'trace.npz', allow_pickle=False) as data:
        actions = data['action'].copy()
    adapter = LeRobotPolicy(capture['policy']['checkpoint'], device=device, n_action_steps=32)
    if tuple(adapter.policy.config.image_features) != tuple(capture['camera_keys']):
        raise ValueError('Camera ordering differs from the original capture')
    records = []
    start = time.monotonic()
    maxima = dict(arm_rms_deg=0., gripper_rms_mm=0.)
    for chunk in capture['chunks']:
        index, source_step = chunk['index'], chunk['source_step']
        with np.load(run/chunk['arrays'], allow_pickle=False) as data:
            observation = {key[5:]: data[key].copy() for key in data.files if key.startswith('obs__')}
        pre_rng, post_rng = _load_rng(run/chunk['rng'], torch)
        def progress(completed, total):
            if completed % 32 == 0:
                print(json.dumps(dict(event='patch_progress', chunk=index, completed=completed,
                    total=total, elapsed_s=round(time.monotonic()-start, 1))), flush=True)
        result = compute_sensitivity(adapter, observation, actions[source_step:source_step+32],
            pre_rng, post_rng, grid_size=grid_size, progress=progress)
        if result['protocol']['cameras'] != capture['camera_keys']:
            raise ValueError('Sensitivity camera ordering does not match display order')
        arrays = {key: value for key, value in result.items() if isinstance(value, np.ndarray)}
        if not all(np.isfinite(value).all() for value in arrays.values()):
            raise RuntimeError('Nonfinite sensitivity result')
        recomputed = sensitivity_metrics(arrays['physical_predictions'], arrays['baseline_physical'])
        if not all(np.array_equal(value, arrays[key]) for key, value in recomputed.items()):
            raise RuntimeError('Saved maps disagree with physical prediction deltas')
        filename = f'chunks/chunk_{index:04d}.npz'
        np.savez_compressed(output/filename, **arrays)
        record = dict(index=index, source_step=source_step, source_time_s=chunk['source_time_s'],
            arrays=filename, arrays_fingerprint=_fingerprint(output/filename),
            source_arrays=chunk['arrays'], source_rng=chunk['rng'],
            checks=result['checks'], protocol=result['protocol'])
        write_json(output/'chunks'/f'chunk_{index:04d}.json', _native(record))
        records.append(record)
        for key in maxima:
            maxima[key] = max(maxima[key], float(arrays[key].max()))
        print(json.dumps(dict(event='chunk_complete', chunk=index, total_chunks=len(capture['chunks']),
            baseline_actions_checked=result['checks']['reference_actions_checked'],
            elapsed_s=round(time.monotonic()-start, 1))), flush=True)
    # One absolute scale per physical quantity, shared by both cameras/all times.
    scales = {key: max(0.001, float(f'{value * 1.001:.4g}')) for key, value in maxima.items()}
    unchanged = all(_fingerprint(relocated_checkpoint_path(path)) == fingerprint for path, fingerprint in
                    {**capture['protected_files'], **protected}.items())
    if not unchanged:
        raise RuntimeError('A protected source or checkpoint changed during analysis')
    video = render_contribution(run, output, capture, records, scales)
    report = dict(started_at=json.loads((output/'started.json').read_text())['started_at'],
        completed_at=experiment_timestamp(), attention_run=str(run), output=str(output),
        source_capture=_fingerprint(capture_path), protected_files=protected,
        policy=capture['policy'], seed=capture['seed'], camera_keys=capture['camera_keys'],
        chunks=records, frames=len(capture['frames']), grid_size=grid_size,
        predictions=sum(record['protocol']['variants']+1 for record in records),
        variants=sum(record['protocol']['variants'] for record in records),
        executed_reference_actions_checked=sum(record['checks']['reference_actions_checked'] for record in records),
        actual_global_maxima=maxima, fixed_display_scales=scales,
        display_metric='RMS over all 32 forecast actions, not a single cached action; last 8 forecasts of final chunk were not executed',
        interpretation='Sensitivity to one specified RGB-D replacement at fixed observations and paired noise, not image attention, additive attribution, or evidence of improved task success',
        source_rollout_success=capture['rollout']['success'],
        validation=dict(valid=bool(unchanged and video['valid']), protected_files_unchanged=unchanged,
            all_baselines_bitwise_equal=True, all_full_rng_streams_paired=True,
            no_intervention_actions_executed=True),
        video=video, preview='image_contribution_preview.png', wall_seconds=time.monotonic()-start)
    write_json(output/'contribution_report.json', _native(report))
    print(json.dumps(dict(event='contribution_complete', output=str(output),
        valid=report['validation']['valid'], frames=report['frames'], predictions=report['predictions'])), flush=True)
    if not report['validation']['valid']:
        raise RuntimeError('Contribution video validation failed')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention-run', required=True)
    parser.add_argument('--grid-size', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    analyze(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
