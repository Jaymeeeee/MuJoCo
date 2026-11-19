"""Passively reconstruct all four v2 policy batches for failure-stage analysis.

The historical runs contain final metrics and videos, not physics trajectories.
This tool therefore creates NEW diagnostic runs and accepts a trajectory as a
reconstruction only when the original physical result and every video frame
match exactly. It does not classify failures or alter the historical evidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from datetime import datetime
import fcntl
import hashlib
from itertools import zip_longest
import json
import multiprocessing
from pathlib import Path
import traceback
from zoneinfo import ZoneInfo

import numpy as np

from mujoco_tools.paths import WORKSPACE_ROOT, checkpoint_dataset_path, relocated_checkpoint_path, relocated_evaluation_path
from . import evaluate, evaluate_regions_v2 as common, summarize as original_summary


GROUPS = ('act_ind', 'act_ood', 'dit_ind', 'dit_ood')
DEFAULT_REFERENCES = {
    'act_ind': 'outputs/test/act_insert_plug_v2_rgbd/ind/20261112_191102',
    'act_ood': 'outputs/test/act_insert_plug_v2_rgbd/ood/20261112_191102',
    'dit_ind': 'outputs/test/diffusion_dit_insert_plug_v2_rgbd/ind/20261112_202421',
    'dit_ood': 'outputs/test/diffusion_dit_insert_plug_v2_rgbd/ood/20261112_202421',
}
DEFAULT_AUDITS = {
    'act': 'outputs/_checks/act_v2_regions_20261112_190814/independent_audit.json',
    'dit': 'outputs/_checks/dit_v2_regions_20261112_201808/independent_audit.json',
}
_POLICIES = {}
PHYSICAL_RESULT_KEYS = (
    'success', 'task_success', 'contact_clean', 'stop_on_contact', 'contact_diagnostics',
    'contact_policy', 'contact_checks', 'stop_reason', 'runtime_error', 'requested_duration_s',
    'simulation_seconds', 'control_hz', 'action_hold_s', 'physics_timestep_s',
    'completed_control_steps', 'attempted_control_steps', 'selected_actions', 'physics_checks',
    'policy_input_keys', 'observation_modalities', 'depth_rendered', 'depth_input_keys',
    'depth_unit', 'depth_convention', 'depth_observation_frames', 'partial_final_control_interval',
    'final_metrics', 'continuous_success_seconds', 'success_required_seconds',
    'success_physics_samples', 'last_action', 'contacts', 'video_frames', 'video_layout', 'execution',
)
SCALAR_METRICS = (
    'radial_error_m', 'mouth_axis_error_m', 'seating_error_m', 'insertion_depth_m',
    'tilt_rad', 'linear_speed_m_s', 'angular_speed_rad_s', 'max_peg_penetration_m',
)
CONTACT_KINDS = ('peg_source', 'peg_target', 'gripper_source', 'gripper_target')


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def read_json(path):
    return original_summary.load_json(path)


def fingerprint(path):
    original_summary.regular_file(Path(path))
    return evaluate.fingerprint(Path(path))


def exact_result_comparison(original, reconstructed):
    """No numerical tolerance may silently convert a changed rollout to a match."""
    differences = []
    for key in PHYSICAL_RESULT_KEYS:
        if key not in original or key not in reconstructed or original[key] != reconstructed[key]:
            differences.append(dict(field=key, original=original.get(key), reconstructed=reconstructed.get(key)))
    return dict(equal=not differences, exact=True, checked_fields=list(PHYSICAL_RESULT_KEYS),
                differences=differences)


def compare_videos(original, reconstructed, expected_frames):
    """Prefer identical encoded bytes; otherwise compare all decoded RGB bytes."""
    import imageio_ffmpeg

    old_hash, new_hash = fingerprint(original), fingerprint(reconstructed)
    if old_hash == new_hash:
        return dict(equal=True, method='identical_complete_video_bytes', original=old_hash,
                    reconstructed=new_hash, frames=expected_frames, mismatched_frames=0,
                    max_absolute_pixel_difference=0, mean_absolute_pixel_difference=0.0,
                    first_mismatched_frame=None)
    readers = [imageio_ffmpeg.read_frames(str(path), pix_fmt='rgb24')
               for path in (original, reconstructed)]
    sentinel = object()
    count = mismatches = absolute_sum = pixel_count = maximum = 0
    first = None
    try:
        metadata = [next(reader) for reader in readers]
        if metadata[0]['size'] != metadata[1]['size'] or metadata[0]['fps'] != metadata[1]['fps']:
            return dict(equal=False, method='decoded_rgb_bytes', original=old_hash, reconstructed=new_hash,
                        error='Video dimensions or frame rate changed', metadata=metadata)
        for index, (left, right) in enumerate(zip_longest(*readers, fillvalue=sentinel)):
            count += 1
            if left is sentinel or right is sentinel:
                mismatches += 1
                if first is None:
                    first = index
                continue
            if left != right:
                mismatches += 1
                if first is None:
                    first = index
                a = np.frombuffer(left, dtype=np.uint8).astype(np.int16)
                b = np.frombuffer(right, dtype=np.uint8).astype(np.int16)
                if a.shape != b.shape:
                    raise ValueError('Decoded frame byte counts differ')
                difference = np.abs(a-b)
                maximum = max(maximum, int(difference.max()))
                absolute_sum += int(difference.sum())
            pixel_count += len(left)
    finally:
        for reader in readers:
            reader.close()
    return dict(equal=mismatches == 0 and count == expected_frames, method='decoded_rgb_bytes',
                original=old_hash, reconstructed=new_hash, frames=count, expected_frames=expected_frames,
                mismatched_frames=mismatches, max_absolute_pixel_difference=maximum,
                mean_absolute_pixel_difference=absolute_sum/pixel_count if pixel_count else None,
                first_mismatched_frame=first)


class TraceRecorder:
    """Record forwarded snapshots into shared buffers without touching live state."""

    def __init__(self, sim, duration):
        import mujoco

        self.size = int(round(duration / sim.model.opt.timestep)) + 1
        self.count = 0
        self.action_index = -1
        self.last_time = None
        self.action_times, self.actions, self.target_tcps, self.target_rotations = [], [], [], []
        self.fk_data = mujoco.MjData(sim.model)
        self.force = np.empty(6, dtype=np.float64)
        vector_shapes = dict(tcp_xyz=(3,), tcp_rotation=(3, 3), peg_tip_xyz=(3,), peg_head_xyz=(3,),
                             peg_rotation=(3, 3), peg_pose_xyzquat=(7,), pad_xyz=(2, 3),
                             qpos=(sim.model.nq,), qvel=(sim.model.nv,), finger_peg_normal_force_n=(2,))
        self.buffers = {name: np.empty((self.size, *shape), dtype=np.float64)
                        for name, shape in vector_shapes.items()}
        for name in ('physics_time_s', 'aperture_m', *SCALAR_METRICS,
                     *(kind+'_normal_force_n' for kind in CONTACT_KINDS)):
            self.buffers[name] = np.empty(self.size, dtype=np.float64)
        self.buffers['physics_action_index'] = np.empty(self.size, dtype=np.int32)
        self.buffers['dual_contact'] = np.empty((self.size, 2), dtype=np.bool_)
        self.buffers['finger_peg_contacts'] = np.empty((self.size, 2), dtype=np.int32)
        for kind in CONTACT_KINDS:
            self.buffers[kind+'_contacts'] = np.empty(self.size, dtype=np.int32)
        self.source_id = sim.model.body('source_socket').id
        self.target_id = sim.model.body('target_socket').id
        self.peg_id = sim.peg_id
        mount = sim.model.body('gripper_mount').id
        self.is_gripper_body = np.zeros(sim.model.nbody, dtype=bool)
        for body in range(sim.model.nbody):
            cursor = body
            while cursor:
                if cursor == mount:
                    self.is_gripper_body[body] = True
                    break
                cursor = int(sim.model.body_parentid[cursor])
        self.body_sides = sim.gripper._side_bodies.copy()
        self.body_ids = sim.model.geom_bodyid
        self.pad_ids = sim.gripper.pad_site_ids.copy()
        source = Path(sim.config['peg']['collision_manifest'])
        source = source if source.is_absolute() else WORKSPACE_ROOT/source
        collision = read_json(source)
        peg_vertices = np.concatenate([np.asarray(part['vertices'], dtype=float)
                                       for part in collision['bolt']['parts']])
        holder_vertices = np.concatenate([np.asarray(part['vertices'], dtype=float)
                                          for part in collision['holder']['parts']])
        self.constants = dict(source_rim_xyz=sim.site('source_hole'), target_rim_xyz=sim.site('target_hole'),
                              peg_collision_vertices=peg_vertices,
                              target_holder_local_bounds_xy=np.stack((holder_vertices[:, :2].min(axis=0),
                                                                      holder_vertices[:, :2].max(axis=0))))

    def before_action(self, sim, action):
        import mujoco

        action = np.asarray(action)
        self.action_index += 1
        self.action_times.append(float(sim.data.time))
        self.actions.append(action.copy())
        # Only this independent FK workspace is changed. TCP is a fixed mount
        # site, so arm qpos suffices; this never commands the live gripper.
        self.fk_data.qpos[:] = sim.data.qpos
        self.fk_data.qpos[sim.arm_qadr] = action[:7]
        mujoco.mj_kinematics(sim.model, self.fk_data)
        self.target_tcps.append(self.fk_data.site('tcp').xpos.copy())
        self.target_rotations.append(self.fk_data.site('tcp').xmat.copy().reshape(3, 3))

    def sample(self, sim, metrics):
        import mujoco

        stamp = float(sim.data.time)
        if self.last_time is not None:
            if stamp == self.last_time:
                return  # run_episode's final metrics repeat the final tick.
            if stamp < self.last_time:
                raise ValueError('Diagnostic sample time went backwards')
        if self.count >= self.size:
            raise ValueError('Diagnostic trace exceeded its declared duration')
        i = self.count
        self.last_time = stamp
        self.buffers['physics_time_s'][i] = stamp
        self.buffers['physics_action_index'][i] = self.action_index
        self.buffers['tcp_xyz'][i] = metrics['tcp_xyz']
        self.buffers['tcp_rotation'][i] = sim.data.site('tcp').xmat.reshape(3, 3)
        self.buffers['peg_tip_xyz'][i] = metrics['peg_tip_xyz']
        self.buffers['peg_head_xyz'][i] = sim.site('peg_head')
        self.buffers['peg_rotation'][i] = sim.data.xmat[self.peg_id].reshape(3, 3)
        self.buffers['peg_pose_xyzquat'][i, :3] = sim.data.xpos[self.peg_id]
        self.buffers['peg_pose_xyzquat'][i, 3:] = sim.data.xquat[self.peg_id]
        self.buffers['pad_xyz'][i] = sim.data.site_xpos[self.pad_ids]
        self.buffers['qpos'][i] = sim.data.qpos
        self.buffers['qvel'][i] = sim.data.qvel
        self.buffers['aperture_m'][i] = metrics['aperture_m']
        self.buffers['dual_contact'][i] = [side in metrics['gripper_contacts'] for side in ('left', 'right')]
        for name in SCALAR_METRICS:
            value = metrics[name]
            self.buffers[name][i] = np.nan if value is None else value
        counts = [0, 0, 0, 0]
        forces = [0., 0., 0., 0.]
        finger_counts, finger_forces = [0, 0], [0., 0.]
        for index, contact in enumerate(sim.data.contact):
            left, right = int(self.body_ids[contact.geom1]), int(self.body_ids[contact.geom2])
            kind = None
            other = None
            if left == self.peg_id or right == self.peg_id:
                other = right if left == self.peg_id else left
                if other == self.source_id:
                    kind = 0
                elif other == self.target_id:
                    kind = 1
            elif left in (self.source_id, self.target_id) or right in (self.source_id, self.target_id):
                fixture = left if left in (self.source_id, self.target_id) else right
                gripper = right if fixture == left else left
                if self.is_gripper_body[gripper]:
                    kind = 2 if fixture == self.source_id else 3
            side = self.body_sides.get(other)
            if kind is None and side is None:
                continue
            mujoco.mj_contactForce(sim.model, sim.data, index, self.force)
            normal = max(0., float(self.force[0]))
            if kind is not None:
                counts[kind] += 1
                forces[kind] += normal
            if side is not None:
                which = 0 if side == 'left' else 1
                finger_counts[which] += 1
                finger_forces[which] += normal
        for j, kind in enumerate(CONTACT_KINDS):
            self.buffers[kind+'_contacts'][i] = counts[j]
            self.buffers[kind+'_normal_force_n'][i] = forces[j]
        self.buffers['finger_peg_contacts'][i] = finger_counts
        self.buffers['finger_peg_normal_force_n'][i] = finger_forces
        self.count += 1

    def arrays(self):
        arrays = {name: array[:self.count] for name, array in self.buffers.items()}
        arrays.update(self.constants)
        arrays.update(action_time_s=np.asarray(self.action_times), action=np.asarray(self.actions),
                      target_tcp_xyz=np.asarray(self.target_tcps), target_tcp_rotation=np.asarray(self.target_rotations))
        return arrays


def simulation_class():
    """Load MuJoCo only in workers; observe exactly the existing success snapshot."""
    from ..environment.simulation import Simulation

    class DiagnosticSimulation(Simulation):
        def __init__(self, *args, diagnostic_duration=60., **kwargs):
            super().__init__(*args, **kwargs)
            self.trace = TraceRecorder(self, diagnostic_duration)
            self.metrics()  # Reset already forwarded this initial state.

        def step(self, action, *, before_substep=None):
            self.trace.before_action(self, action)
            return super().step(action, before_substep=before_substep)

        def metrics(self):
            metrics = super().metrics()
            self.trace.sample(self, metrics)
            return metrics

    return DiagnosticSimulation


def initialize_worker(backend):
    common._initialize_worker(backend)


def case_directory(configuration, job):
    return Path(configuration['output'])/'episodes'/job['group']/f"episode_{job['trial_index']:06d}"


def verify_provenance(configuration):
    for name, inventory in configuration['protected_files'].items():
        if fingerprint(relocated_evaluation_path(relocated_checkpoint_path(name))) != inventory:
            raise ValueError(f'Protected source/reference/checkpoint changed: {name}')


def prepare(args):
    started = now()
    output = evaluate.workspace_path(args.output or Path('outputs/insert_plug/reports') /
        f'act_diffusion_insert_plug_v2_rgbd_ind_ood_failure_analysis_{datetime.fromisoformat(started).strftime("%Y%m%d_%H%M%S")}')
    if output.exists() or output.is_symlink():
        raise FileExistsError(f'Diagnostic output exists; use --resume: {output}')
    protected, jobs, references, source_paths = {}, [], {}, {str(Path(__file__).resolve())}
    source_paths.add(str(WORKSPACE_ROOT/'src/mujoco_tools/robotiq.py'))
    prior_audits = {}
    for model in ('act', 'dit'):
        audit_path = evaluate.workspace_path(getattr(args, model+'_audit'))
        prior = read_json(audit_path)
        if prior.get('valid') is not True or prior.get('episodes') != 200:
            raise ValueError(f'Original {model} independent audit is not valid')
        protected[str(audit_path)] = fingerprint(audit_path)
        prior_audits[model] = prior
    for group in GROUPS:
        reference = evaluate.workspace_path(getattr(args, group))
        cfg = read_json(reference/'eval_config.json')
        summary = read_json(reference/'summary.json')
        audit = read_json(reference/'audit.json')
        if (not summary['complete'] or not summary['artifacts_validated'] or summary['episodes_terminal'] != 100
                or summary['runtime_failures'] or summary['infrastructure_failures'] or summary['artifact_errors']
                or audit['video_check'] != 'full' or audit['errors'] or len(audit['episodes']) != 100):
            raise ValueError(f'Reference is not a complete audited 100-episode batch: {group}')
        expected_type = 'act' if group.startswith('act_') else 'diffusion'
        split = group.rsplit('_', 1)[1]
        model_name = group.split('_', 1)[0]
        prior_group = prior_audits[model_name]['splits'][split]
        if relocated_evaluation_path(prior_group['output']) != reference or len(prior_group['reports']) != 100:
            raise ValueError('Reference differs from its completed independent audit')
        saved = cfg['checkpoint_config']
        if (cfg['evaluation_split'] != split or saved['type'] != expected_type or cfg['n_action_steps'] != 32
                or cfg['duration_s'] != 60. or cfg['backend'] != args.backend or cfg['device'] != args.device
                or (expected_type == 'diffusion' and saved.get('denoiser_type') != 'dit')):
            raise ValueError(f'Reference execution contract differs: {group}')
        recorded_root = evaluate.workspace_path(cfg['training_dataset']['root'])
        checkpoint = relocated_checkpoint_path(cfg['checkpoint'])
        reference_protected = [reference, checkpoint, recorded_root, checkpoint_dataset_path(recorded_root)]
        if any(output == old or output.is_relative_to(old) or old.is_relative_to(output) for old in reference_protected):
            raise ValueError('Diagnostic output overlaps historical evidence, checkpoint or dataset')
        for name in ('eval_config.json', 'summary.json', 'audit.json', 'episodes.csv',
                     'configs/scene_config.yaml', 'configs/positions_config.yaml', 'configs/training_manifest.yaml'):
            protected[str(reference/name)] = fingerprint(reference/name)
        for relative, expected in cfg['checkpoint_files'].items():
            path = checkpoint/relative
            if fingerprint(path) != expected:
                raise ValueError(f'Original checkpoint changed: {path}')
            protected[str(path)] = expected
        for relative, expected in cfg['runtime_source_files'].items():
            path = WORKSPACE_ROOT/relative
            if fingerprint(path) != expected:
                raise ValueError(f'Original runtime source changed: {path}')
            source_paths.add(str(path))
        for index, (row, layout) in enumerate(zip(summary['results'], cfg['positions'])):
            if row['episode_index'] != index or row['seed'] != index or row['source_xy'] != layout['source_xy'] or row['target_xy'] != layout['target_xy']:
                raise ValueError('Reference layout/seed identity mismatch')
            folder = reference/'episodes'/f'episode_{index:06d}'
            original = read_json(folder/'result.json')
            if original['success'] != row['success'] or original['trial_index'] != index or original['seed'] != index:
                raise ValueError('Reference per-trial identity mismatch')
            prior_row = prior_group['reports'][index]
            if prior_row['episode_index'] != index or prior_row['seed'] != index:
                raise ValueError('Original independent audit row identity mismatch')
            for name, prior_key in (('result.json', 'result'), ('scene.xml', 'scene'), ('rollout.mp4', 'video')):
                actual = fingerprint(folder/name)
                if actual != prior_row[prior_key]:
                    raise ValueError(f'Original episode changed after its completed audit: {folder/name}')
                protected[str(folder/name)] = actual
            for component in ('peg', 'board'):
                collision_path = evaluate.workspace_path(original['config'][component]['collision_manifest'])
                source_paths.add(str(collision_path))
            if relocated_checkpoint_path(original['policy']['checkpoint']) != checkpoint:
                raise ValueError('Reference report checkpoint mismatch')
            jobs.append(dict(group=group, trial_index=index, seed=index, reference=str(folder),
                             checkpoint=str(checkpoint), original_success=original['success']))
        if len(summary['results']) != 100 or len(cfg['positions']) != 100:
            raise ValueError('Reference must contain exactly 100 layouts')
        references[group] = str(reference)
    for path in source_paths:
        protected[path] = fingerprint(path)
    if len(jobs) != 400:
        raise ValueError('Expected all 400 reference episodes including successful controls')
    # Interleave groups at each shared trial index. Every worker keeps both
    # models in its own cache; no model reload occurs between later episodes.
    jobs.sort(key=lambda row: (row['trial_index'], GROUPS.index(row['group'])))
    configuration = dict(schema_version=1, protocol='v2_passive_failure_reconstruction', started_at=started,
        output=str(output), references=references, jobs=jobs, episodes_requested=400,
        workers=args.workers, backend=args.backend, device=args.device, n_action_steps=32, duration_s=60.,
        control_hz=10, physics_hz=1000, contact_diagnostics=False, protected_files=protected,
        trace_source='Read-only existing PhysicalSuccessMonitor snapshots; extra FK uses independent MjData',
        classification_included=False, match_requirement='Exact physical result and every RGB video frame',
        trace_semantics=dict(initial_sample=True, physics_sample_period_s=.001,
            contacts='Counts of MuJoCo snapshot contacts; force sums separately indicate active load',
            forces='Sum of nonnegative contact-frame normal forces in newtons',
            mouth_axis_error_m='NaN records the original None when the axis has no upright mouth intersection',
            peg_collision_vertices='Collision vertices in peg body coordinates; transform with peg_pose_xyzquat and peg_rotation',
            target_holder_local_bounds_xy='Rows are XY minimum and XY maximum in holder body coordinates'))
    output.mkdir(parents=True, exist_ok=False)
    (output/'episodes').mkdir()
    for group in GROUPS:
        (output/'episodes'/group).mkdir()
    evaluate.write_json(output/'diagnostic_config.json', configuration)
    return configuration


def execute_case(configuration, job):
    """Reconstruct one original episode; errors never become accepted evidence."""
    from mujoco_tools.lerobot_policy import LeRobotPolicy
    from .rollout import _seed_policy, run_episode

    folder = case_directory(configuration, job)
    folder.mkdir(exist_ok=False)
    evaluate.write_json(folder/'started.json', dict(started_at=now(), job=job))
    recorded_folder = Path(job['reference'])
    original_folder = relocated_evaluation_path(recorded_folder)
    try:
        for name in ('result.json', 'scene.xml', 'rollout.mp4'):
            path = original_folder/name
            if fingerprint(path) != configuration['protected_files'][str(recorded_folder/name)]:
                raise ValueError(f'Original evidence changed before reconstruction: {path}')
        original = read_json(original_folder/'result.json')
        checkpoint = job['checkpoint']
        if checkpoint not in _POLICIES:
            _POLICIES[checkpoint] = LeRobotPolicy(checkpoint, device=configuration['device'], n_action_steps=32)
        policy = _POLICIES[checkpoint]
        _seed_policy(job['seed'])  # Always after model construction; reset() preserves this RNG stream.
        sim = simulation_class()(original_folder/'scene.xml', original['config'], render=True,
                                 diagnostic_duration=configuration['duration_s'])
        try:
            result = run_episode(sim, policy, duration=configuration['duration_s'], video_path=folder/'rollout.mp4',
                                 viewer=False, realtime=False, stop_on_contact=False,
                                 contact_diagnostics=False, quiet=True)
            arrays = sim.trace.arrays()
            np.savez_compressed(folder/'trace.npz', **arrays)
        finally:
            sim.close()
        result.update(config=original['config'], policy=policy.metadata(), group=job['group'],
                      trial_index=job['trial_index'], seed=job['seed'], original_report=str(original_folder/'result.json'))
        evaluate.write_json(folder/'replay_result.json', result)
        physical = exact_result_comparison(original, result)
        video = compare_videos(original_folder/'rollout.mp4', folder/'rollout.mp4', original['video_frames'])
        trace_checks = dict(physics_samples=len(arrays['physics_time_s']),
            expected_physics_samples=result['physics_checks']+1,
            action_samples=len(arrays['action_time_s']), expected_action_samples=result['selected_actions'],
            initial_time=float(arrays['physics_time_s'][0]), final_time=float(arrays['physics_time_s'][-1]),
            uniform_physics_timestamps=bool(np.allclose(np.diff(arrays['physics_time_s']), .001, rtol=0, atol=1e-10)),
            last_action_matches=bool(np.array_equal(arrays['action'][-1], np.asarray(result['last_action']))))
        trace_valid = (trace_checks['physics_samples'] == trace_checks['expected_physics_samples']
                       and trace_checks['action_samples'] == trace_checks['expected_action_samples']
                       and trace_checks['initial_time'] == 0.0
                       and trace_checks['final_time'] == result['simulation_seconds']
                       and trace_checks['uniform_physics_timestamps'] and trace_checks['last_action_matches'])
        matched = bool(physical['equal'] and video['equal'] and trace_valid and result['runtime_error'] is None)
        comparison = dict(status='complete', completed_at=now(), group=job['group'], trial_index=job['trial_index'],
            seed=job['seed'], original_success=original['success'], replay_success=result['success'], matched=matched,
            original_report=str(original_folder/'result.json'), physical_comparison=physical,
            video_comparison=video, trace_checks=trace_checks, trace_valid=trace_valid,
            artifacts={name:fingerprint(folder/name) for name in ('trace.npz', 'replay_result.json', 'rollout.mp4')})
    except Exception:
        comparison = dict(status='reconstruction_error', completed_at=now(), group=job['group'],
                          trial_index=job['trial_index'], seed=job['seed'], matched=False, error=traceback.format_exc())
    evaluate.write_json(folder/'comparison.json', comparison)
    return {key:comparison.get(key) for key in ('group', 'trial_index', 'status', 'matched', 'original_success', 'replay_success', 'error')}


def summarize(configuration, *, audit=False):
    rows = []
    for job in configuration['jobs']:
        folder = case_directory(configuration, job)
        path = folder/'comparison.json'
        row = dict(group=job['group'], trial_index=job['trial_index'], original_success=job['original_success'])
        if path.exists():
            value = read_json(path)
            if value['group'] != job['group'] or value['trial_index'] != job['trial_index'] or value['seed'] != job['seed']:
                raise ValueError('Diagnostic result identity differs from prepared job')
            if audit and value['status'] == 'complete':
                for name, expected in value['artifacts'].items():
                    if fingerprint(folder/name) != expected:
                        raise ValueError(f'Diagnostic artifact changed after comparison: {folder/name}')
                original = read_json(relocated_evaluation_path(job['reference'])/'result.json')
                replay = read_json(folder/'replay_result.json')
                if exact_result_comparison(original, replay) != value['physical_comparison']:
                    raise ValueError('Stored physical comparison differs from actual reports')
                if (value['video_comparison']['original'] != configuration['protected_files'][str(Path(job['reference'])/'rollout.mp4')]
                        or value['video_comparison']['reconstructed'] != value['artifacts']['rollout.mp4']):
                    raise ValueError('Video comparison is not bound to actual artifact hashes')
                expected_match = bool(value['physical_comparison']['equal'] and value['video_comparison']['equal']
                                      and value['trace_valid'] and replay['runtime_error'] is None)
                if value['matched'] is not expected_match:
                    raise ValueError('Stored replay match disagrees with physical/video evidence')
            row.update(status=value['status'], matched=value['matched'], comparison=str(path),
                       replay_success=value.get('replay_success'), error=value.get('error'))
        else:
            row.update(status='running' if folder.exists() else 'pending', matched=False)
        rows.append(row)
    groups = {}
    for group in GROUPS:
        selected = [row for row in rows if row['group'] == group]
        counts = Counter(row['status'] for row in selected)
        terminal = sum(row['status'] not in ('pending', 'running') for row in selected)
        groups[group] = dict(episodes=100, terminal=terminal, matched=sum(row['matched'] for row in selected),
            unmatched=sum(row['status'] == 'complete' and not row['matched'] for row in selected),
            errors=sum(row['status'] not in ('complete', 'pending', 'running') for row in selected),
            status_counts=dict(counts))
    terminal, matched = sum(g['terminal'] for g in groups.values()), sum(g['matched'] for g in groups.values())
    summary = dict(updated_at=now(), output=configuration['output'], episodes_requested=400,
                   episodes_terminal=terminal, matched=matched, complete=terminal == 400,
                   valid=matched == 400 and terminal == 400, artifacts_audited=audit, groups=groups, results=rows)
    evaluate.write_json(Path(configuration['output'])/'reconstruction_summary.json', summary)
    return summary


def run(configuration):
    verify_provenance(configuration)
    jobs = []
    for job in configuration['jobs']:
        folder = case_directory(configuration, job)
        if not folder.exists():
            jobs.append(job)
        elif not (folder/'comparison.json').exists():
            evaluate.write_json(folder/'comparison.json', dict(status='interrupted', matched=False,
                group=job['group'], trial_index=job['trial_index'], seed=job['seed'],
                error='Previous diagnostic attempt ended without complete evidence; not silently retried'))
    summarize(configuration)
    if jobs:
        with ProcessPoolExecutor(max_workers=configuration['workers'], mp_context=multiprocessing.get_context('spawn'),
                                 initializer=initialize_worker, initargs=(configuration['backend'],)) as pool:
            futures = {pool.submit(execute_case, configuration, job):job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    event = future.result()
                except Exception:
                    folder = case_directory(configuration, job)
                    folder.mkdir(exist_ok=True)
                    event = dict(status='worker_error', group=job['group'], trial_index=job['trial_index'],
                                 seed=job['seed'], matched=False, error=traceback.format_exc())
                    evaluate.write_json(folder/'comparison.json', event)
                print(json.dumps(dict(event='diagnostic_complete', **event)), flush=True)
                summarize(configuration)
    verify_provenance(configuration)
    return summarize(configuration, audit=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for group, default in DEFAULT_REFERENCES.items():
        parser.add_argument('--'+group.replace('_', '-'), default=default)
    for model, default in DEFAULT_AUDITS.items():
        parser.add_argument('--'+model+'-audit', default=default)
    parser.add_argument('--output')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--backend', choices=('egl', 'osmesa'), default='egl')
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--resume', action='store_true')
    mode.add_argument('--audit-only', action='store_true')
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error('--workers must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.resume or args.audit_only:
        if not args.output:
            raise ValueError('--resume/--audit-only requires --output')
        configuration = read_json(evaluate.workspace_path(args.output)/'diagnostic_config.json')
        if configuration['output'] != str(evaluate.workspace_path(args.output)):
            raise ValueError('Diagnostic output changed')
        for key in ('workers', 'backend', 'device'):
            if configuration[key] != getattr(args, key):
                raise ValueError(f'Resume must retain {key}')
        verify_provenance(configuration)
    else:
        configuration = prepare(args)
    with ExitStack() as stack:
        lock = stack.enter_context((Path(configuration['output'])/'diagnostic.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.prepare_only:
            print(json.dumps(dict(output=configuration['output'], episodes_requested=400)))
            return 0
        result = summarize(configuration, audit=True) if args.audit_only else run(configuration)
        print(json.dumps({key:result[key] for key in ('complete', 'valid', 'episodes_terminal', 'matched', 'groups')}, indent=2))
        return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
