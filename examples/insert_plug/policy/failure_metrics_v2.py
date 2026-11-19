"""Offline, evidence-based failure-stage classification for v2 policy rollouts.

Arrays describe a directly recorded run or a separately validated replay. The four
requested labels describe observed stages; contact alone is not a causal proof
of jamming, and absence of a grasp alone does not prove an XY positioning error.
Physical extraction and the older strict 0.2 mm stable-grasp criterion are kept
separate. Thresholds and event intervals are returned for review and sensitivity
analysis; this module never loads a policy or changes a simulator.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


CATEGORIES = {
    'success': '任务成功',
    'grasp_miss': '未形成双侧夹持（空抓/抓取未命中）',
    'grasp_contact_no_lift': '形成双侧接触，但未完整提离源底座',
    'transport_target_miss': '提起后未完成目标底座对位',
    'insertion_socket_contact': '下插时碰到目标底座，未完成插入',
    'dropped_after_lift': '提起后在闭爪指令下失去接触并掉落',
    'inserted_not_completed': '曾落座，但释放/撤离或后续保持未完成',
    'released_without_insertion': '到达底座附近后释放，未观察到有效下插接触',
    'no_insertion_attempt': '已提起并到达底座附近，但未观察到下插尝试',
    'unresolved_insertion': '已到目标附近，但接触/下插证据不足以判为受阻',
    'knocked_out_without_grasp': '未形成双侧夹持而物体已离开源底座',
    'unresolved_lift': '物体已离开源底座，但持续携带证据不足',
    'unresolved': '现有轨迹证据不足以归入四类',
    'replay_unverified': '复现一致性未通过，不能归因原试验',
}
USER_CATEGORIES = {'grasp_miss': 1, 'grasp_contact_no_lift': 2,
                   'transport_target_miss': 3, 'insertion_socket_contact': 4}


@dataclass(frozen=True)
class FailureCriteria:
    sample_period_s: float = .001
    opposed_contact_s: float = .02
    strict_grasp_hold_s: float = .2
    strict_grasp_drift_m: float = .0002
    extraction_clearance_m: float = .005
    extraction_hold_s: float = .2
    target_contact_s: float = .02
    target_vertical_neighborhood_m: float = .05
    command_down_gap_m: float = .001
    source_upward_gap_m: float = .005
    stall_hold_s: float = .2
    stall_motion_m: float = .0005
    loss_contact_s: float = .1
    drop_distance_m: float = .005
    seating_hold_s: float = .2
    grasp_location_margin_m: float = .003
    active_normal_force_n: float = .001
    opening_lookback_s: float = .5
    opening_command_increase_m: float = .002
    opening_actual_increase_m: float = .0005
    outside_placement_margin_m: float = .0005
    outside_placement_depth_m: float = .002
    outside_placement_hold_s: float = .2
    descent_motion_window_s: float = .2
    descent_command_motion_m: float = .0002
    descent_actual_motion_m: float = .0001


def _read(arrays, name, shape, *, boolean=False, allow_nan=False):
    value = np.asarray(arrays[name])
    if value.shape != shape:
        raise ValueError(f'{name}: expected {shape}, got {value.shape}')
    if boolean:
        if value.dtype.kind != 'b':
            raise ValueError(f'{name} must contain booleans')
        return value
    value = value.astype(float, copy=False)
    if (np.isinf(value).any() or (not allow_nan and not np.isfinite(value).all())):
        raise ValueError(f'{name} has nonfinite values')
    return value


def _runs(mask):
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(np.int8)
    edges = np.diff(padded)
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1) - 1)


def _first_event(mask, time, hold=0.):
    for start, end in _runs(mask):
        if time[end] - time[start] >= hold - 1e-10:
            # ``end`` comes from NumPy. At an exact-duration boundary min()
            # can retain that np.int64, which JSON cannot serialize.
            confirmed = int(min(end, np.searchsorted(time, time[start] + hold - 1e-10)))
            return dict(start_sample=int(start), confirmed_sample=confirmed, end_sample=int(end),
                        start_time_s=float(time[start]), confirmed_time_s=float(time[confirmed]),
                        end_time_s=float(time[end]), duration_s=float(time[end] - time[start]),
                        samples=int(end-start+1))
    return None


def _max_duration(mask, time):
    return max((float(time[e]-time[s]) for s, e in _runs(mask)), default=0.)


def _window_all(mask, length):
    result = np.zeros(len(mask), dtype=bool)
    if len(mask) >= length:
        count = np.r_[0, np.cumsum(np.asarray(mask, dtype=int))]
        result[length-1:] = (count[length:] - count[:-length]) == length
    return result


def _window_span(value, length):
    """Full motion range, so a return to the same height is not called a stall."""
    result = np.full(len(value), np.inf)
    if len(value) >= length:
        windows = np.lib.stride_tricks.sliding_window_view(value, length)
        result[length-1:] = windows.max(axis=1) - windows.min(axis=1)
    return result


def _rolling_minimum(value, length):
    result = np.minimum.accumulate(value)
    if len(value) >= length:
        result[length-1:] = np.lib.stride_tricks.sliding_window_view(value, length).min(axis=1)
    return result


def _strict_stable_mask(time, tcp, tcp_rotation, tip, opposed, criteria):
    """Equivalent last-sample TCP-relative drift criterion, evaluated in chunks."""
    length = int(round(criteria.strict_grasp_hold_s / criteria.sample_period_s)) + 1
    candidates = np.flatnonzero(_window_all(opposed, length))
    stable = np.zeros(len(time), dtype=bool)
    offset = np.einsum('nji,nj->ni', tcp_rotation, tip-tcp)
    relative = np.arange(length-1, -1, -1)
    tolerance_squared = (criteria.strict_grasp_drift_m + 1e-12) ** 2
    for start in range(0, len(candidates), 1024):
        ids = candidates[start:start+1024]
        values = offset[ids[:, None] - relative] - offset[ids, None, :]
        maximum = np.max(np.sum(values*values, axis=2), axis=1)
        stable[ids] = maximum <= tolerance_squared
    return stable


def _minimum_body_z(body_xyz, rotation, vertices):
    result = np.empty(len(body_xyz), dtype=float)
    for start in range(0, len(result), 2048):
        stop = min(start+2048, len(result))
        result[start:stop] = body_xyz[start:stop, 2] + np.min(
            rotation[start:stop, 2, :] @ vertices.T, axis=1)
    return result


def _minimum(mask, value):
    selected = np.asarray(value)[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.min()) if len(selected) else None


def _maximum(mask, value):
    selected = np.asarray(value)[mask]
    selected = selected[np.isfinite(selected)]
    return float(selected.max()) if len(selected) else None


def classify_trace(arrays, config, *, task_success, replay_verified=True, criteria=None,
                   evidence_origin='reconstructed_original'):
    """Return one primary category plus separate measured stage/mechanism flags.

    By default, ``task_success`` refers to the original audited outcome. Set
    ``replay_verified`` only after checking that its passive reconstruction
    matches the original images/physical endpoint; otherwise this returns
    ``replay_unverified`` rather than assigning the original a cause.
    ``evidence_origin='recorded_run'`` instead classifies the directly recorded
    run itself, whose own outcome must be supplied as ``task_success``. That
    mode makes no claim of matching a historical run and does not require
    ``replay_verified``; its ``original_match_verified`` output is always None.
    Event sample indices and times refer to the 1 kHz physics arrays.
    """
    criteria = criteria or FailureCriteria()
    if not isinstance(criteria, FailureCriteria):
        raise TypeError('criteria must be FailureCriteria')
    if any(not np.isfinite(v) or v <= 0 for v in asdict(criteria).values()):
        raise ValueError('All failure criteria must be finite and positive')
    if type(task_success) is not bool or type(replay_verified) is not bool:
        raise TypeError('Outcome/verification flags must be booleans')
    if evidence_origin not in ('reconstructed_original', 'recorded_run'):
        raise ValueError('evidence_origin must be reconstructed_original or recorded_run')
    original_match_verified = replay_verified if evidence_origin == 'reconstructed_original' else None
    if evidence_origin == 'reconstructed_original' and not replay_verified:
        return dict(schema_version=1, category='replay_unverified', user_category=None,
                    label=CATEGORIES['replay_unverified'], classification_valid=False,
                    evidence_origin=evidence_origin, original_match_verified=False,
                    criteria=asdict(criteria))
    time = np.asarray(arrays['physics_time_s'], dtype=float)
    if time.ndim != 1 or len(time) < 2 or not np.isfinite(time).all():
        raise ValueError('Expected at least two finite physics times')
    n = len(time)
    if not np.allclose(np.diff(time), criteria.sample_period_s, atol=1e-9, rtol=0):
        raise ValueError('Physics samples must be contiguous at the declared sample period')
    tcp = _read(arrays, 'tcp_xyz', (n, 3))
    tcp_rotation = _read(arrays, 'tcp_rotation', (n, 3, 3))
    tip = _read(arrays, 'peg_tip_xyz', (n, 3))
    head = _read(arrays, 'peg_head_xyz', (n, 3))
    peg_rotation = _read(arrays, 'peg_rotation', (n, 3, 3))
    body = _read(arrays, 'peg_pose_xyzquat', (n, 7))[:, :3]
    contacts = _read(arrays, 'dual_contact', (n, 2), boolean=True)
    opposed, any_grip = contacts.all(axis=1), contacts.any(axis=1)
    finger_force = (_read(arrays, 'finger_peg_normal_force_n', (n, 2))
                    if 'finger_peg_normal_force_n' in arrays else None)
    active_opposed = (opposed & (finger_force >= criteria.active_normal_force_n).all(axis=1)
                      if finger_force is not None else np.zeros(n, dtype=bool))
    active_opposed_event = _first_event(active_opposed, time, criteria.opposed_contact_s)
    pad = _read(arrays, 'pad_xyz', (n, 2, 3)).mean(axis=1)
    aperture = _read(arrays, 'aperture_m', (n,))
    source = _read(arrays, 'source_rim_xyz', (3,))
    target = _read(arrays, 'target_rim_xyz', (3,))
    vertices = np.asarray(arrays['peg_collision_vertices'], dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError('Expected finite local peg collision vertices (V,3)')
    holder_bounds = _read(arrays, 'target_holder_local_bounds_xy', (2, 2))
    if not np.all(holder_bounds[1] > holder_bounds[0]):
        raise ValueError('Holder bounds must be rows [min XY, max XY]')
    action = np.asarray(arrays['action'], dtype=float)
    if action.ndim != 2 or action.shape[1] != 8 or not len(action) or not np.isfinite(action).all():
        raise ValueError('Expected finite executed actions (Tc,8)')
    command_tcp = _read(arrays, 'target_tcp_xyz', (len(action), 3))
    command_rotation = _read(arrays, 'target_tcp_rotation', (len(action), 3, 3))
    index = np.asarray(arrays['physics_action_index'])
    if index.shape != (n,) or index.dtype.kind not in 'iu' or np.any(index < -1) or np.any(index >= len(action)):
        raise ValueError('Invalid physics action indices')
    have_action = index >= 0
    safe_index = np.maximum(index, 0)
    commanded_tcp = command_tcp[safe_index]
    relative_tip = np.einsum('nji,nj->ni', tcp_rotation, tip-tcp)
    commanded_tip = commanded_tcp + np.einsum('nij,nj->ni', command_rotation[safe_index], relative_tip)
    command_width = action[safe_index, 7]
    peg_source = _read(arrays, 'peg_source_contacts', (n,)) > 0
    peg_target = _read(arrays, 'peg_target_contacts', (n,)) > 0
    gripper_source = _read(arrays, 'gripper_source_contacts', (n,)) > 0
    gripper_target = _read(arrays, 'gripper_target_contacts', (n,)) > 0
    force = {}
    for name in ('peg_source', 'peg_target', 'gripper_source', 'gripper_target'):
        key = name + '_normal_force_n'
        force[name] = _read(arrays, key, (n,)) if key in arrays else None
    def loaded(name, contact):
        return (contact & (force[name] >= criteria.active_normal_force_n)
                if force[name] is not None else np.zeros(n, dtype=bool))
    active_peg_source = loaded('peg_source', peg_source)
    active_gripper_source = loaded('gripper_source', gripper_source)
    active_peg_target = loaded('peg_target', peg_target)
    active_gripper_target = loaded('gripper_target', gripper_target)
    radial = _read(arrays, 'radial_error_m', (n,))
    mouth = _read(arrays, 'mouth_axis_error_m', (n,), allow_nan=True)
    seating = _read(arrays, 'seating_error_m', (n,))
    tilt = _read(arrays, 'tilt_rad', (n,))
    penetration = _read(arrays, 'max_peg_penetration_m', (n,))
    minimum_z = _minimum_body_z(body, peg_rotation, vertices)
    clearance = minimum_z - source[2]
    complete_clear = clearance >= criteria.extraction_clearance_m - 1e-12
    opposed_event = _first_event(opposed, time, criteria.opposed_contact_s)
    # A slipping peg's live TCP-relative position must not redefine the intended
    # held-object target on each tick. Retain the first observed grasp transform
    # only for the source-lift intent measure; downward tracking retains the live
    # transform because it tests the currently held peg's immediate command gap.
    retained_tip_command = None
    if opposed_event:
        retained_offset = relative_tip[opposed_event['confirmed_sample']]
        retained_tip_command = commanded_tcp + np.einsum('nij,j->ni', command_rotation[safe_index], retained_offset)
    raw_clear_event = _first_event(complete_clear, time, criteria.extraction_hold_s)
    contact_before = np.zeros(n, dtype=bool)
    if opposed_event:
        contact_before[opposed_event['confirmed_sample']:] = True
    lift_event = _first_event(complete_clear & contact_before, time, criteria.extraction_hold_s)
    lifted = lift_event is not None
    strict_mask = _strict_stable_mask(time, tcp, tcp_rotation, tip, opposed, criteria)
    strict_event = _first_event(strict_mask, time)
    if strict_event:
        confirmation = strict_event['start_sample']
        start = max(0, confirmation-int(round(criteria.strict_grasp_hold_s/criteria.sample_period_s)))
        strict_event.update(window_start_sample=start, window_start_time_s=float(time[start]),
                            confirmed_sample=confirmation, confirmed_time_s=float(time[confirmation]))
    after_lift = np.zeros(n, dtype=bool)
    if lift_event:
        after_lift[lift_event['start_sample']:] = True
    held_after_lift = after_lift & opposed
    held_clear_event = _first_event(complete_clear & opposed, time, criteria.opposed_contact_s)
    strict_before_extraction = bool(strict_event and lift_event and
                                   strict_event['confirmed_time_s'] <= lift_event['confirmed_time_s'])
    grip_supported_extraction = bool(lift_event and (held_clear_event or strict_before_extraction))
    extraction_opposed_fraction = None
    extraction_any_contact_fraction = None
    if lift_event:
        window = slice(lift_event['start_sample'], lift_event['confirmed_sample']+1)
        extraction_opposed_fraction = float(opposed[window].mean())
        extraction_any_contact_fraction = float(any_grip[window].mean())
    relative_target_xy = tip[:, :2] - target[:2]
    shaft_radius = float(config['peg']['shaft_radius'])
    in_holder_projection = ((relative_target_xy >= holder_bounds[0]-shaft_radius)
                            & (relative_target_xy <= holder_bounds[1]+shaft_radius)).all(axis=1)
    near_rim_height = np.abs(tip[:, 2]-target[2]) <= criteria.target_vertical_neighborhood_m
    target_region = held_after_lift & in_holder_projection & near_rim_height
    region_event = _first_event(target_region, time, criteria.opposed_contact_s)
    # An earlier pass high over the holder does not establish correct placement.
    # Use a conservative envelope of the entire holder (including its foot).
    # At the rim, a tilted circular shaft has an elliptical intersection whose
    # maximum radius is r/cos(theta); inflating by that radius is conservative.
    axis = peg_rotation[:, :, 2]
    upright_axis = axis[:, 2] > .1
    axis_at_rim_xy = np.full((n, 2), np.nan)
    axis_at_rim_xy[upright_axis] = (relative_target_xy[upright_axis]
        + axis[upright_axis, :2] * ((target[2]-tip[upright_axis, 2])/axis[upright_axis, 2])[:, None])
    axis_projection_radius = np.full(n, np.inf)
    axis_projection_radius[upright_axis] = shaft_radius/axis[upright_axis, 2]
    tip_projection_margin = np.maximum(holder_bounds[0]-shaft_radius-relative_target_xy,
                                        relative_target_xy-holder_bounds[1]-shaft_radius).max(axis=1)
    axis_projection_margin = np.maximum(holder_bounds[0]-axis_projection_radius[:, None]-axis_at_rim_xy,
                                        axis_at_rim_xy-holder_bounds[1]-axis_projection_radius[:, None]).max(axis=1)
    held_outside_placement = (after_lift & active_opposed & upright_axis
        & (tip[:, 2] <= target[2]-criteria.outside_placement_depth_m)
        & (tip_projection_margin >= criteria.outside_placement_margin_m)
        & (axis_projection_margin >= criteria.outside_placement_margin_m))
    outside_placement_event = _first_event(held_outside_placement, time, criteria.outside_placement_hold_s)
    hole_clearance = float(config['board']['hole_radius']) - shaft_radius
    if hole_clearance <= 0:
        raise ValueError('Expected positive radial hole clearance')
    precise = target_region & (radial < hole_clearance) & (mouth < hole_clearance) & (tilt < .10)
    precise_event = _first_event(precise, time, criteria.opposed_contact_s)
    command_down = have_action & (((tip[:, 2]-commanded_tip[:, 2]) >= criteria.command_down_gap_m)
                                 | ((tcp[:, 2]-commanded_tcp[:, 2]) >= criteria.command_down_gap_m))
    motion_lag = int(round(criteria.descent_motion_window_s/criteria.sample_period_s))
    command_descent = np.zeros(n)
    actual_descent = np.zeros(n)
    if n > motion_lag:
        command_descent[motion_lag:] = commanded_tcp[:-motion_lag, 2]-commanded_tcp[motion_lag:, 2]
        actual_descent[motion_lag:] = tcp[:-motion_lag, 2]-tcp[motion_lag:, 2]
    moving_down = (have_action & (command_descent >= criteria.descent_command_motion_m)
                   & (actual_descent >= criteria.descent_actual_motion_m))
    command_up = (have_action & ((retained_tip_command[:, 2]-tip[:, 2]) >= criteria.source_upward_gap_m)
                  if retained_tip_command is not None else np.zeros(n, dtype=bool))
    socket_contact = peg_target | gripper_target
    gap_down_contact = held_after_lift & command_down & socket_contact
    # Moving-target contact can establish a slow insertion attempt despite a
    # small tracking error. It must never relax the stronger blockage test.
    slow_down_contact = after_lift & active_opposed & moving_down & (active_peg_target | active_gripper_target)
    down_contact = gap_down_contact | slow_down_contact
    contact_event = _first_event(down_contact, time, criteria.target_contact_s)
    gap_contact_event = _first_event(gap_down_contact, time, criteria.target_contact_s)
    slow_contact_event = _first_event(slow_down_contact, time, criteria.target_contact_s)
    active_gap_contact = after_lift & active_opposed & command_down & (active_peg_target | active_gripper_target)
    active_down_peg = after_lift & active_opposed & (command_down | moving_down) & active_peg_target
    active_down_gripper = after_lift & active_opposed & (command_down | moving_down) & active_gripper_target
    active_down_contact = active_down_peg | active_down_gripper
    active_peg_target_event = _first_event(active_down_peg, time, criteria.target_contact_s)
    active_gripper_target_event = _first_event(active_down_gripper, time, criteria.target_contact_s)
    lag = int(round(criteria.stall_hold_s/criteria.sample_period_s))
    little_motion = _window_span(tip[:, 2], lag+1) <= criteria.stall_motion_m
    blocked = _window_all(active_gap_contact, lag+1) & little_motion
    blocked_event = _first_event(blocked, time)
    source_blocked = _window_all(active_opposed & command_up & (active_peg_source | active_gripper_source), lag+1) & little_motion
    source_blocked_event = _first_event(source_blocked, time)
    seated = (after_lift & (radial < hole_clearance) & (mouth < hole_clearance)
              & (seating < min(float(config['task']['seating_tolerance']), .0007))
              & (tilt < .10) & (penetration < .00015))
    seated_event = _first_event(seated, time, criteria.seating_hold_s)
    open_threshold = 2*float(config['peg']['head_radius']) + .004
    closure = have_action & (command_width <= open_threshold)
    close_actual = closure & (aperture <= open_threshold+.004)
    opening_window = int(round(criteria.opening_lookback_s / criteria.sample_period_s)) + 1
    opening_command_increase = command_width - _rolling_minimum(command_width, opening_window)
    opening_actual_increase = aperture - _rolling_minimum(aperture, opening_window)
    opening_intent = (have_action & (opening_command_increase >= criteria.opening_command_increase_m)
                      & (opening_actual_increase >= criteria.opening_actual_increase_m))
    loss_event = _first_event(after_lift & ~any_grip, time, criteria.loss_contact_s)
    slip_event = None
    opening_release_event = None
    # An early harmless contact loss may be followed by a genuine later drop.
    # Inspect every loss segment, while retaining the first generic loss above.
    for start, stop in _runs(after_lift & ~any_grip):
        if time[stop]-time[start] < criteria.loss_contact_s-1e-10:
            continue
        if opening_intent[start]:
            if opening_release_event is None:
                confirmed = min(stop,int(np.searchsorted(time,time[start]+criteria.loss_contact_s-1e-10)))
                opening_release_event = dict(start_sample=int(start), confirmed_sample=int(confirmed), end_sample=int(stop),
                    start_time_s=float(time[start]), confirmed_time_s=float(time[confirmed]),
                    end_time_s=float(time[stop]), duration_s=float(time[stop]-time[start]), samples=int(stop-start+1),
                    command_opening_increase_before_loss_m=float(opening_command_increase[start]),
                    actual_opening_increase_before_loss_m=float(opening_actual_increase[start]))
            continue
        if not closure[start]:
            continue
        if slip_event is not None:
            continue
        future = min(stop, int(np.searchsorted(time, time[start]+1.)), n-1)
        opened = np.flatnonzero(~closure[start:future+1] | opening_intent[start:future+1])
        if len(opened):
            future = start+int(opened[0])-1
        if future < start:
            continue
        drops = np.flatnonzero(minimum_z[start]-minimum_z[start:future+1] >= criteria.drop_distance_m)
        if not len(drops):
            continue
        drop_index = start+int(drops[0])
        confirmed = max(drop_index, min(stop,int(np.searchsorted(time,time[start]+criteria.loss_contact_s-1e-10))))
        slip_event = dict(start_sample=int(start), confirmed_sample=int(confirmed), end_sample=int(stop),
            start_time_s=float(time[start]), confirmed_time_s=float(time[confirmed]),
            end_time_s=float(time[stop]), duration_s=float(time[stop]-time[start]), samples=int(stop-start+1),
            drop_distance_m=float(minimum_z[start]-minimum_z[start:future+1].min()),
            drop_end_sample=int(drop_index), drop_end_time_s=float(time[drop_index]),
            command_remained_closed=True)
        # Retain the first slip while still checking later opening-loss events.
    pad_xy_error = np.linalg.norm(pad[:, :2]-head[:, :2], axis=1)
    pad_z_error = np.abs(pad[:, 2]-head[:, 2])
    grasp_height_band = pad_z_error <= float(config['peg']['head_height'])/2+.01
    attempts = close_actual & grasp_height_band
    if not attempts.any():
        attempts = close_actual
    closest_xy = _minimum(attempts, pad_xy_error)
    closest_z = _minimum(close_actual, pad_z_error)
    spatial_miss = bool(opposed_event is None and close_actual.any() and (
        (closest_xy is not None and closest_xy > float(config['peg']['head_radius'])+criteria.grasp_location_margin_m)
        or (closest_z is not None and closest_z > .03)))

    maximum_commanded_tip = (_maximum(contact_before & have_action, retained_tip_command[:, 2])
                             if retained_tip_command is not None else None)
    category, subreason, decision = 'unresolved', 'trajectory_does_not_match_a_supported_stage_rule', None
    if task_success:
        category, subreason = 'success', ('recorded_run_task_success' if evidence_origin == 'recorded_run'
                                         else 'original_and_replayed_task_success')
    elif not lifted:
        if raw_clear_event and opposed_event is None:
            category, subreason, decision = 'knocked_out_without_grasp', 'whole_object_cleared_without_sustained_opposed_contact', raw_clear_event
        elif opposed_event:
            category, decision = 'grasp_contact_no_lift', opposed_event
            if source_blocked_event:
                subreason, decision = 'upward_command_with_source_contact_and_vertical_stall', source_blocked_event
            elif maximum_commanded_tip is not None and maximum_commanded_tip < source[2]+criteria.extraction_clearance_m:
                subreason = 'no_commanded_tip_clearance_after_grasp'
            elif _first_event(contact_before & ~any_grip, time, criteria.loss_contact_s):
                subreason = 'lost_all_finger_contact_before_sustained_extraction'
            elif strict_event is None:
                subreason = 'opposed_contacts_without_strict_stable_grasp'
            else:
                subreason = 'stable_grasp_but_no_complete_extraction'
        else:
            category = 'grasp_miss'
            subreason = ('spatial_grasp_miss_confirmed' if spatial_miss else
                         'single_sided_touch_without_opposed_grasp' if any_grip.any() else
                         'no_closed_grasp_command' if not closure.any() else
                         'no_finger_contact_position_or_closure_unresolved')
            decision = _first_event(attempts, time) or _first_event(closure, time)
    elif not grip_supported_extraction:
        category, subreason, decision = 'unresolved_lift', 'whole_object_cleared_but_no_supported_carried_grasp', lift_event
    elif seated_event:
        category, subreason, decision = 'inserted_not_completed', 'geometrically_seated_but_final_success_not_completed', seated_event
    elif contact_event:
        category, decision = 'insertion_socket_contact', contact_event
        subreason = ('sustained_downward_command_and_socket_contact_with_stall' if blocked_event else
                     'downward_socket_contact_observed_stall_not_confirmed')
    elif slip_event:
        category, subreason, decision = 'dropped_after_lift', 'lost_contact_and_fell_while_command_remained_closed', slip_event
    elif outside_placement_event:
        category, subreason, decision = ('transport_target_miss',
            'actively_held_peg_lowered_outside_target_holder_projection', outside_placement_event)
    elif region_event is None:
        category, subreason, decision = 'transport_target_miss', 'no_held_peg_descent_neighborhood_at_target', lift_event
    elif opening_release_event:
        category, subreason, decision = 'released_without_insertion', 'opening_command_and_actual_widening_before_contact_loss_near_target', opening_release_event
    elif loss_event:
        category, subreason, decision = 'unresolved_insertion', 'contact_lost_without_clear_opening_or_closed_gripper_drop_evidence', loss_event
    elif not np.any(target_region & (command_down | moving_down)):
        category, subreason, decision = 'no_insertion_attempt', 'held_target_region_reached_without_downward_command_or_motion', region_event
    else:
        category, subreason, decision = 'unresolved_insertion', 'downward_intent_near_target_without_sustained_held_socket_contact', region_event

    events = dict(opposed_grasp_contact=opposed_event, strict_stable_grasp=strict_event,
                  active_opposed_grasp_load=active_opposed_event,
                  physical_clearance_without_grasp_requirement=raw_clear_event, actual_extraction=lift_event,
                  opposed_grasp_while_whole_object_clear=held_clear_event,
                  target_region_reached=region_event, precise_hole_alignment=precise_event,
                  held_lowered_outside_target_projection=outside_placement_event,
                  downward_socket_contact=contact_event, sustained_insertion_blockage=blocked_event,
                  downward_socket_contact_from_command_gap=gap_contact_event,
                  downward_socket_contact_from_target_motion=slow_contact_event,
                  active_downward_peg_target_contact=active_peg_target_event,
                  active_downward_gripper_target_contact=active_gripper_target_event,
                  upward_source_blockage=source_blocked_event, finger_contact_loss_after_lift=loss_event,
                  opening_release_after_lift=opening_release_event,
                  closed_gripper_drop_after_lift=slip_event, geometrically_seated=seated_event,
                  classification_event=decision)
    for event in (blocked_event, source_blocked_event):
        if event is not None:
            i = event['confirmed_sample']
            start = max(0, i-lag)
            event.update(window_start_sample=int(start), window_start_time_s=float(time[start]),
                         window_peg_vertical_range_m=float(np.ptp(tip[start:i+1, 2])))
    if outside_placement_event is not None:
        for key, i in (('start_geometry', outside_placement_event['start_sample']),
                       ('confirmed_geometry', outside_placement_event['confirmed_sample'])):
            outside_placement_event[key] = dict(tip_relative_target_xy_m=relative_target_xy[i].tolist(),
                shaft_axis_at_rim_relative_target_xy_m=axis_at_rim_xy[i].tolist(),
                tip_outside_expanded_projection_margin_m=float(tip_projection_margin[i]),
                axis_outside_ellipse_expanded_projection_margin_m=float(axis_projection_margin[i]),
                axis_projection_expansion_radius_m=float(axis_projection_radius[i]),
                tip_below_target_rim_m=float(target[2]-tip[i, 2]))
    if slow_contact_event is not None:
        for key, i in (('start_motion', slow_contact_event['start_sample']),
                       ('confirmed_motion', slow_contact_event['confirmed_sample'])):
            slow_contact_event[key] = dict(window_start_sample=int(max(0,i-motion_lag)),
                window_start_time_s=float(time[max(0,i-motion_lag)]),
                commanded_tcp_descent_m=float(command_descent[i]),actual_tcp_descent_m=float(actual_descent[i]))
    for event in events.values():
        if event is None:
            continue
        i = event['confirmed_sample']
        event['sample_values'] = dict(tcp_xyz=tcp[i].tolist(), peg_tip_xyz=tip[i].tolist(),
            whole_peg_clearance_above_source_m=float(clearance[i]), radial_error_m=float(radial[i]),
            mouth_axis_error_m=float(mouth[i]) if np.isfinite(mouth[i]) else None,
            tilt_rad=float(tilt[i]), aperture_m=float(aperture[i]), command_aperture_m=float(command_width[i]),
            commanded_tip_xyz=commanded_tip[i].tolist(), opposed_contacts=bool(opposed[i]),
            peg_source_contact=bool(peg_source[i]), peg_target_contact=bool(peg_target[i]),
            gripper_source_contact=bool(gripper_source[i]), gripper_target_contact=bool(gripper_target[i]))
        event['sample_values']['active_opposed_grasp_load'] = bool(active_opposed[i]) if finger_force is not None else None
        event['sample_values']['finger_peg_normal_force_n'] = finger_force[i].tolist() if finger_force is not None else None
        event['sample_values']['retained_first_grasp_tip_command_xyz'] = retained_tip_command[i].tolist() if retained_tip_command is not None else None
        for name in ('peg_source_normal_force_n','peg_target_normal_force_n',
                     'gripper_source_normal_force_n','gripper_target_normal_force_n'):
            if name in arrays:
                event['sample_values'][name] = float(np.asarray(arrays[name])[i])
    return dict(schema_version=1, category=category, user_category=USER_CATEGORIES.get(category),
                label=CATEGORIES[category], classification_valid=True, subreason=subreason,
                evidence_origin=evidence_origin, original_match_verified=original_match_verified,
                requires_manual_review=bool(category not in USER_CATEGORIES and category != 'success'
                    or category == 'grasp_miss' and not spatial_miss
                    or category == 'insertion_socket_contact' and (blocked_event is None
                        or slip_event is not None and slip_event['start_time_s'] > contact_event['start_time_s'])),
                flags=dict(actual_extraction=lifted, strict_stable_grasp=strict_event is not None,
                           grasp_supported_extraction=grip_supported_extraction,
                           opposed_grasp_contact=opposed_event is not None,
                           active_opposed_grasp_load=active_opposed_event is not None if finger_force is not None else None,
                           spatial_grasp_miss_confirmed=spatial_miss,
                           target_region_reached=region_event is not None,
                           held_lowered_outside_target_projection=outside_placement_event is not None,
                           within_hole_clearance=precise_event is not None,
                           downward_socket_contact=contact_event is not None,
                           downward_socket_contact_from_command_gap=gap_contact_event is not None,
                           downward_socket_contact_from_target_motion=slow_contact_event is not None,
                           active_downward_peg_target_contact=active_peg_target_event is not None if force['peg_target'] is not None else None,
                           active_downward_gripper_target_contact=active_gripper_target_event is not None if force['gripper_target'] is not None else None,
                           sustained_blocked=blocked_event is not None,
                           upward_source_stall=source_blocked_event is not None,
                           closed_gripper_drop=slip_event is not None,
                           opening_release_after_lift=opening_release_event is not None,
                           geometrically_seated=seated_event is not None),
                measurements=dict(samples=n, maximum_whole_peg_clearance_m=float(clearance.max()),
                    maximum_simultaneous_bilateral_normal_force_n=float(finger_force.min(axis=1).max()) if finger_force is not None else None,
                    maximum_continuous_active_opposed_load_s=_max_duration(active_opposed,time) if finger_force is not None else None,
                    maximum_continuous_active_downward_socket_contact_s=_max_duration(active_down_contact,time),
                    maximum_continuous_gap_downward_socket_contact_s=_max_duration(gap_down_contact,time),
                    maximum_continuous_motion_downward_socket_contact_s=_max_duration(slow_down_contact,time),
                    maximum_continuous_held_outside_target_projection_s=_max_duration(held_outside_placement,time),
                    extraction_window_opposed_contact_fraction=extraction_opposed_fraction,
                    extraction_window_any_contact_fraction=extraction_any_contact_fraction,
                    maximum_continuous_opposed_contact_s=_max_duration(opposed,time),
                    maximum_continuous_extraction_clearance_s=_max_duration(complete_clear,time),
                    maximum_continuous_downward_socket_contact_s=_max_duration(down_contact,time),
                    minimum_closed_grasp_pad_head_xy_error_m=closest_xy,
                    minimum_closed_grasp_pad_head_z_error_m=closest_z,
                    minimum_held_target_radial_error_m=_minimum(held_after_lift,radial),
                    maximum_commanded_tip_height_after_contact_m=_maximum(contact_before & have_action,commanded_tip[:,2])),
                events=events, criteria={**asdict(criteria), 'hole_clearance_m':hole_clearance,
                    'holder_bounds_local_xy_m':holder_bounds.tolist(),
                    'target_region_expansion_m':shaft_radius,
                    'release_command_threshold_m':open_threshold},
                interpretation_limits=[
                    'These are trajectory stage/contact observations, not counterfactual proof of a unique cause.',
                    'No opposed grasp does not by itself establish a spatial miss; inspect spatial_grasp_miss_confirmed and subreason.',
                    'Socket contact during downward intent does not by itself establish sustained jamming; inspect sustained_blocked.',
                    'Moving-target descent with load can establish slow insertion contact; sustained blockage still requires the separate command-gap criterion.',
                    'Actual extraction uses the lowest transformed collision vertex, not just peg-tip height or strict grasp drift.',
                ])
