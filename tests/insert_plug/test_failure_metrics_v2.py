from copy import deepcopy
import json

import numpy as np
import pytest

from examples.insert_plug.policy.failure_metrics_v2 import FailureCriteria, classify_trace


def fixture():
    n = 2201
    time = np.arange(n) * .001
    count = 22
    source, target = np.array([.45,-.07,.04]), np.array([.45,.07,.04])
    tip = np.tile([.45,-.07,.01], (n,1))
    identity = np.repeat(np.eye(3)[None],n,axis=0)
    tcp = tip + [0,0,.03375]
    arrays = dict(physics_time_s=time,physics_action_index=np.r_[-1,(np.arange(n-1)//100)],
        tcp_xyz=tcp.copy(),tcp_rotation=identity.copy(),peg_tip_xyz=tip.copy(),
        peg_head_xyz=tip+[0,0,.03375],peg_rotation=identity.copy(),
        peg_pose_xyzquat=np.c_[tip,np.tile([1,0,0,0],(n,1))],
        aperture_m=np.full(n,.02),dual_contact=np.zeros((n,2),dtype=bool),
        finger_peg_normal_force_n=np.ones((n,2)),
        pad_xyz=np.stack([tcp+[-.01,0,0],tcp+[.01,0,0]],axis=1),
        source_rim_xyz=source,target_rim_xyz=target,
        peg_collision_vertices=np.array([[0,0,0],[0,0,.0375],[.012,0,.03],[-.012,0,.03]]),
        target_holder_local_bounds_xy=np.array([[-.0125,-.0125],[.0125,.0125]]),
        action_time_s=np.arange(count)*.1,
        action=np.c_[np.zeros((count,7)),np.full(count,.02)],
        target_tcp_xyz=np.tile(tcp[0],(count,1)),
        target_tcp_rotation=np.repeat(np.eye(3)[None],count,axis=0),
        peg_source_contacts=np.ones(n),peg_target_contacts=np.zeros(n),
        gripper_source_contacts=np.zeros(n),gripper_target_contacts=np.zeros(n),
        peg_source_normal_force_n=np.ones(n),peg_target_normal_force_n=np.ones(n),
        gripper_source_normal_force_n=np.zeros(n),gripper_target_normal_force_n=np.zeros(n),
        radial_error_m=np.full(n,.14),mouth_axis_error_m=np.full(n,.14),
        seating_error_m=np.zeros(n),tilt_rad=np.zeros(n),max_peg_penetration_m=np.zeros(n))
    config={'peg':{'shaft_radius':.005,'head_radius':.0138564,'head_height':.0075},
            'board':{'hole_radius':.0055},'task':{'seating_tolerance':.002}}
    return arrays,config


def sync_geometry(a):
    tip=a['peg_tip_xyz']; a['peg_pose_xyzquat'][:,:3]=tip
    a['peg_head_xyz']=tip+np.einsum('nij,j->ni',a['peg_rotation'],[0,0,.03375])
    a['tcp_xyz']=a['peg_head_xyz'].copy()
    a['pad_xyz']=np.stack([a['tcp_xyz']+[-.01,0,0],a['tcp_xyz']+[.01,0,0]],axis=1)
    a['radial_error_m']=np.linalg.norm(tip[:,:2]-a['target_rim_xyz'][:2],axis=1)
    a['mouth_axis_error_m']=a['radial_error_m'].copy()
    a['seating_error_m']=np.abs(tip[:,2]-.01)
    a['target_tcp_xyz']=a['tcp_xyz'][np.arange(len(a['action']))*100].copy()


def lifted():
    a,c=fixture(); a['dual_contact'][100:]=True
    a['peg_tip_xyz'][400:,2]=.06
    a['peg_source_contacts'][400:]=0
    sync_geometry(a)
    return a,c


def insertion():
    a,c=lifted()
    a['peg_tip_xyz'][1100:]=[.453,.07,.043]
    a['peg_target_contacts'][1100:]=1
    sync_geometry(a)
    a['target_tcp_xyz'][11:,2]=.01+.03375
    return a,c


def test_spatial_miss_has_actual_pad_head_error_evidence():
    a,c=fixture(); a['pad_xyz'][:,:,0]+=.05
    result=classify_trace(a,c,task_success=False)
    assert result['category']=='grasp_miss'
    assert result['user_category']==1
    assert result['flags']['spatial_grasp_miss_confirmed']
    assert result['measurements']['minimum_closed_grasp_pad_head_xy_error_m']==pytest.approx(.05)


def test_absence_of_grasp_does_not_automatically_mean_spatial_error():
    a,c=fixture()
    result=classify_trace(a,c,task_success=False)
    assert result['category']=='grasp_miss'
    assert not result['flags']['spatial_grasp_miss_confirmed']
    assert result['subreason']=='no_finger_contact_position_or_closure_unresolved'


def test_single_sided_contact_is_not_opposed_grasp():
    a,c=fixture(); a['dual_contact'][100:1000,0]=True
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='grasp_miss'
    assert r['subreason']=='single_sided_touch_without_opposed_grasp'
    assert not r['flags']['opposed_grasp_contact']


def test_source_upward_contact_stall_retains_measured_mechanism():
    a,c=fixture(); a['dual_contact'][100:]=True
    a['target_tcp_xyz'][:,2]+=.02
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='grasp_contact_no_lift'
    assert r['subreason']=='upward_command_with_source_contact_and_vertical_stall'
    assert r['flags']['strict_stable_grasp']
    assert not r['flags']['actual_extraction']
    assert r['events']['upward_source_blockage']['confirmed_time_s']>=.3


def test_actually_lifted_even_when_grasp_has_relative_slip():
    a,c=lifted()
    jitter=.002*np.sin(a['physics_time_s']*100)
    a['tcp_xyz'][:,0]+=jitter
    a['pad_xyz'][:,:,0]+=jitter[:,None]
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['actual_extraction']
    assert not r['flags']['strict_stable_grasp']
    assert r['category']=='transport_target_miss'
    assert r['events']['actual_extraction']['start_time_s']==pytest.approx(.4)


def test_whole_mesh_prevents_flipped_tip_only_false_lift():
    a,c=fixture(); a['dual_contact'][100:]=True
    a['peg_tip_xyz'][:,2]=.06
    a['peg_rotation'][:]=np.diag([1,-1,-1])
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert not r['flags']['actual_extraction']
    assert r['measurements']['maximum_whole_peg_clearance_m']==pytest.approx(-.0175)


def test_knocked_object_is_not_counted_as_grasp_and_lift():
    a,c=lifted(); a['dual_contact'][:]=False
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='knocked_out_without_grasp'
    assert r['user_category'] is None
    assert not r['flags']['actual_extraction']


def test_millimeter_offset_insertion_collision_does_not_require_precise_hole_fit():
    a,c=insertion()
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='insertion_socket_contact'
    assert r['flags']['target_region_reached']
    assert not r['flags']['within_hole_clearance']
    assert r['flags']['downward_socket_contact']
    assert r['flags']['sustained_blocked']
    assert r['events']['downward_socket_contact']['sample_values']['radial_error_m']==pytest.approx(.003)


def test_free_fall_contact_without_holding_is_not_insertion_blockage():
    a,c=insertion(); a['dual_contact'][1000:]=False
    a['peg_tip_xyz'][1000:1100,2]=np.linspace(.06,.043,100)
    sync_geometry(a); a['target_tcp_xyz'][10:,2]=.01+.03375
    r=classify_trace(a,c,task_success=False)
    assert not r['flags']['downward_socket_contact']
    assert r['category']=='dropped_after_lift'


def test_fleeting_contact_is_recorded_but_does_not_claim_sustained_insertion_failure():
    a,c=insertion(); a['peg_target_contacts'][1105:]=0
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='unresolved_insertion'
    assert not r['flags']['downward_socket_contact']
    assert r['measurements']['maximum_continuous_downward_socket_contact_s']==pytest.approx(.003)


def test_seated_but_unfinished_is_not_generic_socket_collision():
    a,c=insertion(); a['peg_tip_xyz'][1100:]=[.45,.07,.01]
    sync_geometry(a); a['target_tcp_xyz'][11:,2]-=.01
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='inserted_not_completed'
    assert r['user_category'] is None
    assert r['flags']['geometrically_seated']


def test_original_success_has_priority_over_contact_events():
    a,c=insertion()
    r=classify_trace(a,c,task_success=True)
    assert r['category']=='success'


def test_wrong_place_opened_release_differs_from_closed_gripper_drop():
    a,c=lifted(); a['dual_contact'][1000:]=False
    a['peg_tip_xyz'][1000:,2]=.01
    a['action'][9:,7]=.075
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert not r['flags']['closed_gripper_drop']
    assert r['category']=='transport_target_miss'


def test_unverified_replay_never_attributes_original_failure():
    r=classify_trace({}, {},task_success=False,replay_verified=False)
    assert r['category']=='replay_unverified' and not r['classification_valid']
    assert r['evidence_origin']=='reconstructed_original'
    assert r['original_match_verified'] is False


def test_directly_recorded_run_uses_its_own_outcome_without_claiming_historical_match():
    a,c=insertion()
    failure=classify_trace(a,c,task_success=False,replay_verified=False,evidence_origin='recorded_run')
    success=classify_trace(a,c,task_success=True,evidence_origin='recorded_run')
    assert failure['category']=='insertion_socket_contact' and failure['classification_valid']
    assert success['category']=='success' and success['classification_valid']
    assert success['subreason']=='recorded_run_task_success'
    assert failure['evidence_origin']==success['evidence_origin']=='recorded_run'
    assert failure['original_match_verified'] is None and success['original_match_verified'] is None


def test_unknown_evidence_origin_cannot_bypass_replay_verification():
    with pytest.raises(ValueError,match='evidence_origin'):
        classify_trace({}, {},task_success=False,replay_verified=False,evidence_origin='replay')


def test_contact_count_without_positive_load_is_not_confirmed_stall():
    a,c=insertion()
    a['peg_target_normal_force_n'][:]=0
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['downward_socket_contact']
    assert not r['flags']['active_downward_peg_target_contact']
    assert not r['flags']['sustained_blocked']
    assert r['requires_manual_review']


def test_source_lift_command_preserves_grasp_transform_during_relative_slip():
    a,c=fixture();a['dual_contact'][100:1000]=True
    a['tcp_xyz'][100:1100,2]+=np.linspace(0,.03,1000)
    a['target_tcp_xyz'][:,2]=a['tcp_xyz'][np.arange(len(a['action']))*100,2]
    a['target_tcp_xyz'][10:,2]=a['tcp_xyz'][100,2]+.06
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='grasp_contact_no_lift'
    assert r['flags']['upward_source_stall']
    assert r['subreason']!='no_commanded_tip_clearance_after_grasp'


def test_returning_to_same_height_after_oscillation_does_not_confirm_stall():
    a,c=insertion()
    a['peg_tip_xyz'][1100:,2]+=.005*np.sin(np.arange(len(a['physics_time_s'])-1100)*2*np.pi/200)
    sync_geometry(a);a['target_tcp_xyz'][11:,2]=.01+.03375
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['downward_socket_contact']
    assert not r['flags']['sustained_blocked']


def test_gradual_active_opening_is_not_a_closed_gripper_slip():
    a,c=insertion();a['peg_target_contacts'][:]=0
    a['dual_contact'][1201:]=False
    a['action'][12:,7]=.024  # below broad 31.7mm closure threshold, but actively opening
    a['aperture_m'][1200:]=.022
    a['peg_tip_xyz'][1200:1401,2]=np.linspace(.043,.01,201)
    a['peg_tip_xyz'][1401:,2]=.01
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['opening_release_after_lift']
    assert not r['flags']['closed_gripper_drop']
    assert r['category']=='released_without_insertion'
    e=r['events']['opening_release_after_lift']
    assert e['command_opening_increase_before_loss_m']==pytest.approx(.004)
    assert e['actual_opening_increase_before_loss_m']==pytest.approx(.002)


def outside_placement():
    a,c=lifted()
    a['peg_tip_xyz'][700:1100]=[.45,.07,.08]  # earlier high pass over target
    a['peg_tip_xyz'][1100:]=[.48,.07,.035]
    sync_geometry(a)
    return a,c


def test_held_lowering_outside_holder_is_position_failure_despite_prior_high_pass():
    a,c=outside_placement()
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['target_region_reached']
    assert r['flags']['held_lowered_outside_target_projection']
    assert r['category']=='transport_target_miss'
    assert r['subreason']=='actively_held_peg_lowered_outside_target_holder_projection'
    assert r['events']['held_lowered_outside_target_projection']['duration_s']>=.2


@pytest.mark.parametrize('reason',['no_load','brief','axis_ellipse_overlaps'])
def test_outside_placement_requires_loaded_duration_and_conservative_shaft_geometry(reason):
    a,c=outside_placement()
    if reason=='no_load':a['finger_peg_normal_force_n'][1100:]=0
    elif reason=='brief':
        a['peg_tip_xyz'][1200:]=[.45,.07,.08];sync_geometry(a)
    else:
        # At the rim: shaft axis x=18.25mm, beyond radius-r expansion (17.5mm)
        # but within the conservative ellipse expansion r/cos(theta)=6.25mm.
        a['peg_tip_xyz'][1100:]=[.4705,.07,.037]
        a['peg_rotation'][1100:]=[[.8,0,-.6],[0,1,0],[.6,0,.8]]
        sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert not r['flags']['held_lowered_outside_target_projection']


def test_slow_loaded_descent_is_contact_but_never_weakens_strong_blockage_gap():
    a,c=lifted()
    a['peg_tip_xyz'][1100:]=[.453,.07,.043]
    a['peg_tip_xyz'][1100:,2]-=np.arange(len(a['physics_time_s'])-1100)*.001*.0012
    a['peg_target_contacts'][1100:]=1
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert r['category']=='insertion_socket_contact'
    assert r['flags']['downward_socket_contact_from_target_motion']
    assert not r['flags']['downward_socket_contact_from_command_gap']
    assert not r['flags']['sustained_blocked']
    assert r['requires_manual_review']


def test_static_loaded_contact_is_not_new_slow_descent_evidence():
    a,c=lifted()
    a['peg_tip_xyz'][1100:]=[.453,.07,.043]
    a['peg_target_contacts'][1800:]=1  # loaded only after all movement is over
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert not r['flags']['downward_socket_contact_from_target_motion']
    assert not r['flags']['downward_socket_contact']
    assert not r['flags']['sustained_blocked']


@pytest.mark.parametrize('kind', ['opposed_contact', 'contact_loss', 'slow_contact'])
def test_exact_duration_events_have_json_native_indices(kind):
    if kind == 'opposed_contact':
        a,c=fixture(); a['dual_contact'][100:121]=True
        key='opposed_grasp_contact'
    elif kind == 'contact_loss':
        a,c=lifted(); a['dual_contact'][1500:1601]=False
        key='finger_contact_loss_after_lift'
    else:
        a,c=lifted()
        a['peg_tip_xyz'][1100:]=[.453,.07,.043]
        a['peg_tip_xyz'][1100:,2]-=np.arange(len(a['physics_time_s'])-1100)*.001*.0012
        a['peg_target_contacts'][1800:1821]=1
        sync_geometry(a)
        key='downward_socket_contact_from_target_motion'
    result=classify_trace(a,c,task_success=False,evidence_origin='recorded_run')
    event=result['events'][key]
    assert event is not None and event['confirmed_sample']==event['end_sample']
    assert type(event['confirmed_sample']) is int
    if kind == 'slow_contact':
        assert type(event['confirmed_motion']['window_start_sample']) is int
    assert json.loads(json.dumps(result,allow_nan=False))==result


@pytest.mark.parametrize('later',['socket_contact','closed_drop'])
def test_existing_contact_and_closed_drop_priorities_survive_outside_placement_event(later):
    a,c=outside_placement()
    if later=='socket_contact':
        a['peg_tip_xyz'][1600:]=[.453,.07,.043]
        a['peg_target_contacts'][1600:]=1
        sync_geometry(a);a['target_tcp_xyz'][16:,2]-=.01
    else:
        a['dual_contact'][1500:]=False
        a['peg_tip_xyz'][1500:1701,2]=np.linspace(.035,.01,201)
        a['peg_tip_xyz'][1701:,2]=.01
        sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['held_lowered_outside_target_projection']
    assert r['category']==('insertion_socket_contact' if later=='socket_contact' else 'dropped_after_lift')


@pytest.mark.parametrize('kind',['gap','backwards','nan'])
def test_malformed_trace_times_are_rejected(kind):
    a,c=fixture()
    if kind=='gap': a['physics_time_s'][10:]+=.001
    elif kind=='backwards': a['physics_time_s'][10]=a['physics_time_s'][9]
    else: a['physics_time_s'][10]=np.nan
    with pytest.raises(ValueError): classify_trace(a,c,task_success=False)


def test_inputs_and_configuration_are_unchanged_by_classification():
    a,c=insertion(); old={k:v.copy() for k,v in a.items()}; oldc=deepcopy(c)
    classify_trace(a,c,task_success=False)
    for k,v in old.items(): np.testing.assert_array_equal(a[k],v)
    assert c==oldc


def test_later_drop_detected_after_earlier_harmless_contact_gap():
    a,c=lifted()
    a['dual_contact'][700:850]=False  # retains its height and is grasped again
    a['dual_contact'][1400:]=False
    a['peg_tip_xyz'][1400:1601,2]=np.linspace(.06,.01,201)
    a['peg_tip_xyz'][1601:,2]=.01
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert r['events']['finger_contact_loss_after_lift']['start_time_s']==pytest.approx(.7)
    assert r['events']['closed_gripper_drop_after_lift']['start_time_s']==pytest.approx(1.4)
    assert r['category']=='dropped_after_lift'


def test_brief_opposed_hit_then_airborne_without_carried_grasp_is_not_assigned_transport():
    a,c=fixture()
    a['dual_contact'][100:131]=True
    a['peg_tip_xyz'][400:,2]=.06
    sync_geometry(a)
    r=classify_trace(a,c,task_success=False)
    assert r['flags']['actual_extraction']
    assert not r['flags']['grasp_supported_extraction']
    assert r['category']=='unresolved_lift'
    assert r['requires_manual_review']
