"""Physical checks for the passive torque proxy and causal estimator."""

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from mujoco_tools.torque_signals import TorqueSignalSampler, estimate_wrench, transform_wrench


def _sim(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    jids = np.array([model.joint('joint').id])
    return SimpleNamespace(model=model, data=data, arm_jids=jids,
                           arm_qadr=model.jnt_qposadr[jids], arm_dadr=model.jnt_dofadr[jids],
                           arm_aids=np.array([0]), tcp_id=model.site('tcp').id,
                           peg_id=-1, gripper=None)


def _pendulum(*, contact=False, gravity='0 0 -9.81', gravcomp=1, armature=.5, friction=.3):
    plane = '<geom type="plane" size="2 2 .1"/>' if contact else ''
    shape = ('<geom type="sphere" pos=".5 0 .09" size=".1" mass="1"/>' if contact else
             '<geom type="capsule" fromto="0 0 0 .5 0 0" size=".02" mass="1" contype="0" conaffinity="0"/>')
    return _sim(f'''<mujoco><option timestep=".001" gravity="{gravity}" solver="Newton" tolerance="1e-12"/>
      <worldbody>{plane}<body name="link" gravcomp="{gravcomp}">
      <joint name="joint" axis="0 1 0" damping=".2" frictionloss="{friction}" armature="{armature}"/>
      {shape}<site name="tcp" pos=".5 0 .09" euler="0 0 90"/>
      </body></worldbody><actuator><position joint="joint" kp="100" kv="10" forcerange="-8 8"/>
      </actuator></mujoco>''')


def _integration_state(sim):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    value = np.empty(mujoco.mj_stateSize(sim.model, spec))
    mujoco.mj_getState(sim.model, sim.data, value, spec)
    return value


@pytest.mark.parametrize('gravcomp', [0, 1])
def test_free_motion_rotor_friction_and_gravity_are_not_external_contact(gravcomp):
    sim = _pendulum(gravcomp=gravcomp)
    sim.data.qpos[:] = .2
    sim.data.qvel[:] = .3
    sim.data.ctrl[:] = .3
    sample = TorqueSignalSampler(sim).sample()
    np.testing.assert_allclose(sample['joint_external_torque_raw'], 0, atol=2e-12)
    assert abs(sample['joint_rotor_inertia_torque'][0]) > .1
    assert abs(sample['joint_internal_friction_torque'][0]) > .1
    assert not np.allclose(sample['joint_torque_measured_sim'], sample['joint_actuator_torque'])
    # Independent spatial Newton-Euler interface wrench projection.  It has no
    # rotor inertia and is not the scalar actuator output.
    scratch = mujoco.MjData(sim.model)
    mujoco.mj_copyData(scratch, sim.model, sim.data)
    mujoco.mj_forward(sim.model, scratch)
    mujoco.mj_rnePostConstraint(sim.model, scratch)
    body = sim.model.jnt_bodyid[sim.arm_jids[0]]
    projected = scratch.cdof[0] @ scratch.cfrc_int[body]
    np.testing.assert_allclose(sample['joint_torque_measured_sim'], [projected], atol=2e-12)


def test_known_off_origin_body_force_has_correct_robot_on_environment_sign():
    sim = _pendulum(gravity='0 0 0')
    sim.data.qpos[:] = .35
    sim.data.ctrl[:] = .35
    mujoco.mj_forward(sim.model, sim.data)
    body = sim.model.body('link').id
    point = sim.data.site_xpos[sim.tcp_id].copy() + np.array([.02, .03, -.01])
    force = np.array([2., -3., 4.])
    torque = np.array([.2, .4, -.1])
    sim.data.xfrc_applied[body, :3] = force
    sim.data.xfrc_applied[body, 3:] = torque + np.cross(point - sim.data.xipos[body], force)
    jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
    mujoco.mj_jac(sim.model, sim.data, jp, jr, point, body)
    expected = -(jp.T @ force + jr.T @ torque)
    sample = TorqueSignalSampler(sim).sample()
    np.testing.assert_allclose(sample['joint_external_torque_raw'], expected, atol=3e-12)
    np.testing.assert_allclose(sample['joint_applied_external_torque'], expected, atol=3e-12)
    assert sample['robot_external_contact_count'] == 0


def test_contact_ground_truth_and_residual_are_independent_but_agree():
    sim = _pendulum(contact=True)
    sample = TorqueSignalSampler(sim).sample()
    assert sample['robot_external_contact_count'] > 0
    assert sample['contact_wrench_at_tcp_world_gt'][2] < 0  # Robot pushes down on ground.
    np.testing.assert_allclose(sample['joint_external_torque_raw'], sample['joint_contact_torque_gt'], atol=2e-10)
    np.testing.assert_allclose(sample['contact_wrench_at_tcp_tcp_gt'][:3],
                               sample['tcp_rotation'].T @ sample['contact_wrench_at_tcp_world_gt'][:3], atol=1e-12)


def test_generalized_applied_force_is_not_hidden_inside_virtual_measurement():
    sim = _pendulum(gravity='0 0 0')
    sim.data.qfrc_applied[0] = 1.75
    sample = TorqueSignalSampler(sim).sample()
    np.testing.assert_allclose(sample['joint_external_torque_raw'], [-1.75], atol=1e-12)
    np.testing.assert_allclose(sample['joint_applied_external_torque'], [-1.75], atol=1e-12)


def test_joint_limit_residual_is_labelled_separately_from_contact_truth():
    sim = _sim('''<mujoco><compiler angle="radian"/><option gravity="0 0 0" timestep=".001"/>
      <worldbody><body><joint name="joint" range="-.1 .1"/>
      <geom type="capsule" fromto="0 0 0 .3 0 0" size=".02" mass="1"/>
      <site name="tcp" pos=".3 0 0"/></body></worldbody>
      <actuator><position joint="joint" kp="100" kv="10"/></actuator></mujoco>''')
    sim.data.qpos[0] = .2
    sim.data.ctrl[0] = .2
    sample = TorqueSignalSampler(sim).sample()
    assert abs(sample['joint_external_torque_raw'][0]) > 1
    assert sample['robot_external_contact_count'] == 0
    np.testing.assert_allclose(sample['joint_contact_torque_gt'], 0)
    np.testing.assert_allclose(sample['joint_external_torque_raw'],
                               sample['joint_noncontact_constraint_torque'], atol=1e-10)


def test_causal_filter_and_reset_have_defined_initialization():
    sim = _pendulum(gravity='0 0 0')
    sampler = TorqueSignalSampler(sim, cutoff_hz=20.)
    first = sampler.sample()
    np.testing.assert_allclose(first['joint_external_torque_estimated'], 0, atol=1e-12)
    sim.data.qfrc_applied[0] = 2.
    sim.data.time = .001
    second = sampler.sample()
    alpha = 1 - np.exp(-2 * np.pi * 20 * .001)
    np.testing.assert_allclose(second['joint_external_torque_estimated'], [-2 * alpha], atol=1e-12)
    with pytest.raises(ValueError, match='strictly increasing'):
        sampler.sample()
    sampler.reset()
    np.testing.assert_allclose(sampler.sample()['joint_external_torque_estimated'], [-2], atol=1e-12)


def test_pd_request_and_saturated_output_are_distinct():
    sim = _pendulum()
    sim.data.ctrl[:] = 1.
    sample = TorqueSignalSampler(sim).sample()
    np.testing.assert_allclose(sample['joint_pd_torque_unclipped'], [100.])
    np.testing.assert_allclose(sample['joint_actuator_torque'], [8.])
    assert sample['joint_actuator_saturated'].all()


def test_wrench_translation_and_rotation_include_lever_arm():
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    # World +Y force at world +X gives +Z moment at the origin.
    actual = transform_wrench([0., 2., 0., 0., 0., .3], rotation, [1, 0, 0], [0, 0, 0])
    np.testing.assert_allclose(actual, [2., 0., 0., 0., 0., 2.3])


def test_scaled_wrench_estimate_recovers_known_six_axis_loading():
    jacobian = np.c_[np.diag([.3, .25, .2, 1., .8, .9]), [.1, .03, .04, .1, -.2, .3]]
    expected = np.array([3., -4., 5., .2, -.3, .4])
    torque = jacobian.T @ expected
    wrench, singular, residual, valid = estimate_wrench(jacobian, torque, damping=1e-8)
    np.testing.assert_allclose(wrench, expected, atol=1e-10)
    np.testing.assert_allclose(residual, 0, atol=1e-11)
    assert valid and np.isfinite(singular).all()
    wrench, singular, residual, valid = estimate_wrench(np.zeros((6, 7)), np.ones(7))
    assert not valid
    assert np.isfinite(np.r_[wrench, singular, residual]).all()


def test_two_finger_forces_keep_signs_while_net_grasp_force_cancels():
    sim = _sim('''<mujoco><option gravity="0 0 0" timestep=".001" tolerance="1e-12"/>
      <worldbody><body name="arm"><joint name="joint" axis="0 0 1"/>
      <geom type="sphere" size=".01" mass="1" contype="0" conaffinity="0"/>
      <site name="tcp" pos=".5 0 0"/>
      <body name="left" pos=".5 .07 0"><geom type="box" size=".05 .04 .05"/></body>
      <body name="right" pos=".5 -.07 0"><geom type="box" size=".05 .04 .05"/></body>
      </body><body name="peg" pos=".5 0 0"><freejoint/><geom type="sphere" size=".04" mass=".1"/></body>
      </worldbody><actuator><position joint="joint" kp="100" kv="10"/></actuator></mujoco>''')
    sim.peg_id = sim.model.body('peg').id
    sim.gripper = SimpleNamespace(_side_bodies={sim.model.body('left').id: 'left', sim.model.body('right').id: 'right'})
    sample = TorqueSignalSampler(sim).sample()
    assert np.all(sample['finger_normal_force'] > 0)
    assert sample['finger_contact_force_world'][0, 1] > 0
    assert sample['finger_contact_force_world'][1, 1] < 0
    np.testing.assert_allclose(sample['finger_contact_force_world'].sum(axis=0), 0, atol=1e-8)
    np.testing.assert_allclose(sample['contact_wrench_at_tcp_world_gt'], 0, atol=1e-8)


def test_sampling_does_not_change_live_state_or_future_contact_trajectory():
    sim = _pendulum(contact=True)
    reference = mujoco.MjData(sim.model)
    mujoco.mj_copyData(reference, sim.model, sim.data)
    sampler = TorqueSignalSampler(sim)
    for i in range(100):
        target = .015 * np.sin(i / 30)
        sim.data.ctrl[:] = reference.ctrl[:] = target
        before = _integration_state(sim)
        sampler.sample()
        np.testing.assert_array_equal(_integration_state(sim), before)
        mujoco.mj_step(sim.model, sim.data)
        mujoco.mj_step(sim.model, reference)
        for field in ('qpos', 'qvel', 'qacc', 'qacc_warmstart', 'qfrc_constraint'):
            np.testing.assert_array_equal(getattr(sim.data, field), getattr(reference, field))


def test_real_fr3_model_matches_component_balance_and_stays_passive(tmp_path):
    from examples.insert_plug.environment.scene import build_scene, load_config
    from examples.insert_plug.environment.simulation import Simulation
    config = load_config()
    path = build_scene(config, tmp_path / 'scene.xml')
    sim = Simulation(path, config, render=False)
    try:
        sampler = TorqueSignalSampler(sim)
        before = _integration_state(sim)
        sample = sampler.sample(np.r_[sim.last_target, sim.gripper_width()])
        np.testing.assert_array_equal(_integration_state(sim), before)
        expected = (sample['joint_contact_torque_gt'] + sample['joint_self_contact_torque_gt']
                    + sample['joint_noncontact_constraint_torque'] + sample['joint_applied_external_torque'])
        np.testing.assert_allclose(sample['joint_external_torque_raw'], expected, atol=1e-7)
        assert sample['joint_torque_measured_sim'].shape == (7,)
        assert sample['ee_external_wrench_tcp_estimated'].shape == (6,)
        assert all(np.isfinite(x).all() for x in sample.values())
        assert sampler.metadata()['virtual_measurement']['kind'] == 'ideal_generalized_link_side_drive_proxy'
    finally:
        sim.close()


def test_real_fr3_tcp_load_recovers_known_six_axis_wrench(tmp_path):
    from examples.insert_plug.environment.scene import build_scene, load_config
    from examples.insert_plug.environment.simulation import Simulation

    config = load_config()
    sim = Simulation(build_scene(config, tmp_path / 'scene.xml'), config, render=False)
    try:
        # Apply an independently specified physical force and moment at TCP.
        # xfrc_applied's torque origin is the body's centre of mass.
        body = int(sim.model.site_bodyid[sim.tcp_id])
        force = np.array([1.2, -.7, 2.3])
        torque = np.array([.04, -.06, .03])
        lever = sim.data.site_xpos[sim.tcp_id] - sim.data.xipos[body]
        sim.data.xfrc_applied[body, :3] = force
        sim.data.xfrc_applied[body, 3:] = torque + np.cross(lever, force)
        sample = TorqueSignalSampler(sim, damping=1e-8).sample()
        expected_wrench = -np.r_[force, torque]
        expected_torque = sample['tcp_jacobian_world'].T @ expected_wrench
        np.testing.assert_allclose(sample['joint_external_torque_raw'], expected_torque, atol=1e-8)
        np.testing.assert_allclose(sample['ee_external_wrench_world_estimated'], expected_wrench, atol=1e-7)
        rotation = sample['tcp_rotation']
        np.testing.assert_allclose(sample['ee_external_wrench_tcp_estimated'],
                                   np.r_[rotation.T @ expected_wrench[:3], rotation.T @ expected_wrench[3:]], atol=1e-7)
        assert sample['wrench_estimate_well_conditioned']
    finally:
        sim.close()
