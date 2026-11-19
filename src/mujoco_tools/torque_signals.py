"""Passive, explicitly labelled torque proxies and force estimates for MuJoCo.

This is an ideal simulation measurement/estimation chain, not an implementation
of Franka's proprietary torque sensor or external-force observer.  In particular
``qfrc_actuator`` is never presented as a measured link-side joint torque.

All calculations use a refreshed MjData copy.  Reading a sample does not advance
or change the live simulation, including its contact solver warm start.
"""

from __future__ import annotations

import math
from typing import Any

import mujoco
import numpy as np


def transform_wrench(wrench, rotation, source_origin, target_origin):
    """Move a [force, torque] world wrench to target origin and target axes.

    ``rotation`` maps target coordinates to world coordinates.  The source
    wrench is expressed in world coordinates about ``source_origin``.
    """
    wrench = np.asarray(wrench, dtype=float)
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    lever = np.asarray(source_origin, dtype=float) - np.asarray(target_origin, dtype=float)
    return np.r_[rotation.T @ wrench[:3],
                 rotation.T @ (wrench[3:] + np.cross(lever, wrench[:3]))]


def estimate_wrench(jacobian, joint_torque, *, length_scale=0.1, damping=1e-4):
    """Damped least-squares equivalent wrench, at the Jacobian's origin.

    Scale the torque coordinates with a characteristic length, so all solution
    coordinates have units N and singular values/damping have units m.
    Returns (wrench, singular_values, joint_residual, well_conditioned).
    """
    jacobian = np.asarray(jacobian, dtype=float)
    joint_torque = np.asarray(joint_torque, dtype=float)
    if jacobian.ndim != 2 or jacobian.shape[0] != 6 or joint_torque.shape != (jacobian.shape[1],):
        raise ValueError('Expected a 6-by-n Jacobian and n joint torques')
    if not np.isfinite(jacobian).all() or not np.isfinite(joint_torque).all():
        raise ValueError('Wrench estimation requires finite inputs')
    if not np.isfinite([length_scale, damping]).all() or length_scale <= 0 or damping <= 0:
        raise ValueError('length_scale and damping must be finite and positive')
    scale = np.array([1., 1., 1., length_scale, length_scale, length_scale])
    matrix = jacobian.T * scale
    u, singular, vt = np.linalg.svd(matrix, full_matrices=False)
    coordinates = vt.T @ ((singular / (singular**2 + damping**2)) * (u.T @ joint_torque))
    wrench = coordinates * scale
    residual = joint_torque - jacobian.T @ wrench
    valid = bool(len(singular) == 6 and singular[-1] > 10 * damping)
    return wrench, singular, residual, valid


class TorqueSignalSampler:
    """Observe a Simulation-compatible object at its current physics tick.

    ``sample(action)`` returns independent NumPy arrays and scalar values.  Call
    once before each integration, after installing that tick's active control.
    The estimator uses ideal simulator acceleration and the exact model.  Its
    external torque is a dynamics residual followed by a causal first-order
    filter, not a renamed contact-force query.
    """

    def __init__(self, sim, *, cutoff_hz=30.0, length_scale=0.1, damping=1e-4):
        self.sim = sim
        self.model = sim.model
        self.data = mujoco.MjData(self.model)
        self.arm_jids = np.asarray(sim.arm_jids, dtype=int).copy()
        self.arm_dadr = np.asarray(sim.arm_dadr, dtype=int).copy()
        self.arm_qadr = np.asarray(sim.arm_qadr, dtype=int).copy()
        self.arm_aids = np.asarray(sim.arm_aids, dtype=int).copy()
        self.tcp_id = int(sim.tcp_id)
        self.cutoff_hz, self.length_scale, self.damping = map(float, (cutoff_hz, length_scale, damping))
        if not np.isfinite([self.cutoff_hz, self.length_scale, self.damping]).all():
            raise ValueError('Estimator parameters must be finite')
        if not 0 < self.cutoff_hz < 0.5 / self.model.opt.timestep:
            raise ValueError('cutoff_hz must lie below the physics Nyquist frequency')
        if min(self.length_scale, self.damping) <= 0:
            raise ValueError('length_scale and damping must be positive')
        if np.any(self.model.jnt_type[self.arm_jids] != mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError('Only scalar hinge arm joints are supported')
        if len(set(self.arm_dadr)) != len(self.arm_dadr) or len(self.arm_aids) != len(self.arm_dadr):
            raise ValueError('Arm requires distinct scalar DOFs and one selected actuator per joint')
        for joint, actuator in zip(self.arm_jids, self.arm_aids):
            if (self.model.actuator_trntype[actuator] != mujoco.mjtTrn.mjTRN_JOINT
                    or self.model.actuator_trnid[actuator, 0] != joint
                    or self.model.actuator_gaintype[actuator] != mujoco.mjtGain.mjGAIN_FIXED
                    or self.model.actuator_biastype[actuator] != mujoco.mjtBias.mjBIAS_AFFINE
                    or self.model.actuator_dyntype[actuator] != mujoco.mjtDyn.mjDYN_NONE):
                raise ValueError('PD diagnostics require direct fixed-gain affine-bias joint actuators')
        # The virtual-drive interpretation of all passive forces below is only
        # intended for the dry, rigid-body, joint-damped task model.
        if self.model.opt.density != 0 or self.model.opt.viscosity != 0:
            raise ValueError('Fluid forces require a separate external-force model')
        if hasattr(self.model, 'tendon_armature') and np.any(self.model.tendon_armature):
            raise ValueError('Tendon armature is not covered by the diagonal rotor correction')
        first_body = int(self.model.jnt_bodyid[self.arm_jids[0]])
        parent = int(self.model.body_parentid[first_body])
        self.robot_root = parent if parent > 0 else first_body
        self.tool_root = int(self.model.site_bodyid[self.tcp_id])
        self.robot_mask = self._subtree_mask(self.robot_root)
        self.tool_mask = self._subtree_mask(self.tool_root)
        self.peg_id = int(getattr(sim, 'peg_id', -1))
        self.finger_sides: dict[int, int] = {}
        gripper = getattr(sim, 'gripper', None)
        if gripper is not None and hasattr(gripper, '_side_bodies'):
            self.finger_sides = {int(body): ('left', 'right').index(side)
                                 for body, side in gripper._side_bodies.items()}
        elif hasattr(sim, 'finger_jids'):
            for index, joint in enumerate(sim.finger_jids[:2]):
                for body in np.flatnonzero(self._subtree_mask(int(self.model.jnt_bodyid[joint]))):
                    self.finger_sides[int(body)] = index
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self.reset()

    def _subtree_mask(self, root):
        mask = np.zeros(self.model.nbody, dtype=bool)
        for body in range(root, self.model.nbody):
            ancestor = body
            while ancestor > 0 and ancestor != root:
                ancestor = int(self.model.body_parentid[ancestor])
            mask[body] = ancestor == root
        return mask

    def reset(self):
        self._last_time = None
        self._filtered = None

    def metadata(self) -> dict[str, Any]:
        return dict(
            schema_version='torque-signals-v1', sample_hz=1 / float(self.model.opt.timestep),
            physics_timestep=float(self.model.opt.timestep), mujoco_version=mujoco.__version__,
            joint_names=[self.model.joint(int(j)).name for j in self.arm_jids],
            arm_qpos_addresses=self.arm_qadr.tolist(), arm_dof_addresses=self.arm_dadr.tolist(),
            arm_actuator_ids=self.arm_aids.tolist(),
            joint_torque_units='N m', force_units='N', wrench_order=['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz'],
            wrench_units=['N', 'N', 'N', 'N m', 'N m', 'N m'],
            pose_order=['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz'], finger_order=['left', 'right'],
            external_sign='robot_on_environment; opposite of environment_on_robot',
            finger_force_sign='object_on_finger; normal_force is an unsigned sum',
            wrench_reference_point='tcp site origin; world or tcp axes as named',
            virtual_measurement=dict(
                kind='ideal_generalized_link_side_drive_proxy',
                formula='actuator + passive + internal_friction - armature*qacc',
                rotor_inertia_removed=True,
                body_gravity_compensation='included once through actuator/passive routing; reinterpreted as virtual drive compensation',
                passive_force_interpretation='all dry joint passive forces treated as internal drive-side contributions',
                limitations='noise-free rigid joints; no calibrated physical torque sensor, gearbox elasticity or motor dynamics; not a libfranka tau_J implementation'),
            external_estimator=dict(
                method='exact_model_acceleration_dynamics_residual_then_causal_first_order_lowpass',
                model_torque='(M_total*qacc + qfrc_bias - dof_armature*qacc)[arm_dof]',
                raw_formula='joint_torque_measured_sim - joint_model_torque',
                acceleration='ideal synchronized simulator qacc, not a real acceleration sensor',
                filter_cutoff_hz=self.cutoff_hz, filter_alpha='1-exp(-2*pi*cutoff_hz*delta_time)',
                filter_initialization='first sample equals first raw residual',
                includes='external contact, applied force, and non-friction constraint residuals including joint limits',
                excludes='DOF/tendon frictionloss is treated as internal friction',
                warning='simulation estimator; not the proprietary Franka observer and not noise-matched to hardware'),
            wrench_estimator=dict(
                method='dimensionally_scaled_damped_least_squares',
                matrix='A=[J_position.T, length_scale*J_rotation.T]',
                solution='y=V diag(s/(s*s+damping*damping)) U.T tau_external_estimated; wrench=[y[:3],length_scale*y[3:]]',
                characteristic_length_m=self.length_scale, damping_m=self.damping,
                valid_condition='six singular values and minimum > 10*damping; validity reports conditioning only',
                assumption='external loading is equivalent to a wrench at the TCP; distributed/arm contact and joint limits can violate this assumption',
                singular_handling='keep bounded DLS estimate and flag ill-conditioned; does not copy Franka zero-near-singularity behavior'),
            contact_ground_truth=dict(
                source='mj_contactForce on synchronized copy; separate from estimator',
                robot_root_body=self.model.body(self.robot_root).name,
                tool_root_body=self.model.body(self.tool_root).name,
                finger_force_scope='contact between each finger-side body and peg only',
                independent_finger_force_available_on_real_robot=False,
                note='simulation-only privileged diagnostic; do not assume real finger sensing'),
            signal_correspondence=dict(
                joint_torque_measured_sim='approximate signal role of tau_J; not identical sensing physics',
                joint_external_torque_estimated='signal role of tau_ext_hat_filtered; different estimator',
                ee_external_wrench_tcp_estimated='signal role of K_F_ext_hat_K only after explicitly matching K to TCP',
                joint_actuator_torque='simulator drive output, not tau_J',
                joint_pd_torque_unclipped='simulator requested PD output, not automatically tau_J_d'),
            robot_model_mass_kg=self.model.body_mass[self.robot_mask].tolist(),
            body_mass_kg=self.model.body_mass.tolist(), body_inertia_kg_m2=self.model.body_inertia.tolist(),
            body_gravcomp=self.model.body_gravcomp.tolist(), dof_armature=self.model.dof_armature.tolist(),
            dof_damping=self.model.dof_damping.tolist(), dof_frictionloss=self.model.dof_frictionloss.tolist(),
        )

    def _constraint_component(self, kinds):
        force = np.zeros(self.data.nefc)
        selection = np.isin(self.data.efc_type, kinds)
        force[selection] = self.data.efc_force[selection]
        result = np.zeros(self.model.nv)
        if self.data.nefc:
            mujoco.mj_mulJacTVec(self.model, self.data, result, force)
        return result

    def _contacts(self, tcp_position, tcp_rotation):
        model, data = self.model, self.data
        joint_external = np.zeros(model.nv)
        joint_self = np.zeros(model.nv)
        wrench_world = np.zeros(6)
        finger_normal = np.zeros(2)
        finger_xyz = np.zeros((2, 3))
        finger_count = np.zeros(2, dtype=np.int32)
        robot_count = tool_count = arm_count = 0
        local = np.zeros(6)
        for index, contact in enumerate(data.contact):
            if contact.geom1 < 0 or contact.geom2 < 0:
                continue  # Flex contacts are not part of this rigid-body task.
            bodies = model.geom_bodyid[[contact.geom1, contact.geom2]]
            inside = self.robot_mask[bodies]
            if not inside.any():
                continue
            mujoco.mj_contactForce(model, data, index, local)
            # MuJoCo's contact frame normal points geom1 -> geom2, and the
            # returned wrench is the force applied to geom2 by geom1.
            rotation = np.asarray(contact.frame).reshape(3, 3)
            world = np.r_[rotation.T @ local[:3], rotation.T @ local[3:]]
            for which, body in enumerate(bodies):
                side = self.finger_sides.get(int(body))
                if side is not None and bodies[1-which] == self.peg_id:
                    on_finger = world[:3] * (-1 if which == 0 else 1)
                    finger_normal[side] += max(0., float(local[0]))
                    finger_xyz[side] += on_finger
                    if contact.efc_address >= 0:
                        finger_count[side] += 1
            if inside.all():
                for which, body in enumerate(bodies):
                    on_robot = world * (-1 if which == 0 else 1)
                    mujoco.mj_applyFT(model, data, -on_robot[:3], -on_robot[3:],
                                     contact.pos, int(body), joint_self)
                continue
            which = 0 if inside[0] else 1
            body = int(bodies[which])
            on_environment = world * (1 if which == 0 else -1)
            mujoco.mj_applyFT(model, data, on_environment[:3], on_environment[3:],
                             contact.pos, body, joint_external)
            wrench_world += transform_wrench(on_environment, np.eye(3), contact.pos, tcp_position)
            if contact.efc_address >= 0:
                robot_count += 1
                if self.tool_mask[body]:
                    tool_count += 1
                else:
                    arm_count += 1
        return dict(
            joint_contact_torque_gt=joint_external[self.arm_dadr],
            joint_self_contact_torque_gt=joint_self[self.arm_dadr],
            contact_wrench_at_tcp_world_gt=wrench_world,
            contact_wrench_at_tcp_tcp_gt=np.r_[tcp_rotation.T @ wrench_world[:3], tcp_rotation.T @ wrench_world[3:]],
            finger_normal_force=finger_normal, finger_contact_force_world=finger_xyz,
            finger_contact_force_tcp=finger_xyz @ tcp_rotation,
            finger_contact_count=finger_count, robot_external_contact_count=robot_count,
            tool_external_contact_count=tool_count, arm_external_contact_count=arm_count,
        )

    def sample(self, action=None):
        model, data = self.model, self.data
        stamp = float(self.sim.data.time)
        if self._last_time is not None and stamp <= self._last_time:
            raise ValueError('Torque samples require strictly increasing timestamps; reset after simulation reset')
        mujoco.mj_copyData(data, model, self.sim.data)
        mujoco.mj_forward(model, data)
        arm = self.arm_dadr
        friction = self._constraint_component([mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF,
                                                mujoco.mjtConstraint.mjCNSTR_FRICTION_TENDON])
        other_constraint = self._constraint_component([mujoco.mjtConstraint.mjCNSTR_EQUALITY,
                                                       mujoco.mjtConstraint.mjCNSTR_LIMIT_JOINT,
                                                       mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON])
        inertia_force = np.zeros(model.nv)
        mujoco.mj_mulM(model, data, inertia_force, data.qacc)
        rotor = model.dof_armature * data.qacc
        model_torque = (inertia_force + data.qfrc_bias - rotor)[arm]
        measured = (data.qfrc_actuator + data.qfrc_passive + friction - rotor)[arm]
        raw = measured - model_torque
        alpha = 1.0 if self._last_time is None else -math.expm1(-2 * math.pi * self.cutoff_hz * (stamp - self._last_time))
        filtered = raw.copy() if self._filtered is None else self._filtered + alpha * (raw - self._filtered)
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.tcp_id)
        jacobian = np.vstack([self._jacp[:, arm], self._jacr[:, arm]])
        wrench, singular, residual, valid = estimate_wrench(
            jacobian, filtered, length_scale=self.length_scale, damping=self.damping)
        tcp_position = data.site_xpos[self.tcp_id].copy()
        tcp_rotation = data.site_xmat[self.tcp_id].reshape(3, 3).copy()
        quaternion = np.empty(4)
        mujoco.mju_mat2Quat(quaternion, tcp_rotation.reshape(-1))
        controls = data.ctrl[self.arm_aids].copy()
        limited = model.actuator_ctrllimited[self.arm_aids].astype(bool)
        limits = model.actuator_ctrlrange[self.arm_aids]
        controls[limited] = np.clip(controls[limited], limits[limited, 0], limits[limited, 1])
        bias = model.actuator_biasprm[self.arm_aids]
        unclipped_force = (model.actuator_gainprm[self.arm_aids, 0] * controls + bias[:, 0]
                           + bias[:, 1] * data.actuator_length[self.arm_aids]
                           + bias[:, 2] * data.actuator_velocity[self.arm_aids])
        unclipped_torque = unclipped_force * model.actuator_gear[self.arm_aids, 0]
        saturated = (model.actuator_forcelimited[self.arm_aids].astype(bool)
                     & ((unclipped_force < model.actuator_forcerange[self.arm_aids, 0])
                        | (unclipped_force > model.actuator_forcerange[self.arm_aids, 1])))
        applied = data.qfrc_applied.copy()
        for body in np.flatnonzero(np.any(data.xfrc_applied != 0, axis=1)):
            mujoco.mj_applyFT(model, data, data.xfrc_applied[body, :3], data.xfrc_applied[body, 3:],
                             data.xipos[body], int(body), applied)
        result = dict(
            timestamp=stamp, qpos=data.qpos.copy(), qvel=data.qvel.copy(), qacc=data.qacc.copy(), ctrl=data.ctrl.copy(),
            joint_position=data.qpos[self.arm_qadr].copy(), joint_velocity=data.qvel[arm].copy(),
            joint_acceleration=data.qacc[arm].copy(), joint_position_target=data.ctrl[self.arm_aids].copy(),
            tcp_pose=np.r_[tcp_position, quaternion], tcp_rotation=tcp_rotation,
            tcp_velocity_world=np.r_[self._jacp @ data.qvel, self._jacr @ data.qvel],
            joint_torque_measured_sim=measured, joint_model_torque=model_torque,
            joint_external_torque_raw=raw, joint_external_torque_estimated=filtered.copy(),
            joint_actuator_torque=data.qfrc_actuator[arm].copy(), joint_pd_torque_unclipped=unclipped_torque,
            joint_actuator_saturated=saturated, joint_passive_torque=data.qfrc_passive[arm].copy(),
            joint_internal_friction_torque=friction[arm], joint_rotor_inertia_torque=rotor[arm],
            joint_gravity_compensation_torque=data.qfrc_gravcomp[arm].copy(),
            joint_noncontact_constraint_torque=-other_constraint[arm], joint_applied_external_torque=-applied[arm],
            ee_external_wrench_world_estimated=wrench,
            ee_external_wrench_tcp_estimated=np.r_[tcp_rotation.T @ wrench[:3], tcp_rotation.T @ wrench[3:]],
            tcp_jacobian_world=jacobian, wrench_jacobian_singular_values=singular,
            wrench_joint_residual=residual, wrench_estimate_well_conditioned=valid, external_filter_alpha=alpha,
        )
        if action is not None:
            result['action'] = np.asarray(action, dtype=float).copy()
        if hasattr(self.sim, 'gripper_width'):
            result['gripper_width'] = float(self.sim.gripper_width(data))
        result.update(self._contacts(tcp_position, tcp_rotation))
        if not all(np.isfinite(value).all() for value in result.values()):
            raise RuntimeError('Nonfinite torque sample')
        self._last_time, self._filtered = stamp, filtered.copy()
        return result
