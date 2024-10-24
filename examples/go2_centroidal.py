import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, CentroidalProblem, MPC, IKIDSolver
import example_robot_data
import time
from QP_utils import IKIDSolver_f3
import pinocchio as pin
from aligator import manifolds


def compute_ID_references(
    space,
    rmodel,
    rdata,
    FL_id,
    FR_id,
    RL_id,
    RR_id,
    base_id,
    x0_multibody,
    x_measured,
    FL_refs,
    FR_refs,
    RL_refs,
    RR_refs,
    dt,
):
    FL_vel_lin = pin.getFrameVelocity(rmodel, rdata, FL_id, pin.LOCAL).linear
    FR_vel_lin = pin.getFrameVelocity(rmodel, rdata, FR_id, pin.LOCAL).linear
    RL_vel_lin = pin.getFrameVelocity(rmodel, rdata, RL_id, pin.LOCAL).linear
    RR_vel_lin = pin.getFrameVelocity(rmodel, rdata, RR_id, pin.LOCAL).linear
    FL_vel_ang = pin.getFrameVelocity(rmodel, rdata, FL_id, pin.LOCAL).angular
    FR_vel_ang = pin.getFrameVelocity(rmodel, rdata, FR_id, pin.LOCAL).angular
    RL_vel_ang = pin.getFrameVelocity(rmodel, rdata, RL_id, pin.LOCAL).angular
    RR_vel_ang = pin.getFrameVelocity(rmodel, rdata, RR_id, pin.LOCAL).angular

    q_diff = -space.difference(x0_multibody, x_measured)[: rmodel.nv]
    dq_diff = -space.difference(x0_multibody, x_measured)[rmodel.nv :]
    FL_diff = np.zeros(3)
    FL_diff = FL_refs[0].translation - rdata.oMf[FL_id].translation
    FR_diff = np.zeros(3)
    FR_diff = FR_refs[0].translation - rdata.oMf[FR_id].translation
    RL_diff = np.zeros(3)
    RL_diff = RL_refs[0].translation - rdata.oMf[RL_id].translation
    RR_diff = np.zeros(3)
    RR_diff = RR_refs[0].translation - rdata.oMf[RR_id].translation

    dFL_diff = np.zeros(3)
    dFL_diff = (FL_refs[1].translation - FL_refs[0].translation) / dt - FL_vel_lin
    dFR_diff = np.zeros(3)
    dFR_diff = (FR_refs[1].translation - FR_refs[0].translation) / dt - FR_vel_lin
    dRL_diff = np.zeros(3)
    dRL_diff = (RL_refs[1].translation - RL_refs[0].translation) / dt - RL_vel_lin
    dRR_diff = np.zeros(3)
    dRR_diff = (RR_refs[1].translation - RR_refs[0].translation) / dt - RR_vel_lin

    base_diff = -pin.log3(rdata.oMf[base_id].rotation)
    dbase_diff = -pin.getFrameVelocity(rmodel, rdata, base_id, pin.LOCAL).angular

    return (
        q_diff,
        dq_diff,
        FL_diff,
        dFL_diff,
        FR_diff,
        dFR_diff,
        RL_diff,
        dRL_diff,
        RR_diff,
        dRR_diff,
        base_diff,
        dbase_diff,
    )


SRDF_SUBPATH = "/go2_description/srdf/go2.srdf"
URDF_SUBPATH = "/go2_description/urdf/go2.urdf"
modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
# ####### CONFIGURATION  ############
# ### RobotWrapper
design_conf = dict(
    urdf_path=modelPath + URDF_SUBPATH,
    srdf_path=modelPath + SRDF_SUBPATH,
    robot_description="",
    root_name="root_joint",
    base_configuration="standing",
    controlled_joints_names=[
        "root_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
    ],
    end_effector_names=[
        "FL_foot",
        "FR_foot",
        "RL_foot",
        "RR_foot",
    ],
)
handler = RobotHandler()
handler.initialize(design_conf)


x0 = np.zeros(9)
x0[:3] = handler.getComPosition()
force_size = 3
nk = len(handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref))

w_control_linear = np.ones(3) * 0.0001
w_u = np.diag(
    np.concatenate(
        (w_control_linear, w_control_linear, w_control_linear, w_control_linear)
    )
)
w_linear_mom = np.diag(np.array([0, 0, 0]))
w_linear_acc = 1 * np.eye(3)
w_angular_mom = np.diag(np.array([1, 1, 1]))
w_angular_acc = 1 * np.eye(3)

problem_conf = dict(
    x0=x0,
    u0=u0,
    DT=0.01,
    w_u=w_u,
    w_linear_mom=w_linear_mom,
    w_angular_mom=w_angular_mom,
    w_linear_acc=w_linear_acc,
    w_angular_acc=w_angular_acc,
    gravity=gravity,
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=force_size,
)
T = 100

problem = CentroidalProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getCentroidalState(), T, force_size, gravity[2])

T_ds = 50
T_ss = 50

mpc_conf = dict(
    ddpIteration=1,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=2,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    T=T,
    x_translation=0.0,
    y_translation=0.0,
)

mpc = MPC()
mpc.initialize(mpc_conf, problem)

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift_FL = {
    "FL_foot": False,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}

contact_phases = (
    [contact_phase_quadru] * T_ds
    + [contact_phase_lift_FL] * T_ss
    + [contact_phase_quadru] * T_ds
    # + [contact_phase_lift_FR] * T_ss
)

mpc.generateCycleHorizon(contact_phases)

""" Initialize inverse dynamics QP """
g_basepos = [0, 0, 0, 1, 1, 1]
g_legpos = [1, 1, 1, 1, 1, 1]

g_q = np.array(g_basepos + g_legpos * 2) * 1000

g_p = np.array([10, 10, 10])
g_b = np.array([10, 10, 10])

Kp_gains = [g_q, g_p, g_b]
Kd_gains = [2 * np.sqrt(g_q), 2 * np.sqrt(g_p), 2 * np.sqrt(g_b)]
contact_ids = handler.getFeetIds()
fixed_frame_ids = [handler.getRootId()]
ikid_conf = dict(
    Kp_gains=Kp_gains,
    Kd_gains=Kd_gains,
    contact_ids=contact_ids,
    fixed_frame_ids=fixed_frame_ids,
    x0=handler.getState(),
    dt=0.01,
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=force_size,
    w_qref=1000,
    w_footpose=0,
    w_centroidal=100,
    w_baserot=10,
    w_force=10,
    verbose=False,
)

qp = IKIDSolver()
qp.initialize(ikid_conf, handler.getModel())

x0_multibody = handler.getState().copy()

g_p = 200
g_b = 10
K_gains = []
g_basepos = [1, 1, 1, 1, 1, 1]
g_legpos = [1, 1, 1, 1, 1, 1]
g_q = np.array(g_basepos + g_legpos * 2) * 10000

K_gains.append([g_q, 2 * np.sqrt(g_q)])
K_gains.append([np.eye(3) * g_p, np.eye(3) * 2 * np.sqrt(g_p)])
K_gains.append([np.eye(3) * g_b, np.eye(3) * 2 * np.sqrt(g_b)])

weights_IKID = [
    10,
    1000,
    10000,
    100,
    10,
]  # qref, foot_pose, centroidal, base_rot, force
IKID_solver = IKIDSolver_f3(
    handler.getModel(),
    weights_IKID,
    K_gains,
    nk,
    0.8,
    contact_ids,
    handler.getRootId(),
    False,
)

""" Initialize simulation"""
device = BulletRobot(
    design_conf["controlled_joints_names"],
    modelPath,
    URDF_SUBPATH,
    1e-3,
    handler.getModel(),
    handler.getState()[:3],
)

device.initializeJoints(handler.getConfiguration())
device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])
q_current, v_current = device.measureState()
nq = mpc.getHandler().getModel().nq
nv = mpc.getHandler().getModel().nv

x_measured = mpc.getHandler().shapeState(q_current, v_current)
q_current = x_measured[:nq]
v_current = x_measured[nq:]

device.showQuadrupedFeet(
    mpc.getHandler().getFootPose("FL_foot"),
    mpc.getHandler().getFootPose("FR_foot"),
    mpc.getHandler().getFootPose("RL_foot"),
    mpc.getHandler().getFootPose("RR_foot"),
)
space_multibody = manifolds.MultibodyPhaseSpace(handler.getModel())
for t in range(1000):
    # mpc.switchToStand()
    print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    mpc.iterate(q_current, v_current)

    contact_states = (
        mpc.getTrajOptProblem()
        .stages[0]
        .dynamics.differential_dynamics.contact_map.contact_states.tolist()
    )
    print(contact_states)
    foot_ref = [mpc.getReferencePose(0, name) for name in handler.getFeetNames()]
    foot_ref_next = [mpc.getReferencePose(1, name) for name in handler.getFeetNames()]
    dH = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[3:9]
    )
    qp.computeDifferences(
        mpc.getHandler().getData(), x_measured, foot_ref, foot_ref_next
    )

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    FL_refs = [mpc.getReferencePose(0, "FL_foot"), mpc.getReferencePose(1, "FL_foot")]
    FR_refs = [mpc.getReferencePose(0, "FR_foot"), mpc.getReferencePose(1, "FR_foot")]
    RL_refs = [mpc.getReferencePose(0, "RL_foot"), mpc.getReferencePose(1, "RL_foot")]
    RR_refs = [mpc.getReferencePose(0, "RR_foot"), mpc.getReferencePose(1, "RR_foot")]

    (
        q_diff,
        dq_diff,
        FL_diff,
        dFL_diff,
        FR_diff,
        dFR_diff,
        RL_diff,
        dRL_diff,
        RR_diff,
        dRR_diff,
        base_diff,
        dbase_diff,
    ) = compute_ID_references(
        space_multibody,
        handler.getModel(),
        mpc.getHandler().getData(),
        contact_ids[0],
        contact_ids[1],
        contact_ids[2],
        contact_ids[3],
        handler.getRootId(),
        x0_multibody,
        x_measured,
        FL_refs,
        FR_refs,
        RL_refs,
        RR_refs,
        0.01,
    )
    if t == 140:
        exit()

    for j in range(10):
        time.sleep(0.001)
        q_current, v_current = device.measureState()
        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        mpc.getHandler().updateState(q_current, v_current, True)
        x_centroidal = mpc.getHandler().getCentroidalState()
        state_diff = mpc.xs[0] - x_centroidal

        forces = (
            mpc.us[0]
            - 1
            * mpc.getSolver().results.controlFeedbacks()[0][: nk * force_size]
            @ state_diff
        )

        qp.solve_qp(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            forces,
            dH,
            mpc.getHandler().getMassMatrix(),
        )

        """ new_acc, new_forces, torque = IKID_solver.solve(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            q_diff,
            dq_diff,
            FL_diff,
            dFL_diff,
            FR_diff,
            dFR_diff,
            RL_diff,
            dRL_diff,
            RR_diff,
            dRR_diff,
            base_diff,
            dbase_diff,
            forces,
            dH,
            mpc.getHandler().getMassMatrix(),
        ) """

        """ print(qp.solved_torque)

        if (not(contact_states[0])):
            exit() """

        device.execute(qp.solved_torque)
