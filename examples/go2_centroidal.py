import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, CentroidalProblem, MPC, IKIDSolver
import example_robot_data
import time

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
w_linear_mom = np.diag(np.array([1, 1, 1]))
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

T_ds = 20
T_ss = 500

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
    + [contact_phase_lift_FR] * T_ss
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
for t in range(10000):
    # print("Time " + str(t))
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

        """ print(qp.solved_torque)

        if (not(contact_states[0])):
            exit() """

        device.execute(qp.solved_torque)
