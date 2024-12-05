import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
import time
from simple_mpc import RobotHandler, KinodynamicsProblem, MPC, IDSolver

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)

""" Define robot model through handler"""
design_conf = dict(
    urdf_path=modelPath + URDF_SUBPATH,
    srdf_path=modelPath + SRDF_SUBPATH,
    robot_description="",
    root_name="root_joint",
    base_configuration="half_sitting",
    controlled_joints_names=[
        "root_joint",
        "leg_left_1_joint",
        "leg_left_2_joint",
        "leg_left_3_joint",
        "leg_left_4_joint",
        "leg_left_5_joint",
        "leg_left_6_joint",
        "leg_right_1_joint",
        "leg_right_2_joint",
        "leg_right_3_joint",
        "leg_right_4_joint",
        "leg_right_5_joint",
        "leg_right_6_joint",
        "torso_1_joint",
        "torso_2_joint",
        "arm_left_1_joint",
        "arm_left_2_joint",
        "arm_left_3_joint",
        "arm_left_4_joint",
        "arm_right_1_joint",
        "arm_right_2_joint",
        "arm_right_3_joint",
        "arm_right_4_joint",
    ],
    end_effector_names=[
        "left_sole_link",
        "right_sole_link",
    ],
    feet_to_base_trans=[
        np.array([0., 0.1, 0.]),
        np.array([0., -0.1, 0.]),
    ]
)
handler = RobotHandler()
handler.initialize(design_conf)

nq = handler.getModel().nq
nv = handler.getModel().nv

x0 = handler.getState()
nu = handler.getModel().nv - 6 + len(handler.getFeetNames()) * 6

""" Define kinodynamics problem """
gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -handler.getMass() / len(handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref, np.zeros(nv - 6)))

w_basepos = [0, 0, 1000, 1000, 1000, 1000]
w_legpos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
w_torsopos = [1, 1000]
w_armpos = [1, 1, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [1, 1, 1, 1, 1, 1]
w_torsovel = [0.1, 100]
w_armvel = [10, 10, 10, 10]
w_x = np.array(
    w_basepos
    + w_legpos * 2
    + w_torsopos
    + w_armpos * 2
    + w_basevel
    + w_legvel * 2
    + w_torsovel
    + w_armvel * 2
)
w_x = np.diag(w_x) * 10
w_linforce = np.array([0.001, 0.001, 0.01])
w_angforce = np.ones(3) * 0.1
w_u = np.concatenate(
    (
        w_linforce,
        w_angforce,
        w_linforce,
        w_angforce,
        np.ones(handler.getModel().nv - 6) * 1e-4,
    )
)
w_u = np.diag(w_u)
w_LFRF = 100000
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.1, 0.1, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.ones(3) * 0.1
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

problem_conf = dict(
    DT=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=6,
    w_frame=np.eye(6) * w_LFRF,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    kinematics_limits=True,
    force_cone=True,
)

T = 100

problem = KinodynamicsProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getState(), T, 6, gravity[2], False)

""" Define MPC object """
T_ds = 20
T_ss = 80
mpc_conf = dict(
    ddpIteration=1,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    T=T,
    dt=0.01,
    kinematics_limits=True,
    force_cone=True,
)

mpc = MPC()
mpc.initialize(mpc_conf, problem)

""" Define contact sequence throughout horizon"""
total_steps = 3
contact_phase_double = {
    "left_sole_link": True,
    "right_sole_link": True,
}
contact_phase_left = {
    "left_sole_link": True,
    "right_sole_link": False,
}
contact_phase_right = {
    "left_sole_link": False,
    "right_sole_link": True,
}
contact_phases = [contact_phase_double] * T_ds
contact_phases += [contact_phase_left] * T_ss
contact_phases += [contact_phase_double] * T_ds
contact_phases += [contact_phase_right] * T_ss

mpc.generateCycleHorizon(contact_phases)

""" Initialize whole-body inverse dynamics QP"""
contact_ids = handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=handler.getState(),
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
    kd=0,
    w_force=100,
    w_acc=1,
    w_tau=0,
    verbose=False,
)

qp = IDSolver()
qp.initialize(id_conf, handler.getModel())

""" Initialize simulation"""
device = BulletRobot(
    design_conf["controlled_joints_names"],
    modelPath + "/talos_data/robots/",
    URDF_FILENAME,
    1e-3,
    handler.getCompleteModel(),
)
device.initializeJoints(handler.getCompleteConfiguration())
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])
q_current, v_current = device.measureState()

x_measured = mpc.getHandler().shapeState(q_current, v_current)

q_current = x_measured[:nq]
v_current = x_measured[nq:]

Tmpc = len(contact_phases)
nk = 2
force_size = 6

device.showTargetToTrack(
    mpc.getHandler().getFootPose("left_sole_link"),
    mpc.getHandler().getFootPose("right_sole_link"),
)

v = np.zeros(6)
v[0] = 0.2
mpc.setVelocityBase(v)
for t in range(600):
    # print("Time " + str(t))
    if t == 400:
        print("SWITCH TO STAND")
        mpc.switchToStand()

    land_LF = mpc.getFootLandCycle("left_sole_link")
    land_RF = mpc.getFootLandCycle("right_sole_link")
    takeoff_LF = mpc.getFootTakeoffCycle("left_sole_link")
    takeoff_RF = mpc.getFootTakeoffCycle("right_sole_link")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    start = time.time()
    mpc.iterate(q_current, v_current)
    end = time.time()
    print("MPC iterate = " + str(end - start))
    a0 = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[nv:]
    )
    contact_states = (
        mpc.getTrajOptProblem().stages[0].dynamics.differential_dynamics.contact_states
    )

    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    """ if t == 60:
        top=mpc.getTrajOptProblem()
        exit()
        for s in range(T):
            device.resetState(mpc.xs[s][:nq])
            time.sleep(0.1)
            print("s = " + str(s))
        exit()  """
    for j in range(10):
        q_current, v_current = device.measureState()
        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        state_diff = mpc.getHandler().difference(x_measured, mpc.xs[0])
        mpc.getHandler().updateState(q_current, v_current, True)
        a0[6:] = (
            mpc.us[0][nk * force_size :]
            - 1
            * mpc.getSolver().results.controlFeedbacks()[0][nk * force_size :]
            @ state_diff
        )
        forces = (
            mpc.us[0][: nk * force_size]
            - 1
            * mpc.getSolver().results.controlFeedbacks()[0][: nk * force_size]
            @ state_diff
        )
        qp.solveQP(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            a0,
            np.zeros(12),
            forces,
            mpc.getHandler().getMassMatrix(),
        )

        device.execute(qp.solved_torque)
