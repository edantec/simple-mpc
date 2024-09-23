import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
import time
from simple_mpc import RobotHandler, KinodynamicsProblem, MPC, IDSolver
from QP_utils import IDSolverPython

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
# ####### CONFIGURATION  ############
# ### RobotWrapper
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
)
handler = RobotHandler()
handler.initialize(design_conf)

nq = handler.getModel().nq
nv = handler.getModel().nv
T = 100

x0 = handler.getState()
nu = handler.getModel().nv - 6 + len(handler.getFeetNames()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -handler.getMass() / len(handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref, np.zeros(nv - 6)))

w_x = np.array(
    [
        0,
        0,
        1000,
        1000,
        1000,
        1000,  # Base pos/ori
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,  # Left leg
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,
        0.1,  # Right leg
        1,
        1000,  # Torso
        1,
        1,
        10,
        10,  # Left arm
        1,
        1,
        10,
        10,  # Right arm
        0.1,
        0.1,
        0.1,
        1000,
        1000,
        1000,  # Base pos/ori vel
        1,
        1,
        1,
        1,
        1,
        1,  # Left leg vel
        1,
        1,
        1,
        1,
        1,
        1,  # Right leg vel
        0.1,
        100,  # Torso vel
        10,
        10,
        10,
        10,  # Left arm vel
        10,
        10,
        10,
        10,  # Right arm vel
    ]
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
    x0=handler.getState(),
    u0=u0,
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
)

problem = KinodynamicsProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getState(), T, 6, gravity[2])

T_ds = 20
T_ss = 80

mpc_conf = dict(
    totalSteps=3,
    ddpIteration=1,
    min_force=150,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-5,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    T=100,
    x_translation=0.0,
    y_translation=0.0,
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
for s in range(total_steps):
    contact_phases += (
        [contact_phase_left] * T_ss
        + [contact_phase_double] * T_ds
        + [contact_phase_right] * T_ss
        + [contact_phase_double] * T_ds
    )

mpc.generateFullHorizon(contact_phases)
contact_ids = handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=handler.getState(),
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
    kd=0,
    w_force=10000,
    w_acc=1,
    verbose=False,
)

qp = IDSolver()
qp.initialize(id_conf, handler.getModel())

weights_ID = [1, 10000]  # Acceleration, forces
mu = 0.8
Lfoot = 0.1
Wfoot = 0.075
force_size = 6
ID_solver = IDSolverPython(
    handler.getModel(), weights_ID, 2, mu, Lfoot, Wfoot, contact_ids, force_size, False
)

print(mpc.getSolver().results)
# exit()
""" Initialize simulation"""
device = BulletRobot(
    design_conf["controlled_joints_names"],
    modelPath,
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
for t in range(Tmpc):
    # print("Time " + str(t))
    LF_takeoffs = mpc.getFootTakeoffTimings("left_sole_link")
    RF_takeoffs = mpc.getFootTakeoffTimings("right_sole_link")
    LF_lands = mpc.getFootLandTimings("left_sole_link")
    RF_lands = mpc.getFootLandTimings("right_sole_link")

    LF_land = -1 if LF_lands == [] else LF_lands[0]
    RF_land = -1 if RF_lands == [] else RF_lands[0]
    LF_takeoff = -1 if LF_takeoffs == [] else LF_takeoffs[0]
    RF_takeoff = -1 if RF_takeoffs == [] else RF_takeoffs[0]
    print(
        "takeoff_RF = " + str(RF_takeoff) + ", landing_RF = ",
        str(RF_land) + ", takeoff_LF = " + str(LF_takeoff) + ", landing_LF = ",
        str(LF_land),
    )

    mpc.iterate(q_current, v_current)
    a0 = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[nv:]
    )
    contact_states = (
        mpc.getTrajOptProblem().stages[0].dynamics.differential_dynamics.contact_states
    )

    if t == 70:
        for s in range(T):
            device.resetState(mpc.xs[s][:nq])
            time.sleep(0.1)
            print("s = " + str(s))
        exit()
    # print("Left " + str(contact_states[0]) + ", right " + str(contact_states[1]))
    for j in range(10):
        q_current, v_current = device.measureState()
        x_measured = np.concatenate((q_current, v_current))

        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        state_diff = handler.difference(x_measured, mpc.xs[0])
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
        qp.solve_qp(
            handler.getData(),
            contact_states,
            v_current,
            a0,
            forces,
            handler.getMassMatrix(),
        )
        new_acc, new_forces, torque_qp = ID_solver.solve(
            mpc.getHandler().getData(),
            contact_states,
            x_measured[nq:],
            a0,
            forces,
            handler.getMassMatrix(),
        )
        solved_acc = qp.solved_acc
        solved_forces = qp.solved_forces
        solved_torque = qp.solved_torque
        device.execute(solved_torque)
