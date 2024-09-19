import numpy as np
import matplotlib.pyplot as plt
import example_robot_data
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, KinodynamicsProblem, MPC, IDSolver
import pinocchio as pin

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)


def difference(x1, x2, model):
    dq = pin.difference(model, x1[: model.nq], x2[: model.nq])
    dv = x2[model.nq :] - x1[model.nq :]
    dx = np.concatenate((dq, dv))

    return dx


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

T = 100

x0 = handler.getState()
nu = handler.getModel().nv - 6 + len(handler.getFeetNames()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -handler.getMass() / len(handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref, np.zeros(handler.getModel().nv - 6)))

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
w_x = np.diag(w_x)
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
)

problem = KinodynamicsProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getState(), T, 6, gravity[2])

mpc_conf = dict(
    totalSteps=4,
    ddpIteration=1,
    min_force=150,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=2,
    swing_apex=0.15,
    T_fly=80,
    T_contact=20,
    T=100,
    x_translation=0.0,
    y_translation=0.0,
)

mpc = MPC(handler.getState(), u0)
mpc.initialize(mpc_conf, problem)

T_ds = 100
T_ss = 80

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
    kd=100,
    w_force=100,
    w_acc=100,
    verbose=False,
)

qp = IDSolver(id_conf, handler.getModel())

""" Initialize simulation"""
print("Initialize simu")
device = BulletRobot(
    handler.getControlledJointsIDs(),
    modelPath,
    URDF_FILENAME,
    1e-3,
    handler.getModelComplete(),
)
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])
device.initializeJoints(handler.getCompleteConfiguration())
q_current, v_current = device.measureState()

Tmpc = len(contact_phases)

for t in range(Tmpc):
    print("Time " + str(t))
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

    for j in range(10):
        x_measured = np.concatenate((q_current, v_current))

        current_torque = mpc.us[0] - mpc.K0 @ difference(
            handler.getModel(), x_measured, mpc.xs[0]
        )
        device.execute(current_torque)
