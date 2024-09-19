import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
import pinocchio as pin
from simple_mpc import RobotHandler, FullDynamicsProblem, MPC

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


def shapeState(q_current, v_current, nq, nxq, cj_ids):
    x_internal = np.zeros(nxq)
    x_internal[:7] = q_current[:7]
    x_internal[nq : nq + 6] = v_current[:6]
    i = 0
    for jointID in cj_ids:
        if jointID > 1:
            x_internal[i + 7] = q_current[jointID + 5]
            x_internal[nq + i + 6] = v_current[jointID + 4]
            i += 1

    return x_internal


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
nu = handler.getModel().nv - 6
w_x = np.array(
    [
        0,
        0,
        0,
        100,
        100,
        100,  # Base pos/ori
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
        10,
        10,  # Torso
        1,
        1,
        1,
        1,  # Left arm
        1,
        1,
        1,
        1,  # Right arm
        1,
        1,
        1,
        1,
        1,
        1,  # Base pos/ori vel
        0.1,
        0.1,
        0.1,
        0.1,
        0.01,
        0.01,  # Left leg vel
        0.1,
        0.1,
        0.1,
        0.1,
        0.01,
        0.01,  # Right leg vel
        10,
        10,  # Torso vel
        1,
        1,
        1,
        1,  # Left arm vel
        1,
        1,
        1,
        1,  # Right arm vel
    ]
)
w_cent_lin = np.array([0.0, 0.0, 10])
w_cent_ang = np.array([0.0, 0.0, 10])
w_cent_ang = np.ones(3) * 0
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])
w_forces_ang = np.ones(3) * 0.0001

gravity = np.array([0, 0, -9.81])

problem_conf = dict(
    x0=handler.getState(),
    u0=np.zeros(nu),
    DT=0.01,
    w_x=np.diag(w_x),
    w_u=np.eye(nu) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=6,
    w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
    w_frame=np.eye(6) * 2000,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
)

problem = FullDynamicsProblem(handler)
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
    x_translation=0.1,
    y_translation=0.1,
)

u0 = np.zeros(handler.getModel().nv - 6)
mpc = MPC()
mpc.initialize(mpc_conf, problem)

T_ds = 20
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


""" Initialize simulation"""
print("Initialize simu")
device = BulletRobot(
    handler.getControlledJointsIDs(),
    modelPath,
    URDF_FILENAME,
    1e-3,
    handler.getCompleteModel(),
)
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])
device.initializeJoints(handler.getCompleteConfiguration())
qc_current, vc_current = device.measureState()

x_measured = shapeState(
    qc_current, vc_current, nq, nq + nv, handler.getControlledJointsIDs()
)

q_current = x_measured[:nq]
v_current = x_measured[nq:]

Tmpc = len(contact_phases)
nk = 2
force_size = 6

for i in range(Tmpc):
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

    mpc.iterate(x_measured[:nq], x_measured[nq:])
    for j in range(10):
        qc_current, vc_current = device.measureState()
        x_measured = shapeState(
            qc_current, vc_current, nq, nq + nv, handler.getControlledJointsIDs()
        )

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        state_diff = difference(x_measured, mpc.xs[0], handler.getModel())
        solved_torque = mpc.us[0] - mpc.K0 @ difference(
            x_measured, mpc.xs[0], handler.getModel()
        )
        device.execute(solved_torque)
