"""
This script launches a locomotion MPC scheme which solves repeatedly an
optimal control problem based on the full dynamics model of the humanoid robot Talos.
The contacts forces are modeled as 6D wrenches.
"""

import numpy as np
import time
from bullet_robot import BulletRobot
import example_robot_data
from simple_mpc import MPC, FullDynamicsProblem, RobotHandler

# ####### CONFIGURATION  ############
# ### RobotWrapper
URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
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
nu = nv - 6

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
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])
w_forces_ang = np.ones(3) * 0.0001

gravity = np.array([0, 0, -9.81])

problem_conf = dict(
    x0=x0,
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

T = 100
dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(x0, T, 6, gravity[2])

""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0.1
y_forward = 0.0
foot_yaw = 0
y_gap = 0.18
x_depth = 0.0
T_ss = 80
T_ds = 20
nsteps = 100
totalSteps = 1
mpc_conf = dict(
    totalSteps=totalSteps,
    ddpIteration=1,
    min_force=150,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=2,
    swing_apex=0.15,
    x_translation=0.0,
    y_translation=0,
    T_fly=T_ss,
    T_contact=T_ds,
    T=nsteps,
)

mpc = MPC()
mpc.initialize(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
total_steps = totalSteps
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

Tmpc = len(contact_phases)
mpc.generateFullHorizon(contact_phases)
problem = mpc.getTrajOptProblem()

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

land_LF = -1
land_RF = -1
takeoff_LF = -1
takeoff_RF = -1
device.showTargetToTrack(
    mpc.getHandler().getFootPose("left_sole_link"),
    mpc.getHandler().getFootPose("right_sole_link"),
)
for t in range(Tmpc):
    print("Time " + str(t))
    land_LFs = mpc.getFootLandTimings("left_sole_link")
    land_RFs = mpc.getFootLandTimings("right_sole_link")
    takeoff_LFs = mpc.getFootTakeoffTimings("left_sole_link")
    takeoff_RFs = mpc.getFootTakeoffTimings("right_sole_link")

    if len(land_LFs) > 0:
        land_LF = land_LFs[0]
    else:
        land_LF == -1
    if len(land_RFs) > 0:
        land_RF = land_RFs[0]
    else:
        land_RF == -1
    if len(takeoff_LFs) > 0:
        takeoff_LF = takeoff_LFs[0]
    else:
        takeoff_LF == -1
    if len(takeoff_RFs) > 0:
        takeoff_RF = takeoff_RFs[0]
    else:
        takeoff_RF == -1

    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    mpc.iterate(q_current, v_current)
    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    for j in range(10):
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        current_torque = mpc.us[0] - mpc.K0 @ handler.difference(x_measured, mpc.xs[0])
        device.execute(current_torque)
