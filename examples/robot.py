import numpy as np
import matplotlib.pyplot as plt
import example_robot_data

from simple_mpc import RobotHandler, FullDynamicsProblem, MPC

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

T = 100

x0 = handler.get_x0()
nu = handler.get_rmodel().nv - 6
w_x_vec = np.array(
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
w_cent_lin = np.ones(3) * 0
w_cent_ang = np.array([0.1, 0.1, 100])
w_cent_ang = np.ones(3) * 0
w_forces_lin = np.ones(3) * 0.0001
w_forces_ang = np.ones(3) * 0.01

gravity = np.array([0, 0, -9.81])

problem_conf = dict(
    x0=handler.get_x0(),
    u0=np.zeros(nu),
    DT=0.01,
    w_x=np.diag(w_x_vec),
    w_u=np.eye(nu) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=6,
    w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
    w_frame=np.eye(6) * 2000,
    umin=-handler.get_rmodel().effortLimit[6:],
    umax=handler.get_rmodel().effortLimit[6:],
    qmin=handler.get_rmodel().lowerPositionLimit[7:],
    qmax=handler.get_rmodel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
)

problem = FullDynamicsProblem(handler)
problem.initialize(problem_conf)
problem.create_problem(handler.get_x0(), T, 6, gravity[2])

mpc_conf = dict(
    totalSteps=4,
    ddpIteration=1,
    min_force=150,
    support_force=-handler.get_mass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=2,
)

u0 = np.zeros(handler.get_rmodel().nv - 6)
mpc = MPC(handler.get_x0(), u0)
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

# mpc.generateFullHorizon(contact_phases)
