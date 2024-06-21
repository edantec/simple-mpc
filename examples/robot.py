import numpy as np
import matplotlib.pyplot as plt
import example_robot_data
from aligator import (
    ContactMap,
)
from simple_mpc import RobotHandler, Problem, FullDynamicsProblem, MPC

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

contact_sequence = []
force_sequence = []
for t in range(T):
    contact_state = [True, True]
    contact_pose = [
        handler.get_ee_pose(0).translation,
        handler.get_ee_pose(1).translation,
    ]
    force_ref = {
        "left_sole_link": [0, 0, 400, 0, 0, 0],
        "right_sole_link": [0, 0, 400, 0, 0, 0],
    }
    contact_names = ["left_sole_link", "right_sole_link"]
    contact_sequence.append(ContactMap(contact_names, contact_state, contact_pose))
    force_sequence.append(force_ref)

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

problem_conf = dict(
    x0=handler.get_x0(),
    u0=np.zeros(nu),
    DT=0.01,
    w_x=np.diag(w_x_vec),
    w_u=np.eye(nu) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=np.array([0, 0, 9.81]),
    force_size=6,
    w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
    w_frame=np.eye(6) * 2000,
    umin=-handler.get_rmodel().effortLimit[6:],
    umax=handler.get_rmodel().effortLimit[6:],
    qmin=handler.get_rmodel().lowerPositionLimit[6:],
    qmax=handler.get_rmodel().upperPositionLimit[6:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
)

problem = FullDynamicsProblem(handler)
problem.initialize(problem_conf)

contact_sequence = []
force_sequence = []
fref = np.array([0, 0, handler.get_mass() * 9.81 / 2.0, 0, 0, 0])
for i in range(100):
    contact_names = ["left_sole_link", "right_sole_link"]
    contact_phase = [True, True]
    contact_pose = [
        handler.get_ee_pose(0).translation,
        handler.get_ee_pose(1).translation,
    ]
    contact_sequence.append(ContactMap(contact_names, contact_phase, contact_pose))
    force_sequence.append({"left_sole_link": fref, "right_sole_link": fref})

problem.create_stage(contact_sequence[0], force_sequence[0])
problem.create_problem(handler.get_x0(), contact_sequence, force_sequence)

mpc_conf = dict(
    totalSteps=4,
    T=100,
    ddpIteration=1,
    min_force=150,
    support_force=1000,
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=2,
)

u0 = np.zeros(handler.get_rmodel().nv - 6)
mpc = MPC(handler.get_x0(), u0)
mpc.initialize(mpc_conf, problem)
