import numpy as np
import matplotlib.pyplot as plt
import example_robot_data

from simple_mpc import RobotHandler, KinodynamicsProblem, MPC

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
nu = handler.get_rmodel().nv - 6 + len(handler.get_ee_names()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -handler.get_mass() / len(handler.get_ee_names()) * gravity[2]
u0 = np.concatenate((fref, fref, np.zeros(handler.get_rmodel().nv - 6)))

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
        np.ones(handler.get_rmodel().nv - 6) * 1e-4,
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
    x0=handler.get_x0(),
    u0=u0,
    DT=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=6,
    w_frame=np.eye(6) * w_LFRF,
    umin=-handler.get_rmodel().effortLimit[6:],
    umax=handler.get_rmodel().effortLimit[6:],
    qmin=handler.get_rmodel().lowerPositionLimit[7:],
    qmax=handler.get_rmodel().upperPositionLimit[7:],
)

problem = KinodynamicsProblem(handler)
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

mpc.generateFullHorizon(contact_phases)
