import numpy as np
import matplotlib.pyplot as plt
import example_robot_data

from simple_mpc import RobotHandler, CentroidalProblem, MPC, IKIDSolver

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

x0 = np.zeros(9)
x0[:3] = handler.getComPosition()
nu = handler.getModel().nv - 6 + len(handler.getFeetNames()) * 6

gravity = np.array([0, 0, -9.81])
fref = np.zeros(6)
fref[2] = -handler.getMass() / len(handler.getFeetNames()) * gravity[2]
u0 = np.concatenate((fref, fref))

w_control_linear = np.ones(3) * 0.001
w_control_angular = np.ones(3) * 0.1
w_u = np.diag(
    np.concatenate(
        (w_control_linear, w_control_angular, w_control_linear, w_control_angular)
    )
)
w_linear_mom = np.diag(np.array([0.01, 0.01, 100]))
w_linear_acc = 0.01 * np.eye(3)
w_angular_mom = np.diag(np.array([0.1, 0.1, 1000]))
w_angular_acc = 0.01 * np.eye(3)

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
    force_size=6,
)

problem = CentroidalProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getState(), T, 6, gravity[2])

T_ds = 20
T_ss = 80

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
    T_fly=T_ss,
    T_contact=T_ds,
    T=T,
    x_translation=0.1,
    y_translation=0.1,
)

mpc = MPC(x0, u0)
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
g_q = np.array(
    [
        0,
        0,
        0,
        100,
        100,
        100,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        10,
        10,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
        100,
    ]
)

g_p = np.array([400, 400, 400, 400, 400, 400])
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
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
    w_qref=500,
    w_footpose=50000,
    w_centroidal=10,
    w_baserot=1000,
    w_force=100,
)

contact_states = [True, True]
forces = np.array([0, 0, 400, 0, 0, 0, 0, 0, 400, 0, 0, 0])
foot_refs = [handler.getFootPose(0), handler.getFootPose(1)]
foot_refs_next = [handler.getFootPose(0), handler.getFootPose(1)]
dH = np.random.rand(6)
M = handler.getMassMatrix()
qp = IKIDSolver(ikid_conf, handler.getModel())
qp.solve_qp(
    handler.getData(),
    contact_states,
    handler.getState(),
    forces,
    foot_refs,
    foot_refs_next,
    dH,
    M,
)

print(qp.solved_acc)
