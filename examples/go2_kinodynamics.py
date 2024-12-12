import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, KinodynamicsOCP, MPC, IDSolver
import example_robot_data
import pinocchio as pin
import time
import copy
from utils import save_trajectory

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
    feet_to_base_trans=[
        np.array([0.17, 0.15, 0.]),
        np.array([0.17, -0.15, 0.]),
        np.array([-0.24, 0.15, 0.]),
        np.array([-0.24, -0.15, 0.]),
    ]
)
handler = RobotHandler()
handler.initialize(design_conf)

force_size = 3
nk = len(handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(handler.getModel().nv - 6)))


w_basepos = [0, 0, 100, 10, 10, 0]
w_legpos = [1, 1, 1]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_x = np.diag(w_x)
w_linforce = np.array([0.01, 0.01, 0.01])
w_u = np.concatenate(
    (
        w_linforce,
        w_linforce,
        w_linforce,
        w_linforce,
        np.ones(handler.getModel().nv - 6) * 1e-5,
    )
)
w_u = np.diag(w_u)
w_LFRF = 2000
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.1, 0.1, 10])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_centder_lin = np.ones(3) * 0.0
w_centder_ang = np.ones(3) * 0.1
w_centder = np.diag(np.concatenate((w_centder_lin, w_centder_ang)))

problem_conf = dict(
    timestep=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=False,
)
T = 50

dynproblem = KinodynamicsOCP(problem_conf, handler)
dynproblem.createProblem(handler.getState(), T, force_size, gravity[2], False)

T_ds = 10
T_ss = 30

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
    timestep=0.01,
)

mpc = MPC()
mpc.initialize(mpc_conf, dynproblem)

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
    "RR_foot": False,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": False,
}
contact_phases = [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FL] * T_ss
contact_phases += [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FR] * T_ss

""" contact_phases = [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift] * T_ss
contact_phases += [contact_phase_quadru] * T_ds """
mpc.generateCycleHorizon(contact_phases)

""" Initialize whole-body inverse dynamics QP"""
contact_ids = handler.getFeetIds()
id_conf = dict(
    contact_ids=contact_ids,
    x0=handler.getState(),
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    force_size=3,
    kd=0,
    w_force=100,
    w_acc=1,
    w_tau=0,
    verbose=False,
)

qp = IDSolver(id_conf, handler.getModel())

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

force_FL = []
force_FR = []
force_RL = []
force_RR = []
FL_measured = []
FR_measured = []
RL_measured = []
RR_measured = []
FL_references = []
FR_references = []
RL_references = []
RR_references = []
x_multibody = []
u_multibody = []
u_riccati = []
com_measured = []
solve_time = []
L_measured = []

N_simu = 10
v = np.zeros(6)
v[0] = 0.2
mpc.velocity_base = v
for t in range(300):
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

    """ if t == 500:
        mpc.switchToStand()
    if t == 800:
        v = np.zeros(6)
        v[0] = 0.2
        mpc.switchToWalk(v) """
    start = time.time()
    mpc.iterate(q_current, v_current)
    end = time.time()
    solve_time.append(end - start)

    force_FL.append(mpc.us[0][:3])
    force_FR.append(mpc.us[0][3:6])
    force_RL.append(mpc.us[0][6:9])
    force_RR.append(mpc.us[0][9:12])

    FL_measured.append(mpc.getHandler().getFootPose("FL_foot").translation)
    FR_measured.append(mpc.getHandler().getFootPose("FR_foot").translation)
    RL_measured.append(mpc.getHandler().getFootPose("RL_foot").translation)
    RR_measured.append(mpc.getHandler().getFootPose("RR_foot").translation)
    FL_references.append(mpc.getReferencePose(0, "FL_foot").translation)
    FR_references.append(mpc.getReferencePose(0, "FR_foot").translation)
    RL_references.append(mpc.getReferencePose(0, "RL_foot").translation)
    RR_references.append(mpc.getReferencePose(0, "RR_foot").translation)
    com_measured.append(mpc.getHandler().getComPosition().copy())
    L_measured.append(mpc.getHandler().getData().hg.angular.copy())

    a0 = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[nv:]
    )
    a1 = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[1]
        .dynamics_data.continuous_data.xdot[nv:]
    )
    a0[6:] = mpc.us[0][nk * force_size :]
    a1[6:] = mpc.us[1][nk * force_size :]
    forces0 = mpc.us[0][: nk * force_size]
    forces1 = mpc.us[1][: nk * force_size]
    contact_states = (
        mpc.getTrajOptProblem().stages[0].dynamics.differential_dynamics.contact_states
    )

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    """ if t == 590:
        for s in range(T):
            device.resetState(mpc.xs[s][:handler.getModel().nq])
            time.sleep(0.1)
            print("s = " + str(s))
            device.moveQuadrupedFeet(
                mpc.getReferencePose(s, "FL_foot").translation,
                mpc.getReferencePose(s, "FR_foot").translation,
                mpc.getReferencePose(s, "RL_foot").translation,
                mpc.getReferencePose(s, "RR_foot").translation,
            )
        exit()  """

    for j in range(N_simu):
        # time.sleep(0.01)
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))
        mpc.getHandler().updateState(q_current, v_current, True)

        a_interp = (N_simu - j) / N_simu * a0 + j / N_simu * a1
        f_interp = (N_simu - j) / N_simu * forces0 + j / N_simu * forces1

        qp.solveQP(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            a_interp,
            np.zeros(12),
            f_interp,
            mpc.getHandler().getMassMatrix(),
        )

        device.execute(qp.solved_torque)
        u_multibody.append(copy.deepcopy(qp.solved_torque))
        x_multibody.append(x_measured)


force_FL = np.array(force_FL)
force_FR = np.array(force_FR)
force_RL = np.array(force_RL)
force_RR = np.array(force_RR)
solve_time = np.array(solve_time)
FL_measured = np.array(FL_measured)
FR_measured = np.array(FR_measured)
RL_measured = np.array(RL_measured)
RR_measured = np.array(RR_measured)
FL_references = np.array(FL_references)
FR_references = np.array(FR_references)
RL_references = np.array(RL_references)
RR_references = np.array(RR_references)
com_measured = np.array(com_measured)
L_measured = np.array(L_measured)

save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "kinodynamics")
