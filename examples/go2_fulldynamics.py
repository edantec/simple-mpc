import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, FullDynamicsProblem, MPC
import example_robot_data
import time
from utils import save_trajectory, extract_forces

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
    hip_names=[
        "FL_thigh",
        "FR_thigh",
        "RL_thigh",
        "RR_thigh",
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
u0 = np.zeros(handler.getModel().nv - 6)

w_basepos = [0, 0, 0, 0, 0, 0]
w_legpos = [10, 10, 10]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_cent_lin = np.array([0.0, 0.0, 0])
w_cent_ang = np.array([0, 0, 0])
w_forces_lin = np.array([0.001, 0.001, 0.001])
w_frame = np.diag(np.array([1000, 1000, 1000]))

dt = 0.01
problem_conf = dict(
    DT=dt,
    w_x=np.diag(w_x),
    w_u=np.eye(u0.size) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=3,
    w_forces=np.diag(w_forces_lin),
    w_frame=w_frame,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    Kp_correction=np.array([0, 0, 0]),
    Kd_correction=np.array([100, 100, 100]),
    mu=1,
    Lfoot=0.01,
    Wfoot=0.01,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=False,
)
T = 50

dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(handler.getState(), T, force_size, gravity[2], False)

T_ds = 10
T_ss = 30
N_simu = int(0.01 / 0.001)
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
contact_phases = [contact_phase_quadru] * int(T_ds / 2)
contact_phases += [contact_phase_lift_FL] * T_ss
contact_phases += [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FR] * T_ss
contact_phases += [contact_phase_quadru] * int(T_ds / 2)

""" contact_phases = [contact_phase_quadru] * int(T_ds / 2)
contact_phases += [contact_phase_lift] * T_ss
contact_phases += [contact_phase_quadru] * int(T_ds / 2) """

mpc.generateCycleHorizon(contact_phases)

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
for i in range(40):
    device.setFrictionCoefficients(i, 10, 0)
#device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])
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
Tmpc = len(contact_phases)



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
com_measured = []
solve_time = []
L_measured = []

v = np.zeros(6)
v[0] = 0.2
mpc.setVelocityBase(v)
for t in range(2000):
    print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    """ print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    ) """

    if t == 1000:
        for s in range(T):
            device.resetState(mpc.xs[s][:nq])
            #device.resetState(state_ref[s])
            time.sleep(0.1)
            print("s = " + str(s))
        exit()

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    start = time.time()
    mpc.iterate(q_current, v_current)
    end = time.time()
    solve_time.append(end - start)

    """ if t == 500:
        mpc.switchToStand()
    if t == 700:
        mpc.switchToWalk(v) """

    FL_f, FR_f, RL_f, RR_f, contact_states = extract_forces(mpc.getTrajOptProblem(), mpc.getSolver().workspace, 0)
    force_FL.append(FL_f)
    force_FR.append(FR_f)
    force_RL.append(RL_f)
    force_RR.append(RR_f)

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

    for j in range(N_simu):
        # time.sleep(0.01)
        u_interp = (N_simu - j) / N_simu * mpc.us[0] + j / N_simu * mpc.us[1]
        x_interp = (N_simu - j) / N_simu * mpc.xs[0] + j / N_simu * mpc.xs[1]
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))
        mpc.getHandler().updateState(q_current, v_current, True)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        current_torque = u_interp - 1. * mpc.Ks[0] @ handler.difference(
            x_measured, x_interp
        )

        device.execute(current_torque)

        u_multibody.append(current_torque)
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

""" save_trajectory(x_multibody, u_multibody, com_measured, force_FL, force_FR, force_RL, force_RR, solve_time,
                FL_measured, FR_measured, RL_measured, RR_measured,
                FL_references, FR_references, RL_references, RR_references, L_measured, "fulldynamics") """
