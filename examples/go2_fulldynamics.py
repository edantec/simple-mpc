import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, FullDynamicsProblem, MPC
import example_robot_data

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
)
handler = RobotHandler()
handler.initialize(design_conf)

x0 = np.zeros(9)
x0[:3] = handler.getComPosition()
force_size = 3
nk = len(handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -handler.getMass() / nk * gravity[2]
u0 = np.zeros(handler.getModel().nv - 6)

w_x = np.array(
    [
        0,
        0,
        0,
        100,
        100,
        100,  # Base pos/ori
        1,
        1,
        1,  # FL
        1,
        1,
        1,  # FR
        1,
        1,
        1,  # RL
        1,
        1,
        1,  # RR
        1,
        1,
        1,
        1,
        1,
        1,  # Base pos/ori vel
        0.1,
        0.1,
        0.1,  # FL
        0.1,
        0.1,
        0.1,  # FR
        0.1,
        0.1,
        0.1,  # RL
        0.1,
        0.1,
        0.1,  # RR
    ]
)
w_cent_lin = np.array([0.0, 0.0, 10])
w_cent_ang = np.array([0.0, 0.0, 10])
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])

problem_conf = dict(
    x0=handler.getState(),
    u0=u0,
    DT=0.01,
    w_x=np.diag(w_x),
    w_u=np.eye(u0.size) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=3,
    w_forces=np.diag(w_forces_lin),
    w_frame=np.eye(3) * 2000,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
)
T = 50

dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(handler.getState(), T, force_size, gravity[2])

T_ds = 10
T_ss = 40

mpc_conf = dict(
    ddpIteration=1,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=1,
    swing_apex=0.15,
    T_fly=T_ss,
    T_contact=T_ds,
    T=T,
    x_translation=0.0,
    y_translation=0.0,
)

mpc = MPC()
mpc.initialize(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
total_steps = 2
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
    "RR_foot": True,
}
contact_phase_lift_FR = {
    "FL_foot": True,
    "FR_foot": False,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phases = [contact_phase_quadru] * T_ds
for s in range(total_steps):
    contact_phases += (
        [contact_phase_lift_FL] * T_ss
        + [contact_phase_quadru] * T_ds
        + [contact_phase_lift_FR] * T_ss
        + [contact_phase_quadru] * T_ds
    )

contact_phases += [contact_phase_quadru] * T * 2

mpc.generateFullHorizon(contact_phases)

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
device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])
q_current, v_current = device.measureState()
nq = mpc.getHandler().getModel().nq
nv = mpc.getHandler().getModel().nv

x_measured = mpc.getHandler().shapeState(q_current, v_current)
q_current = x_measured[:nq]
v_current = x_measured[nq:]

Tmpc = len(contact_phases)
for t in range(Tmpc):
    # print("Time " + str(t))
    LF_takeoffs = mpc.getFootTakeoffTimings("FL_foot")
    RF_takeoffs = mpc.getFootTakeoffTimings("FR_foot")
    LF_lands = mpc.getFootLandTimings("FL_foot")
    RF_lands = mpc.getFootLandTimings("FR_foot")

    LF_land = -1 if LF_lands.tolist() == [] else LF_lands[0]
    RF_land = -1 if RF_lands.tolist() == [] else RF_lands[0]
    LF_takeoff = -1 if LF_takeoffs.tolist() == [] else LF_takeoffs[0]
    RF_takeoff = -1 if RF_takeoffs.tolist() == [] else RF_takeoffs[0]
    print(
        "takeoff_RF = " + str(RF_takeoff) + ", landing_RF = ",
        str(RF_land) + ", takeoff_LF = " + str(LF_takeoff) + ", landing_LF = ",
        str(LF_land),
    )

    mpc.iterate(q_current, v_current)

    for j in range(10):
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        current_torque = mpc.us[0] - mpc.K0 @ handler.difference(x_measured, mpc.xs[0])
        device.execute(current_torque)
