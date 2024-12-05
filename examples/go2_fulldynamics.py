import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, FullDynamicsProblem, MPC
import example_robot_data
import time

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
        np.array([0.2, 0.15, 0.]),
        np.array([0.2, -0.15, 0.]),
        np.array([-0.2, 0.15, 0.]),
        np.array([-0.2, -0.15, 0.]),
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

w_basepos = [0, 0, 10, 10, 10, 0]
w_legpos = [1, 1, 1]

w_basevel = [10, 10, 10, 10, 10, 10]
w_legvel = [1, 1, 1]
w_x = np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)
w_cent_lin = np.array([0.0, 0.0, 1])
w_cent_ang = np.array([0.0, 0.0, 1])
w_forces_lin = np.array([0.001, 0.001, 0.001])
w_frame = np.diag(np.array([1000, 1000, 1000]))

problem_conf = dict(
    DT=0.01,
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
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    torque_limits=True,
    kinematics_limits=True,
    force_cone=True,
)
T = 50

dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(handler.getState(), T, force_size, gravity[2])

""" T_ds = 80
T_ss = 30 """

T_ds = 5
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
device.changeCamera(1.0, 60, -15, [0.6, -0.2, 0.5])
q_current, v_current = device.measureState()
nq = mpc.getModelHandler().getModel().nq
nv = mpc.getModelHandler().getModel().nv

x_measured = mpc.getModelHandler().shapeState(q_current, v_current)
q_current = x_measured[:nq]
v_current = x_measured[nq:]

""" device.showQuadrupedFeet(
    mpc.getDataHandler().getFootPose("FL_foot"),
    mpc.getDataHandler().getFootPose("FR_foot"),
    mpc.getDataHandler().getFootPose("RL_foot"),
    mpc.getDataHandler().getFootPose("RR_foot"),
) """
rmodel = handler.getModel()
a1 = mpc.getDataHandler().getData().oMf[rmodel.getFrameId("FL_thigh")]
a2 = mpc.getDataHandler().getData().oMf[rmodel.getFrameId("FR_thigh")]
a3 = mpc.getDataHandler().getData().oMf[rmodel.getFrameId("RL_thigh")]
a4 = mpc.getDataHandler().getData().oMf[rmodel.getFrameId("RR_thigh")]
a1.translation[2] = 0
a2.translation[2] = 0
a3.translation[2] = 0
a4.translation[2] = 0
device.showQuadrupedFeet(a1, a2, a3, a4)
Tmpc = len(contact_phases)

v = np.zeros(6)
v[0] = 0.2
mpc.setVelocityBase(v)
for t in range(10000):
    print("Time " + str(t))
    land_LF = mpc.getFootLandCycle("FL_foot")
    land_RF = mpc.getFootLandCycle("RL_foot")
    takeoff_LF = mpc.getFootTakeoffCycle("FL_foot")
    takeoff_RF = mpc.getFootTakeoffCycle("RL_foot")
    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    device.moveQuadrupedFeet(
        mpc.getReferencePose(0, "FL_foot").translation,
        mpc.getReferencePose(0, "FR_foot").translation,
        mpc.getReferencePose(0, "RL_foot").translation,
        mpc.getReferencePose(0, "RR_foot").translation,
    )

    mpc.iterate(q_current, v_current)
    if t == 500:
        mpc.switchToStand()
    if t == 700:
        mpc.switchToWalk(v)

    for j in range(10):
        # time.sleep(0.01)
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        current_torque = mpc.us[0] - 1 * mpc.K0 @ handler.difference(
            x_measured, mpc.xs[0]
        )
        device.execute(current_torque)
