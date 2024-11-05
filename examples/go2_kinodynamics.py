import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, KinodynamicsProblem, MPC, IDSolver
import example_robot_data
import pinocchio as pin
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
)
handler = RobotHandler()
handler.initialize(design_conf)

force_size = 3
nk = len(handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -handler.getMass() / nk * gravity[2]
u0 = np.concatenate((fref, fref, fref, fref, np.zeros(handler.getModel().nv - 6)))


w_basepos = [0, 0, 0, 1000, 1000, 0]
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
        np.ones(handler.getModel().nv - 6) * 1e-4,
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
    DT=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    w_centder=w_centder,
    gravity=gravity,
    force_size=3,
    w_frame=np.eye(3) * w_LFRF,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.01,
    Wfoot=0.01,
    kinematics_limits=True,
    force_cone=True,
)
T = 50

dynproblem = KinodynamicsProblem(handler)
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
contact_phases = [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FL] * T_ss
contact_phases += [contact_phase_quadru] * T_ds
contact_phases += [contact_phase_lift_FR] * T_ss

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
    verbose=False,
)

qp = IDSolver()
qp.initialize(id_conf, handler.getModel())

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
v = np.zeros(6)
v[5] = 0.4
mpc.setVelocityBase(v)
for t in range(5000):
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

    mpc.iterate(q_current, v_current)

    a0 = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[nv:]
    )
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

    for j in range(10):
        # time.sleep(0.01)
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        state_diff = mpc.getHandler().difference(x_measured, mpc.xs[0])
        mpc.getHandler().updateState(q_current, v_current, True)

        a0[6:] = (
            mpc.us[0][nk * force_size :]
            - 1
            * mpc.getSolver().results.controlFeedbacks()[0][nk * force_size :]
            @ state_diff
        )
        forces = (
            mpc.us[0][: nk * force_size]
            - 1
            * mpc.getSolver().results.controlFeedbacks()[0][: nk * force_size]
            @ state_diff
        )
        # a0[6:] = mpc.us[0][nk * force_size :]
        # forces = mpc.us[0][: nk * force_size]

        qp.solve_qp(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            a0,
            forces,
            mpc.getHandler().getMassMatrix(),
        )

        device.execute(qp.solved_torque)
