import numpy as np
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, FullDynamicsProblem, MPC
import example_robot_data
import time
import pinocchio as pin
from aligator import (
    manifolds,
    dynamics,
    constraints,
)
import aligator
import copy

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
        np.array([0.2, 0.15, 0.]),
        np.array([0.2, -0.15, 0.]),
        np.array([-0.2, 0.15, 0.]),
        np.array([-0.2, -0.15, 0.]),
    ]
)
handler = RobotHandler()
handler.initialize(design_conf)
rmodel = handler.getModel()

nv = handler.getModel().nv
nu = nv - 6
force_size = 3
nk = len(handler.getFeetNames())
gravity = np.array([0, 0, -9.81])
fref = np.zeros(force_size)
fref[2] = -handler.getMass() / nk * gravity[2]
u0 = np.zeros(handler.getModel().nv - 6)

w_basepos = [0, 0, 0, 0, 0, 0]
w_legpos = [1, 1, 1]

w_basevel = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
w_legvel = [0.1, 0.1, 0.1]
w_x = np.diag(np.array(w_basepos + w_legpos * 4 + w_basevel + w_legvel * 4)) * 0.1

w_u = np.eye(nu) * 1e-4
w_cent_lin = np.array([0, 0, 0])
w_cent_ang = np.array([1, 1, 1])
w_cent = np.diag(np.concatenate((w_cent_lin, w_cent_ang)))
w_forces = np.diag(np.array([1e-6, 1e-6, 0]))
v_ref = pin.Motion()
v_ref.np[:] = 0.0

problem_conf = dict(
    x0=handler.getState(),
    u0=u0,
    DT=0.01,
    w_x=w_x,
    w_u=w_u,
    w_cent=w_cent,
    gravity=gravity,
    force_size=3,
    w_forces=w_forces,
    w_frame=np.eye(3) * 0,
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
nsteps = 100

dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(handler.getState(), nsteps, force_size, gravity[2])

T_ground = 100
T_fly = 30

""" Define contact sequence throughout horizon"""
contact_phase_quadru = {
    "FL_foot": True,
    "FR_foot": True,
    "RL_foot": True,
    "RR_foot": True,
}
contact_phase_lift = {
    "FL_foot": False,
    "FR_foot": False,
    "RL_foot": False,
    "RR_foot": False,
}
contact_phases = [contact_phase_quadru] * T_ground
contact_phases += [contact_phase_lift] * T_fly
contact_phases += [contact_phase_quadru] * T_ground

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
nq = handler.getModel().nq
nv = handler.getModel().nv

x_measured = handler.getState()
q_current = x_measured[:nq]
v_current = x_measured[nq:]

Tmpc = len(contact_phases)

for t in range(Tmpc * 20):
    print("Time " + str(t))
    """ if t == 250:
        for s in range(nsteps):
            device.resetState(mpc.xs[s][:rmodel.nq])
            time.sleep(0.1)
            print("s = " + str(s))
        exit()  """

    mpc.iterate(q_current, v_current)
    for j in range(10):
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        current_torque = mpc.us[0] - mpc.K0 @ handler.difference(x_measured, mpc.xs[0])
        device.execute(current_torque)
