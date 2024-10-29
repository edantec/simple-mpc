import numpy as np
import example_robot_data
from bullet_robot import BulletRobot
from simple_mpc import RobotHandler, CentroidalProblem, MPC, IKIDSolver
import ndcurves
import time

URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)


class footTrajectory:
    def __init__(
        self,
        start_pose_left,
        start_pose_right,
        T_ss,
        T_ds,
        nsteps,
        swing_apex,
        x_forward,
        y_forward,
        foot_angle,
        y_gap,
        z_height,
    ):
        self.translationRight = np.array([x_forward, -y_gap - y_forward, z_height])
        self.translationLeft = np.array([x_forward, y_gap, z_height])
        self.rotationDiff = self.yawRotation(foot_angle)

        self.start_pose_left = start_pose_left
        self.start_pose_right = start_pose_right
        self.final_pose_left = start_pose_left
        self.final_pose_right = start_pose_right

        self.T_ds = T_ds
        self.T_ss = T_ss
        self.nsteps = nsteps
        self.swing_apex = swing_apex

    def updateForward(
        self,
        x_f_left,
        x_f_right,
        y_gap,
        y_forward,
        z_height_left,
        z_height_right,
        swing_apex,
    ):
        self.translationRight = np.array(
            [x_f_right, -y_gap - y_forward, z_height_right]
        )
        self.translationLeft = np.array([x_f_left, y_gap, z_height_left])
        self.swing_apex = swing_apex

    def updateTrajectory(
        self, takeoff_RF, takeoff_LF, land_RF, land_LF, LF_pose, RF_pose
    ):
        if land_LF < 0:
            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = LF_pose.copy()

        if land_RF < 0:
            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = RF_pose.copy()

        if takeoff_RF < self.T_ds and takeoff_RF >= 0:
            # print("Update right trajectory")
            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = LF_pose.copy()
            yawLeft = self.extractYaw(LF_pose.rotation)
            self.final_pose_right.translation += (
                self.yawRotation(yawLeft) @ self.translationRight
            )
            self.final_pose_right.rotation = (
                self.rotationDiff @ self.final_pose_right.rotation
            )

            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = self.final_pose_right.copy()
            yawRight = self.extractYaw(self.final_pose_right.rotation)
            self.final_pose_left.translation += (
                self.yawRotation(yawRight) @ self.translationLeft
            )

        if takeoff_LF < self.T_ds and takeoff_LF >= 0:
            # print("Update left trajectory")
            self.start_pose_left = LF_pose.copy()
            self.final_pose_left = RF_pose.copy()
            yawRight = self.extractYaw(RF_pose.rotation)
            self.final_pose_left.translation += (
                self.yawRotation(yawRight) @ self.translationLeft
            )

            self.start_pose_right = RF_pose.copy()
            self.final_pose_right = self.final_pose_left.copy()
            yawLeft = self.extractYaw(self.final_pose_left.rotation)
            self.final_pose_right.translation += (
                self.yawRotation(yawLeft) @ self.translationRight
            )
            self.final_pose_right.rotation = (
                self.rotationDiff @ self.final_pose_right.rotation
            )

        swing_trajectory_left = self.defineBezier(
            self.swing_apex, 0, 1, self.start_pose_left, self.final_pose_left
        )
        swing_trajectory_right = self.defineBezier(
            self.swing_apex, 0, 1, self.start_pose_right, self.final_pose_right
        )

        LF_refs = (
            self.foot_trajectory(
                self.nsteps,
                land_LF,
                self.start_pose_left,
                self.final_pose_left,
                swing_trajectory_left,
                self.T_ss,
            )
            if land_LF > -1
            else ([self.start_pose_left for i in range(self.nsteps)])
        )

        RF_refs = (
            self.foot_trajectory(
                self.nsteps,
                land_RF,
                self.start_pose_right,
                self.final_pose_right,
                swing_trajectory_right,
                self.T_ss,
            )
            if land_RF > -1
            else ([self.start_pose_right for i in range(self.nsteps)])
        )

        return LF_refs, RF_refs

    def defineBezier(
        self, height, time_init, time_final, placement_init, placement_final
    ):
        wps = np.zeros([3, 9])
        for i in range(4):  # init position. init vel,acc and jerk == 0
            wps[:, i] = placement_init.translation
        # compute mid point (average and offset along z)
        wps[:, 4] = (
            placement_init.translation * 3 / 4 + placement_final.translation * 1 / 4
        )
        wps[2, 4] += height
        for i in range(5, 9):  # final position. final vel,acc and jerk == 0
            wps[:, i] = placement_final.translation
        translation = ndcurves.bezier(wps, time_init, time_final)
        pBezier = ndcurves.piecewise_SE3(
            ndcurves.SE3Curve(
                translation, placement_init.rotation, placement_final.rotation
            )
        )
        return pBezier

    def foot_trajectory(
        self,
        T,
        time_to_land,
        initial_pose,
        final_pose,
        trajectory_swing,
        TsingleSupport,
    ):
        placement = []
        for t in range(time_to_land, time_to_land - T, -1):
            if t <= 0:
                placement.append(final_pose)
            elif t > TsingleSupport:
                placement.append(initial_pose)
            else:
                swing_pose = initial_pose.copy()
                swing_pose.translation = trajectory_swing.translation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                swing_pose.rotation = trajectory_swing.rotation(
                    float(TsingleSupport - t) / float(TsingleSupport)
                )
                placement.append(swing_pose)

        return placement

    def yawRotation(self, yaw):
        Ro = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        return Ro

    def extractYaw(self, Ro):
        return np.arctan2(Ro[1, 0], Ro[0, 0])


def scan_list(list_Fs):
    for i in range(len(list_Fs)):
        list_Fs[i] -= 1
    if len(list_Fs) > 0 and list_Fs[0] == -1:
        list_Fs.remove(list_Fs[0])


def update_timings(land_LFs, land_RFs, takeoff_LFs, takeoff_RFs):
    scan_list(land_LFs)
    scan_list(land_RFs)
    scan_list(takeoff_LFs)
    scan_list(takeoff_RFs)
    land_LF = -1
    land_RF = -1
    takeoff_LF = -1
    takeoff_RF = -1
    if len(land_LFs) > 0:
        land_LF = land_LFs[0]
    if len(land_RFs) > 0:
        land_RF = land_RFs[0]
    if len(takeoff_LFs) > 0:
        takeoff_LF = takeoff_LFs[0]
    if len(takeoff_RFs) > 0:
        takeoff_RF = takeoff_RFs[0]
    return takeoff_RF, takeoff_LF, land_RF, land_LF


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
w_vbase = np.eye(6) * 10

problem_conf = dict(
    x0=x0,
    u0=u0,
    DT=0.01,
    w_u=w_u,
    w_linear_mom=w_linear_mom,
    w_angular_mom=w_angular_mom,
    w_linear_acc=w_linear_acc,
    w_angular_acc=w_angular_acc,
    w_vbase=w_vbase,
    gravity=gravity,
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
    force_size=6,
)
T = 100

problem = CentroidalProblem(handler)
problem.initialize(problem_conf)
problem.createProblem(handler.getCentroidalState(), T, 6, gravity[2])

T_ds = 20
T_ss = 80

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
    dt=0.01,
)

mpc = MPC()
mpc.initialize(mpc_conf, problem)

""" Define contact sequence throughout horizon"""
total_steps = 2
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
contact_phases += [contact_phase_left] * T_ss
contact_phases += [contact_phase_double] * T_ds
contact_phases += [contact_phase_right] * T_ss

mpc.generateCycleHorizon(contact_phases)

""" Initialize inverse dynamics QP """
g_basepos = [0, 0, 0, 10, 10, 10]
g_legpos = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
g_torsopos = [1, 1]
g_armpos = [10, 10, 100, 10]

g_q = np.array(g_basepos + g_legpos * 2 + g_torsopos + g_armpos * 2) * 10

g_p = np.array([2000, 2000, 2000, 2000, 2000, 2000])
g_b = np.array([10, 10, 10])

Kp_gains = [g_q, g_p, g_b]
Kd_gains = [2 * np.sqrt(g_q), 2 * np.sqrt(g_p), 2 * np.sqrt(g_b)]
contact_ids = handler.getFeetIds()
fixed_frame_ids = [handler.getRootId(), handler.getModel().getFrameId("torso_2_link")]
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
    verbose=False,
)

qp = IKIDSolver()
qp.initialize(ikid_conf, handler.getModel())

""" Initialize simulation"""
device = BulletRobot(
    design_conf["controlled_joints_names"],
    modelPath + "/talos_data/robots/",
    URDF_FILENAME,
    1e-3,
    handler.getCompleteModel(),
)
device.initializeJoints(handler.getCompleteConfiguration())
device.changeCamera(1.0, 90, -5, [1.5, 0, 1])
q_current, v_current = device.measureState()
nq = mpc.getHandler().getModel().nq
nv = mpc.getHandler().getModel().nv

x_measured = mpc.getHandler().shapeState(q_current, v_current)
q_current = x_measured[:nq]
v_current = x_measured[nq:]

Tmpc = len(contact_phases)
nk = 2
force_size = 6
x_centroidal = mpc.getHandler().getCentroidalState()

device.showTargetToTrack(
    mpc.getHandler().getFootPose("left_sole_link"),
    mpc.getHandler().getFootPose("right_sole_link"),
)
import pinocchio as pin

v = pin.Motion.Zero()
v.linear[0] = 0.1
v.angular[2] = 0.0
mpc.setVelocityBase(v)
for t in range(600):
    # print("Time " + str(t))
    if t == 400:
        print("SWITCH TO STAND")
        mpc.switchToStand()

    land_LF = mpc.getFootLandCycle("left_sole_link")
    land_RF = mpc.getFootLandCycle("right_sole_link")
    takeoff_LF = mpc.getFootTakeoffCycle("left_sole_link")
    takeoff_RF = mpc.getFootTakeoffCycle("right_sole_link")

    print(
        "takeoff_RF = " + str(takeoff_RF) + ", landing_RF = ",
        str(land_RF) + ", takeoff_LF = " + str(takeoff_LF) + ", landing_LF = ",
        str(land_LF),
    )

    mpc.iterate(q_current, v_current)

    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
    )

    contact_states = (
        mpc.getTrajOptProblem()
        .stages[0]
        .dynamics.differential_dynamics.contact_map.contact_states.tolist()
    )
    foot_ref = [mpc.getReferencePose(0, name) for name in handler.getFeetNames()]
    foot_ref_next = [mpc.getReferencePose(1, name) for name in handler.getFeetNames()]
    dH = (
        mpc.getSolver()
        .workspace.problem_data.stage_data[0]
        .dynamics_data.continuous_data.xdot[3:9]
    )
    qp.computeDifferences(
        mpc.getHandler().getData(), x_measured, foot_ref, foot_ref_next
    )
    for j in range(10):
        time.sleep(0.001)
        q_current, v_current = device.measureState()
        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        mpc.getHandler().updateState(q_current, v_current, True)
        x_centroidal = mpc.getHandler().getCentroidalState()
        state_diff = mpc.xs[0] - x_centroidal

        forces = (
            mpc.us[0][: nk * force_size]
            - 1 * mpc.getSolver().results.controlFeedbacks()[0] @ state_diff
        )

        qp.solve_qp(
            mpc.getHandler().getData(),
            contact_states,
            v_current,
            forces,
            dH,
            mpc.getHandler().getMassMatrix(),
        )

        device.execute(qp.solved_torque)
