"""
This script launches a locomotion MPC scheme which solves repeatedly an
optimal control problem based on the full dynamics model of the humanoid robot Talos.
The contacts forces are modeled as 6D wrenches.
"""

import numpy as np
import time
from bullet_robot import BulletRobot
import example_robot_data
from simple_mpc import MPC, FullDynamicsProblem, RobotHandler
import ndcurves


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
URDF_FILENAME = "talos_reduced.urdf"
SRDF_FILENAME = "talos.srdf"
SRDF_SUBPATH = "/talos_data/srdf/" + SRDF_FILENAME
URDF_SUBPATH = "/talos_data/robots/" + URDF_FILENAME

modelPath = example_robot_data.getModelPath(URDF_SUBPATH)
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
nq = handler.getModel().nq
nv = handler.getModel().nv
nu = nv - 6

x0 = handler.getState()
nu = handler.getModel().nv - 6
w_x = np.array(
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
w_cent_lin = np.array([0.0, 0.0, 10])
w_cent_ang = np.array([0.0, 0.0, 10])
w_forces_lin = np.array([0.0001, 0.0001, 0.0001])
w_forces_ang = np.ones(3) * 0.0001

gravity = np.array([0, 0, -9.81])

problem_conf = dict(
    x0=x0,
    u0=np.zeros(nu),
    DT=0.01,
    w_x=np.diag(w_x),
    w_u=np.eye(nu) * 1e-4,
    w_cent=np.diag(np.concatenate((w_cent_lin, w_cent_ang))),
    gravity=gravity,
    force_size=6,
    w_forces=np.diag(np.concatenate((w_forces_lin, w_forces_ang))),
    w_frame=np.eye(6) * 2000,
    umin=-handler.getModel().effortLimit[6:],
    umax=handler.getModel().effortLimit[6:],
    qmin=handler.getModel().lowerPositionLimit[7:],
    qmax=handler.getModel().upperPositionLimit[7:],
    mu=0.8,
    Lfoot=0.1,
    Wfoot=0.075,
)

T = 100
dynproblem = FullDynamicsProblem(handler)
dynproblem.initialize(problem_conf)
dynproblem.createProblem(x0, T, 6, gravity[2])

""" Define feet trajectory """
T_ss = 80
T_ds = 20
totalSteps = 2
mpc_conf = dict(
    ddpIteration=1,
    support_force=-handler.getMass() * gravity[2],
    TOL=1e-4,
    mu_init=1e-8,
    max_iters=1,
    num_threads=8,
    swing_apex=0.15,
    x_translation=0.1,
    y_translation=0,
    T_fly=T_ss,
    T_contact=T_ds,
    T=T,
)

mpc = MPC()
mpc.initialize(mpc_conf, dynproblem)

""" Define contact sequence throughout horizon"""
total_steps = totalSteps
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
contact_phases += [contact_phase_double] * T * 2

Tmpc = len(contact_phases)
mpc.generateFullHorizon(contact_phases)
problem = mpc.getTrajOptProblem()

""" Initialize simulation"""
device = BulletRobot(
    design_conf["controlled_joints_names"],
    modelPath + "/talos_data/robots/",
    URDF_FILENAME,
    1e-3,
    handler.getCompleteModel(),
)
device.initializeJoints(handler.getCompleteConfiguration())
# device.changeCamera(1.0, 50, -15, [1.7, -0.5, 1.2])
device.changeCamera(1.0, 90, -5, [1.5, 0, 1])
q_current, v_current = device.measureState()

x_measured = mpc.getHandler().shapeState(q_current, v_current)

""" Define feet trajectory """
swing_apex = 0.15
x_forward = 0.1
y_forward = 0.0
foot_yaw = 0
y_gap = 0.18
x_depth = 0.0
LF_id = handler.getModel().getFrameId("left_sole_link")
RF_id = handler.getModel().getFrameId("right_sole_link")
foottraj = footTrajectory(
    mpc.getHandler().getData().oMf[LF_id].copy(),
    mpc.getHandler().getData().oMf[RF_id].copy(),
    T_ss,
    T_ds,
    100,
    swing_apex,
    x_forward,
    y_forward,
    foot_yaw,
    y_gap,
    x_depth,
)

land_LFs2 = mpc.getFootLandTimings("left_sole_link").tolist()
land_RFs2 = mpc.getFootLandTimings("right_sole_link").tolist()
takeoff_LFs2 = mpc.getFootTakeoffTimings("left_sole_link").tolist()
takeoff_RFs2 = mpc.getFootTakeoffTimings("right_sole_link").tolist()

q_current = x_measured[:nq]
v_current = x_measured[nq:]

land_LF = -1
land_RF = -1
takeoff_LF = -1
takeoff_RF = -1
device.showTargetToTrack(
    mpc.getHandler().getFootPose("left_sole_link"),
    mpc.getHandler().getFootPose("right_sole_link"),
)
for t in range(Tmpc):
    # print("Time " + str(t))
    land_LFs = mpc.getFootLandTimings("left_sole_link")
    land_RFs = mpc.getFootLandTimings("right_sole_link")
    takeoff_LFs = mpc.getFootTakeoffTimings("left_sole_link")
    takeoff_RFs = mpc.getFootTakeoffTimings("right_sole_link")

    if len(land_LFs) > 0:
        land_LF = land_LFs[0]
    else:
        land_LF == -1
    if len(land_RFs) > 0:
        land_RF = land_RFs[0]
    else:
        land_RF == -1
    if len(takeoff_LFs) > 0:
        takeoff_LF = takeoff_LFs[0]
    else:
        takeoff_LF == -1
    if len(takeoff_RFs) > 0:
        takeoff_RF = takeoff_RFs[0]
    else:
        takeoff_RF == -1

    takeoff_RF2, takeoff_LF2, land_RF2, land_LF2 = update_timings(
        land_LFs2, land_RFs2, takeoff_LFs2, takeoff_RFs2
    )

    if land_RF2 == -1:
        foottraj.updateForward(0, 0, y_gap, y_forward, 0, 0, swing_apex)

    LF_refs, RF_refs = foottraj.updateTrajectory(
        takeoff_RF2,
        takeoff_LF2,
        land_RF2,
        land_LF2,
        mpc.getHandler().getData().oMf[LF_id].copy(),
        mpc.getHandler().getData().oMf[RF_id].copy(),
    )

    print(
        "takeoff_RF = " + str(takeoff_RF2) + ", landing_RF = ",
        str(land_RF2) + ", takeoff_LF = " + str(takeoff_LF2) + ", landing_LF = ",
        str(land_LF2),
    )
    start = time.time()
    mpc.iterate(q_current, v_current)
    end = time.time()
    print("MPC iterate = " + str(end - start))
    device.moveMarkers(
        mpc.getReferencePose(0, "left_sole_link").translation,
        mpc.getReferencePose(0, "right_sole_link").translation,
        # LF_refs[0].translation,
        # RF_refs[0].translation,
    )

    """ for j in range(100):
        mpc.setReferencePose(j, "left_sole_link", LF_refs[j])
        mpc.setReferencePose(j, "right_sole_link", RF_refs[j]) """

    for j in range(10):
        q_current, v_current = device.measureState()

        x_measured = np.concatenate((q_current, v_current))

        x_measured = mpc.getHandler().shapeState(q_current, v_current)

        q_current = x_measured[:nq]
        v_current = x_measured[nq:]

        current_torque = mpc.us[0] - mpc.K0 @ handler.difference(x_measured, mpc.xs[0])
        device.execute(current_torque)
