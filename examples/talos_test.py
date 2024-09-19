import numpy as np
import aligator
import pinocchio as pin


def load_talos_no_wristhead():
    import example_robot_data as erd

    robot = erd.load("talos")
    qref = robot.model.referenceConfigurations["half_sitting"]
    locked_joints = [20, 21, 22, 23, 28, 29, 30, 31, 32, 33]
    red_bot = robot.buildReducedRobot(locked_joints, qref)
    return robot, red_bot


robotComplete, robot = load_talos_no_wristhead()
rmodel: pin.Model = robot.model
RF_id = rmodel.getFrameId("right_sole_link")
nv = rmodel.nv
nu = nv - 6

space = aligator.manifolds.MultibodyPhaseSpace(rmodel)

posref = pin.SE3.Identity()
frame_fn_RF = aligator.FramePlacementResidual(space.ndx, nu, rmodel, posref, RF_id)

rcost = aligator.CostStack(space, nu)
rcost.addCost(aligator.QuadraticResidualCost(space, frame_fn_RF, np.eye(6)))

""" pos = pin.SE3.Identity()
pos.translation[2] = 0.2
print("translation stored = ", str(rcost.components[0].residual.getReference().translation))
rcost.components[0].residual.setReference(pos)
print("new translation stored = ", str(rcost.components[0].residual.getReference().translation)) """

rcost = aligator.CostStack(space, nu)
residual = aligator.QuadraticResidualCost(space, frame_fn_RF, np.eye(6))
rcost.addCost(residual)

pos = pin.SE3.Identity()
pos.translation[2] = 0.2
print(
    "translation stored = ",
    str(rcost.components[0].residual.getReference().translation),
)
rcost.components[0].residual.setReference(pos)
print(
    "new translation stored = ",
    str(rcost.components[0].residual.getReference().translation),
)
