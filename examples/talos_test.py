import numpy as np
import aligator
import pinocchio as pin
import example_robot_data as erd

robot = erd.load("talos")
rmodel = robot.model
RF_id = rmodel.getFrameId("right_sole_link")
nv = rmodel.nv
nu = nv - 6

space = aligator.manifolds.MultibodyPhaseSpace(rmodel)

posref = pin.SE3.Identity()
frame_fn_RF = aligator.FramePlacementResidual(space.ndx, nu, rmodel, posref, RF_id)
rcost = aligator.CostStack(space, nu)
residual = aligator.QuadraticResidualCost(space, frame_fn_RF, np.eye(6))
rcost.addCost(residual)

newpos = pin.SE3.Identity()
newpos.translation[2] = 0.2
print(
    "Translation stored = ",
    str(rcost.getComponent(0).residual.getReference().translation),
)
rcost.getComponent(0).residual.setReference(newpos)
print(
    "Set reference translation to ",
    str(newpos.translation),
)
print(
    "New translation stored = ",
    str(rcost.getComponent(0).residual.getReference().translation),
)
