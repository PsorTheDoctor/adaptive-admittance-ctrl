"""
import pybullet as p
import pybullet_data
import math
import time
from controller import AdaptiveAdmittanceCtrl

p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# Create environment
plane = p.loadURDF('plane.urdf')
p.changeDynamics(plane, -1, lateralFriction=60)

# Create robot
robot = p.loadURDF('kuka_iiwa/model.urdf', basePosition=[0, 0, 0.05])
n_joints = p.getNumJoints(robot)
end_effector = n_joints - 1

# Create object
visual_shape = p.createVisualShape(
    shapeType=p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]
)
collision_shape = p.createCollisionShape(
    shapeType=p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.1]
)
p.createMultiBody(
    baseMass=1,
    baseCollisionShapeIndex=collision_shape,
    baseVisualShapeIndex=visual_shape,
    basePosition=[1, 0, 0.05],
    useMaximalCoordinates=True
)

pos = [0.25, 0, 0.2]
orn = p.getQuaternionFromEuler([0, -math.pi, 0])

while True:
    pos[0] += 0.0001

    joint_poses = p.calculateInverseKinematics(
        robot, end_effector, pos, orn
    )
    target_poses = joint_poses

    for i in range(n_joints):
        p.setJointMotorControl2(
            bodyIndex=robot,
            jointIndex=i,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_poses[i],
            targetVelocity=0,
            force=500,
            positionGain=0.03,
            velocityGain=1
        )
    p.stepSimulation()
    time.sleep(1./240.)
"""