import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation
from controller import AdaptiveAdmittanceCtrl


def run_controller(desired_pos, desired_vel, desired_force):
    dof = 3
    trials = 7
    x = np.linspace(0, 2 * np.pi, 1500)
    samples = len(x)

    tau = np.ones(dof)
    ks = np.ones(dof)
    kd = np.ones(dof)
    v = np.ones(dof)

    ctrl = AdaptiveAdmittanceCtrl(dof, trials, samples)

    for trial in range(trials):
        idx = 0
        while idx < samples:
            # Simulating the actual force as desired force with random noise
            actual_force = desired_force + 0.01 * np.random.random((dof, samples))

            ctrl.fit(
                desired_pos[:, idx],
                desired_vel[:, idx],
                trial, ks, kd, v, tau,
                actual_force[:, idx],
                desired_force[:, idx],
                idx
            )
            # time.sleep(1./240.)
            idx += 1

    return ctrl


pos = []
vel = []
force = []
r = 0.1  # Radius
incl = np.radians(15)

for t in np.linspace(0, 2 * math.pi, 1500):
    x = 0
    y = r * np.cos(t)
    z = r * np.sin(t)

    rot = Rotation.from_euler('y', 15, degrees=True).as_matrix()

    pos.append(rot @ [x, y, z])
    vel.append(rot @ [x, -r * np.sin(t), r * np.cos(t)])
    force.append(rot @ [x, y, z])

pos = np.transpose(pos)
vel = np.transpose(vel)
force = np.transpose(force)

ctrl = run_controller(desired_pos=pos, desired_vel=vel, desired_force=force)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=15, azim=-135)
ax.plot3D(pos[0, :], pos[1, :], pos[2, :], label='Demonstration')
ax.plot3D(ctrl.pos_list[6, 0, :],
          ctrl.pos_list[6, 1, :],
          ctrl.pos_list[6, 2, :], label='Controller')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend()
plt.show()
