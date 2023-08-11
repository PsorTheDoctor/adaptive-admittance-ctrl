import numpy as np
import matplotlib.pyplot as plt
import math
from controller import AdaptiveAdmittanceCtrl
from utils import *


def run_controller(desired_pos, desired_vel, desired_force):
    dof = 3
    trials = 15
    x = np.linspace(0, 2 * np.pi, 1500)
    samples = len(x)

    f = np.zeros(dof)
    ks = np.zeros(dof)
    kd = np.zeros(dof)
    v = np.zeros(dof)

    ctrl = AdaptiveAdmittanceCtrl(dof, trials, samples)

    for trial in range(trials):
        idx = 0
        while idx < samples:
            # Simulating the actual force as desired force with random noise
            actual_force = desired_force + 0.1 * np.random.random((dof, samples))

            ctrl.fit(
                desired_pos[:, idx],
                desired_vel[:, idx],
                trial, ks, kd, v, f,
                actual_force[:, idx],
                desired_force[:, idx],
                idx
            )
            idx += 1

    return ctrl


pos = []
vel = []
force = []
r = 0.1  # Radius
tilt = np.radians(15)

for t in np.linspace(0, 2 * math.pi, 1500):
    x = r * np.cos(tilt) * np.cos(t)
    y = r * np.sin(t)
    z = r * np.sin(tilt) * np.cos(t)

    dx = -r * np.cos(tilt) * np.sin(t)
    dy = r * np.cos(t)
    dz = -r * np.sin(tilt) * np.sin(t)

    pos.append([x, y, z])
    vel.append([dx, dy, dz])
    force.append(10 * np.array([x, y, z])) #  * np.random.random(3))

pos = np.transpose(pos)
vel = np.transpose(vel)
force = np.transpose(force)

ctrl = run_controller(desired_pos=pos, desired_vel=vel, desired_force=force)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.view_init(elev=30, azim=-135)
ax.plot3D(pos[0, :], pos[1, :], pos[2, :], label='Demonstration', color='red')
ax.plot3D(ctrl.pos_list[1, 0, :],
          ctrl.pos_list[1, 1, :],
          ctrl.pos_list[1, 2, :], label='First trial', color='lightgray')
ax.plot3D(ctrl.pos_list[7, 0, :],
          ctrl.pos_list[7, 1, :],
          ctrl.pos_list[7, 2, :], label='Middle trial', color='gray')
ax.plot3D(ctrl.pos_list[14, 0, :],
          ctrl.pos_list[14, 1, :],
          ctrl.pos_list[14, 2, :], label='Last trial')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
# ax.legend()

plt.show()
