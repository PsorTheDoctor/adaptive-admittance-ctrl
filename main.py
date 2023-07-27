import numpy as np
import matplotlib.pyplot as plt
from controller import AdaptiveAdmittanceCtrl

dof = 3
trials = 7
x = np.linspace(-np.pi, np.pi, 1000)
samples = len(x)

desired_pos = np.zeros((dof, len(x)))
desired_vel = np.zeros((dof, len(x)))
desired_force = np.zeros((dof, len(x)))

desired_pos[0] = np.sin(x)
desired_vel[0] = np.cos(x)  # Derivative of sinus
desired_force[0] = np.sin(x)

tau = np.ones(dof)
ks = np.ones(dof)
kd = np.ones(dof)
v = np.ones(dof)

ctrl = AdaptiveAdmittanceCtrl(dof, trials, samples)

for trial in range(trials):

    # ctrl.prev_pos = np.zeros(dof)
    # ctrl.prev_vel = np.zeros(dof)

    sample = 0
    while sample < len(x):
        # Simulating the actual force as desired force with random noise
        actual_force = desired_force + 0.01 * np.random.random(len(x))

        ctrl.fit(
            desired_pos[:, sample],
            desired_vel[:, sample],
            trial, ks, kd, v, tau,
            actual_force[:, sample],
            desired_force[:, sample],
            sample
        )
        # time.sleep(1./240.)
        sample += 1

fig, ax = plt.subplots(3, 1, sharex=True)
ax[0].plot(x, ctrl.tau_list[0, 0, :], label='First trial')
ax[0].plot(x, ctrl.tau_list[3, 0, :], label='Middle trial')
ax[0].plot(x, ctrl.tau_list[6, 0, :], label='Final trial')
ax[0].legend()

ax[1].plot(x, ctrl.tau_list[0, 1, :], label='First trial')
ax[1].plot(x, ctrl.tau_list[3, 1, :], label='Middle trial')
ax[1].plot(x, ctrl.tau_list[6, 1, :], label='Final trial')
ax[1].legend()

ax[2].plot(x, ctrl.tau_list[0, 2, :], label='First trial')
ax[2].plot(x, ctrl.tau_list[3, 2, :], label='Middle trial')
ax[2].plot(x, ctrl.tau_list[6, 2, :], label='Final trial')
ax[2].legend()

fig, ax = plt.subplots(4, 1, sharex=True)
ax[0].plot(x, ctrl.ks_list[0, 0, :], label='First trial')
ax[0].plot(x, ctrl.ks_list[3, 0, :], label='Middle trial')
ax[0].plot(x, ctrl.ks_list[6, 0, :], label='Final trial')
ax[0].set_xlabel('t')
ax[0].set_ylabel('Ks')
ax[0].legend()

ax[1].plot(x, ctrl.kd_list[0, 1, :], label='First trial')
ax[1].plot(x, ctrl.kd_list[3, 1, :], label='Middle trial')
ax[1].plot(x, ctrl.kd_list[6, 1, :], label='Final trial')
ax[1].set_xlabel('t')
ax[1].set_ylabel('Kd')
ax[1].legend()

ax[2].plot(x, ctrl.pos_err_list[0, 2, :], label='First trial')
ax[2].plot(x, ctrl.pos_err_list[3, 2, :], label='Middle trial')
ax[2].plot(x, ctrl.pos_err_list[6, 2, :], label='Final trial')
ax[2].set_xlabel('t')
ax[2].set_ylabel('Position error')
ax[2].legend()

ax[3].plot(x, ctrl.vel_err_list[0, 2, :], label='First trial')
ax[3].plot(x, ctrl.vel_err_list[3, 2, :], label='Middle trial')
ax[3].plot(x, ctrl.vel_err_list[6, 2, :], label='Final trial')
ax[3].set_xlabel('t')
ax[3].set_ylabel('Velocity error')
ax[3].legend()

plt.show()
