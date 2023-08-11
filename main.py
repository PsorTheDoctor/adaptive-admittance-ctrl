import numpy as np
import matplotlib.pyplot as plt
from controller import AdaptiveAdmittanceCtrl
from utils import *


def run_controller(desired_pos, desired_vel, desired_force):
    dof = 3
    trials = 15
    x = np.linspace(0, 2 * np.pi, 1500)
    samples = len(x)

    f = np.ones(dof)
    ks = np.ones(dof)
    kd = np.ones(dof)
    v = np.ones(dof)

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


def plot_input(pos, vel, force):
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 4))
    ax[0].set_title('Desired trajectory')
    ax[0].plot(pos[0])
    ax[0].set_ylabel('Positon')
    ax[1].plot(vel[0])
    ax[1].set_ylabel('Velocity')
    ax[2].plot(force[0])
    ax[2].set_ylabel('Force')
    ax[2].set_xlabel('t')


def plot_results(ctrl):
    x = np.linspace(0, 2 * np.pi, 1500)

    fig, ax = plt.subplots(3, 1, sharex=True, figsize=(8, 4))
    ax[0].plot(x, ctrl.f_list[0, 0, :], label='First trial', color='lightgray')
    ax[0].plot(x, ctrl.f_list[3, 0, :], label='Middle trial', color='gray')
    ax[0].plot(x, ctrl.f_list[6, 0, :], label='Final trial', color='red')
    ax[0].legend()

    ax[1].plot(x, ctrl.f_list[0, 1, :], label='First trial', color='lightgray')
    ax[1].plot(x, ctrl.f_list[3, 1, :], label='Middle trial', color='gray')
    ax[1].plot(x, ctrl.f_list[6, 1, :], label='Final trial', color='red')
    ax[1].legend()

    ax[2].plot(x, ctrl.f_list[0, 2, :], label='First trial', color='lightgray')
    ax[2].plot(x, ctrl.f_list[3, 2, :], label='Middle trial', color='gray')
    ax[2].plot(x, ctrl.f_list[6, 2, :], label='Final trial', color='red')
    ax[2].legend()

    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    ax[0].plot(x, ctrl.ks_list[0, 0, :], label='First trial', color='lightgray')
    ax[0].plot(x, ctrl.ks_list[7, 0, :], label='Middle trial', color='gray')
    ax[0].plot(x, ctrl.ks_list[14, 0, :], label='Final trial', color='red')
    ax[0].set_ylabel('Ks')

    ax[1].plot(x, ctrl.kd_list[0, 0, :], label='First trial', color='lightgray')
    ax[1].plot(x, ctrl.kd_list[7, 0, :], label='Middle trial', color='gray')
    ax[1].plot(x, ctrl.kd_list[14, 0, :], label='Final trial', color='red')
    ax[1].set_ylabel('Kd')

    ax[2].plot(x, np.abs(ctrl.pos_err_list[0, 0, :]), label='First trial', color='lightgray')
    ax[2].plot(x, np.abs(ctrl.pos_err_list[7, 0, :]), label='Middle trial', color='gray')
    ax[2].plot(x, np.abs(ctrl.pos_err_list[14, 0, :]), label='Final trial', color='red')
    ax[2].set_ylabel('Position error')

    ax[3].plot(x, np.abs(ctrl.vel_err_list[0, 0, :]), label='First trial', color='lightgray')
    ax[3].plot(x, np.abs(ctrl.vel_err_list[7, 0, :]), label='Middle trial', color='gray')
    ax[3].plot(x, np.abs(ctrl.vel_err_list[14, 0, :]), label='Final trial', color='red')
    ax[3].set_xlabel('t')
    ax[3].set_ylabel('Velocity error')
    ax[3].legend()

    plt.show()


def main():
    # pos, vel, force = load_simulated_data()
    # ctrl = run_controller(desired_pos=pos, desired_vel=vel, desired_force=force)
    # plot_input(pos, vel, force)
    # plot_results(ctrl)

    pos, vel, force = load_empirical_data()
    ctrl = run_controller(desired_pos=pos, desired_vel=vel, desired_force=force)
    plot_input(pos, vel, force)
    plot_results(ctrl)


if __name__ == '__main__':
    main()
