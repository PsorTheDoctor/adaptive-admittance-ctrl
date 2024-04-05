import numpy as np
import math
from phase_variable import PhaseVariable


class AdaptiveAdmittanceCtrl:

    def __init__(self, dof, trials, samples):
        # Mass, spring, and damper constants for each DoF
        self.M = 1  # * np.eye(dof)
        self.K = 40 # * np.eye(dof)
        self.D = 2 * np.sqrt(self.M * self.K + 270)

        self.trials = trials
        self.dof = dof
        self.samples = samples
        self.dt = 0.002

        self.pos = np.zeros(dof)
        self.vel = np.zeros(dof)
        self.acc = np.zeros(dof)

        # self.pos_err = np.zeros(dof)
        # self.vel_err = np.zeros(dof)
        self.force_err = np.zeros(dof)
        # self.tracking_err = np.zeros(dof)

        self.f = np.zeros(dof)  # Controller output
        self.v = np.zeros(dof)  # Feedforward term
        self.gamma = 5 # Tracking error coeff
        # self._lambda = 0.1  # Forgetting factor
        self.basis_functions = 3
        self.g = self.radialBasis(
            alpha=48, basis_functions=self.basis_functions
        )
        # Adaptation rates for stiffness, damping, and feedforward term
        # that control the convergence speed
        # Values chosen arbitrarily
        self.qs = 22
        self.qd = 16
        self.qv = 14

        self.pos_list = np.zeros((trials, dof, self.samples))
        self.vel_list = np.zeros((trials, dof, self.samples))

        self.pos_err_list = np.zeros((trials, dof, self.samples))
        self.vel_err_list = np.zeros((trials, dof, self.samples))
        self.tracking_err_list = np.zeros((trials, dof, self.samples))

        self.ks_list = np.zeros((trials, dof, self.samples))
        self.kd_list = np.zeros((trials, dof, self.samples))
        self.v_list = np.zeros((trials, dof, self.samples))
        self.f_list = np.zeros((trials, dof, self.samples))

    def radialBasis(self, alpha, basis_functions):
        pv = PhaseVariable()

        # Centres of Gaussian basis functions
        c = np.exp(alpha * np.linspace(0, 1, basis_functions))
        print(c)

        # Variances of Gaussian basis functions
        h = 1.0 / np.gradient(c) ** 2
        print(h)

        f = 0.002 * self.samples
        x = np.arange(0, f, 0.002)
        g = []
        for xi in pv.rollout(x):
            w = np.exp(-0.5 * h * (xi - c) ** 2)
            w /= w.sum()  # Normalization
            g.append(w)

        return np.array(g)

    def mass_spring_damper(self):
        """
        Implementation of 2nd order mass-spring-damper:
        Mp'' + Kp' + Dp = f
        """
        self.spring_force = self.K * self.pos
        self.damper_force = self.D * self.vel
        # noise = np.random.normal(0, 0, self.dof)

        self.acc = (-self.spring_force - self.damper_force + self.f + self.force_err) / self.M

        self.vel = self.vel + self.acc * self.dt
        self.pos = self.pos + self.vel * self.dt

    def fit(self, desired_pos, desired_vel, trial, ks, kd, v, f, actual_force, desired_force, sample):
        i = trial
        j = sample

        self.force_err = actual_force - desired_force
        pos_err = self.pos - desired_pos
        vel_err = self.vel - desired_vel
        tracking_err = self.gamma * pos_err + vel_err  # epsilon

        self.mass_spring_damper()

        # Update gains
        if i == 0:
            self.ks = ks
            self.kd = kd
            self.v = v
            self.f = f
        else:
            self.ks = self.ks_list[i - 1][:, j] + self.qs * self.tracking_err_list[i - 1][:, j] * \
                      self.pos_err_list[i - 1][:, j] * self.g[j, self.basis_functions - self.dof:self.basis_functions]

            self.kd = self.kd_list[i - 1][:, j] + self.qd * self.tracking_err_list[i - 1][:, j] * \
                      self.vel_err_list[i - 1][:, j] * self.g[j, self.basis_functions - self.dof:self.basis_functions]

            self.v = self.v_list[i - 1][:, j] + self.qv * self.tracking_err_list[i - 1][:, j] * \
                     self.g[j, self.basis_functions - self.dof:self.basis_functions]

            # Combine gains into force
            self.f = -(self.ks * pos_err + self.kd * vel_err) - self.v

        for k in range(self.dof):
            self.pos_list[i][k][j] = self.pos[k]
            self.vel_list[i][k][j] = self.vel[k]
            self.pos_err_list[i][k][j] = pos_err[k]
            self.vel_err_list[i][k][j] = vel_err[k]
            self.tracking_err_list[i][k][j] = tracking_err[k]
            self.ks_list[i][k][j] = self.ks[k]
            self.kd_list[i][k][j] = self.kd[k]
            self.v_list[i][k][j] = self.v[k]
            self.f_list[i][k][j] = self.f[k]


ctrl = AdaptiveAdmittanceCtrl(3, 7, 1000)
ctrl.radialBasis(48, 10)