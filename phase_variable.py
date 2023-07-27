import numpy as np


class PhaseVariable:
    def __init__(self):
        self.step_vectorized = np.vectorize(self.step, otypes=[float])
        self.reset()

    def step(self, dt):
        self.s -= self.s * dt
        return self.s

    def rollout(self, t):
        self.reset()
        return self.step_vectorized(np.gradient(t))

    def reset(self):
        self.s = 1.0
