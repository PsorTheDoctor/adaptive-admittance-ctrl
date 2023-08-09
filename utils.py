import numpy as np
import pandas as pd


def load_simulated_data():
    """
    Simulated sinus function.
    """
    dof = 3
    x = np.linspace(0, 2 * np.pi, 1500)

    pos = np.zeros((dof, len(x)))
    vel = np.zeros((dof, len(x)))
    force = np.zeros((dof, len(x)))

    pos[0] = np.sin(x)
    vel[0] = np.cos(x)  # Derivative of sinus
    force[0] = np.sin(x)
    return pos, vel, force


def load_empirical_data():
    """
    Data collected during the experiment with UR5.
    """
    file = 'data.xls'
    data = np.array(pd.read_excel(file))
    data = np.transpose(data)

    n = 1500
    pos = data[0:3, :n]
    vel = data[3:6, :n]
    force = data[6:9, :n]

    return pos, vel, force
