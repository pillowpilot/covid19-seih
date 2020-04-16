import numpy as np
import pandas as pd
from DataStructures import SEIHParameters, TimeIntervals, InitialConditions, RealData
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def buildY_0(initial_conditions: InitialConditions, param: SEIHParameters):
    H_0: float = initial_conditions.H
    D_0: float = initial_conditions.D
    Q_0: float = initial_conditions.Q

    Rh_0: float = 0.0
    R_0: float = 0.0
    I_0: np.ndarray = np.zeros(3)

    I_0[0] = param.ro * Q_0  # infectados sintomas leves (iniciales) [I1]
    I_0[2] = param.ro * H_0  # infectados sintomas leves sin reportes (iniciales)  [I2]
    I_0[1] = ((I_0[0] + I_0[2]) * param.k) / (1 - param.k)  # # infectados sintomas severos (iniciales)  [I3]

    E_0: float = np.sum(I_0) * param.ro

    y_0 = np.zeros(10)  # vector de variables del rhs
    y_0[0:2] = np.reshape([param.N - E_0 - np.sum(I_0), E_0], (2))  # param.N - E_0 - np.sum(I_0) == suceptibles
    y_0[2:5] = I_0
    y_0[5:10] = np.reshape([Q_0, Rh_0, H_0, R_0, D_0], (5))

    return y_0


class SEIH:
    def __init__(self, parameters: SEIHParameters, time_intervals: TimeIntervals,
                 x: np.ndarray):
        #self.x: np.ndarray = x  # x are parameters to optimize with a metaheuristic approach ie genetic algorithms
        # looks like x has shape (1, 14)
        self.time_intervals: TimeIntervals = time_intervals
        self.parameters: SEIHParameters = parameters
        self.M: np.ndarray = np.zeros([10, 10])
        self.initializeMMatrix(parameters)

    def initializeMMatrix(self, parameters: SEIHParameters):
        lmbda: np.ndarray = np.array([1 - parameters.k - parameters.severe, parameters.k, parameters.severe])
        # TODO Maybe transpose lmbda (see https://stackoverflow.com/a/5954747/7555119)
        delta_l = parameters.delta[0]
        delta_r = parameters.delta[1]

        self.M[0, 2:5] = (-1) * parameters.beta[0] * parameters.contact
        self.M[1, 1] = (-1) * parameters.alpha
        self.M[1, 2:5] = parameters.beta[0] * parameters.contact
        self.M[2:5, 1:2] = np.transpose([lmbda * parameters.alpha])
        self.M[2:5, 2:5] = (-1) * np.diag(parameters.gamma)
        self.M[5:8, 2:5] = np.diag(parameters.gamma)
        self.M[5, 5] = (-1) * delta_l
        self.M[7, 7] = (-1) * delta_r
        self.M[8, 5] = delta_l
        self.M[8, 7] = (1 - parameters.drate[0]) * delta_r
        self.M[9, 7] = parameters.drate[0] * delta_r

    # See https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
    # evaluate must be callable with the signature evaluate(t, y).
    # t must be a scalar.
    # y can have shape (n,) or (n, k). According to MATLAB-files, shape(y) = (10,).
    # the return value shape must be the same as the shape of y.
    # TODO Define every item in y
    # TODO Make self.evaluate pure and run them in parallel?
    def evaluate(self, time: float, y: np.ndarray) -> np.ndarray:
        assert len(y) == 10
        assert self.M.shape[0] == len(y)
        S = y[0]
        H = y[7]

        time_interval_index = max(0, np.count_nonzero(
            self.time_intervals.intervals.map(pd.Timestamp.toordinal) < time) - 1)
        beta = self.parameters.beta[time_interval_index]

        self.M[0, 2:5] = (-1) * beta * self.parameters.contact  # This is an analogous to the same block init at __init__
        self.M[1, 2:5] = beta * self.parameters.contact  # This is an analogous to the same block init at __init__
        self.M[0:2, 2:5] = self.M[0:2, 2:5] * (S / self.parameters.N)

        H_interval_index = max(0, np.count_nonzero(self.parameters.Hlimit < H) - 1)
        drate = self.parameters.drate[H_interval_index]

        self.M[8:10, 7:8] = np.array([[1 - drate], [drate]]) * self.parameters.delta[1]

        rhs = self.M.dot(y)

        return rhs
