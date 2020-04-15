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

    # self.param.ro = self.x[0]
    # self.ro: float = self.param.ro

    I_0[0] = param.ro * Q_0
    I_0[2] = param.ro * H_0
    I_0[1] = ((I_0[0] + I_0[2]) * param.k) / (1 - param.k)

    E_0: float = np.sum(I_0) * param.ro

    y_0 = np.zeros(10)
    y_0[0:2] = np.reshape([param.N - E_0 - np.sum(I_0), E_0], (2))
    y_0[2:5] = I_0
    y_0[5:10] = np.reshape([Q_0, Rh_0, H_0, R_0, D_0], (5))

    return y_0


class SEIH:
    def __init__(self, param: SEIHParameters, time_intervals: TimeIntervals,
                 # initial_conditions: InitialConditions, real_data: RealData,
                 x: np.ndarray):
        #self.x: np.ndarray = x  # x are parameters to optimize with a metaheuristic approach ie genetic algorithms
        # looks like x has shape (1, 14)

        self.time_intervals: TimeIntervals = time_intervals

        self.param: SEIHParameters = param

        # H_0: float = initial_conditions.H
        # D_0: float = initial_conditions.D
        # Q_0: float = initial_conditions.Q
        #
        # Rh_0: float = 0.0
        # R_0: float = 0.0
        # I_0: np.ndarray = np.zeros(3)
        #
        # # self.param.ro = self.x[0]
        # # self.ro: float = self.param.ro
        #
        # I_0[0] = param.ro * Q_0
        # I_0[2] = param.ro * H_0
        # I_0[1] = ((I_0[0] + I_0[2]) * param.k) / (1 - param.k)
        #
        # E_0: float = np.sum(I_0) * param.ro
        #
        # self.y_0 = np.zeros(10)
        # self.y_0[0:2] = np.reshape([param.N - E_0 - np.sum(I_0), E_0], (2))
        # self.y_0[2:5] = I_0
        # self.y_0[5:10] = np.reshape([Q_0, Rh_0, H_0, R_0, D_0], (5))

        # n_time_intervals: int = len(time_intervals)

        # self.param.beta = self.x[1:n_time_intervals+1]
        # self.param.drate = self.x[n_time_intervals + 1:n_time_intervals*2+1]

        # self.lmbda: np.ndarray = np.array([1 - self.param.k - self.param.severe, self.param.k, self.param.severe])
        # # TODO Maybe transpose lmbda (see https://stackoverflow.com/a/5954747/7555119)
        # self.delta_l = self.param.delta[0]
        # self.delta_r = self.param.delta[1]
        # self.M: np.ndarray = np.zeros([10, 10])
        # self.M[0, 2:5] = (-1) * self.param.beta[0] * self.param.contact
        # self.M[1, 1] = (-1) * self.param.alpha
        # self.M[1, 2:5] = self.param.beta[0] * self.param.contact
        # self.M[2:5, 1:2] = np.transpose([self.lmbda*self.param.alpha])
        # self.M[2:5, 2:5] = (-1) * np.diag(self.param.gamma)
        # self.M[5:8, 2:5] = np.diag(self.param.gamma)
        # self.M[5, 5] = (-1) * self.delta_l
        # self.M[7, 7] = (-1) * self.delta_r
        # self.M[8, 5] = self.delta_l
        # self.M[8, 7] = (1 - self.param.drate[0]) * self.delta_r
        # self.M[9, 7] = self.param.drate[0] * self.delta_r

        self.M: np.ndarray = np.zeros([10, 10])
        self.initializeMMatrix(param)

        # delay_infected: int = initial_conditions.t.toordinal() - real_data.infected['date'][0].toordinal()
        # delay_dead: int = real_data.dead['date'][0].toordinal() - initial_conditions.t.toordinal()
        # time: pd.Series = real_data.infected['date'][delay_infected:]
        # tspan = (time[0].toordinal(), time[len(time) - 1].toordinal())
        #
        # sol = solve_ivp(self.evaluate, tspan, y_0)
        #
        # cumulative_infected = sol.y[5] + sol.y[7] + sol.y[8] + sol.y[9]
        #
        # plt.figure(1)
        # plt.plot(sol.t, cumulative_infected, sol.t, sol.y[0])
        # plt.yscale('log')
        # plt.show()

    def initializeMMatrix(self, param: SEIHParameters):
        lmbda: np.ndarray = np.array([1 - param.k - param.severe, param.k, param.severe])
        # TODO Maybe transpose lmbda (see https://stackoverflow.com/a/5954747/7555119)
        delta_l = param.delta[0]
        delta_r = param.delta[1]

        self.M[0, 2:5] = (-1) * param.beta[0] * param.contact
        self.M[1, 1] = (-1) * param.alpha
        self.M[1, 2:5] = param.beta[0] * param.contact
        self.M[2:5, 1:2] = np.transpose([lmbda * param.alpha])
        self.M[2:5, 2:5] = (-1) * np.diag(param.gamma)
        self.M[5:8, 2:5] = np.diag(param.gamma)
        self.M[5, 5] = (-1) * delta_l
        self.M[7, 7] = (-1) * delta_r
        self.M[8, 5] = delta_l
        self.M[8, 7] = (1 - param.drate[0]) * delta_r
        self.M[9, 7] = param.drate[0] * delta_r

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
        beta = self.param.beta[time_interval_index]

        self.M[0, 2:5] = (-1) * beta * self.param.contact  # This is an analogous to the same block init at __init__
        self.M[1, 2:5] = beta * self.param.contact  # This is an analogous to the same block init at __init__
        self.M[0:2, 2:5] = self.M[0:2, 2:5] * (S / self.param.N)

        H_interval_index = max(0, np.count_nonzero(self.param.Hlimit < H) - 1)
        drate = self.param.drate[H_interval_index]

        self.M[8:10, 7:8] = np.array([[1 - drate], [drate]]) * self.param.delta[1]

        rhs = self.M.dot(y)

        return rhs
