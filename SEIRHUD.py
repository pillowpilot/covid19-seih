from math import floor

import numpy as np
import pandas as pd
import pylab
import time
from scipy.integrate import solve_ivp
from scipy.optimize import brute


class Model:
    def __init__(self, alpha, beta, gamma, population, imported):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.population = population
        self.imported = imported

    def dSdt(self, s, i):
        return - self.beta * s * i / self.population

    def dEdt(self, s, i, e):
        first_term = self.beta * s * i / self.population
        second_term = - self.alpha * e
        return first_term + second_term

    def dIdt(self, i, e):
        first_term = self.alpha * e
        second_term = - self.gamma * i
        return first_term + second_term

    def dRdt(self, t, i):
        return self.gamma * i + self.imported[floor(t)]


class Adapter:
    def __init__(self, model):
        self.model = model

    def rhs(self, t, y):  # t is scalar, y in n-d vector, rhs is n-d vector
        s = y[0]
        e = y[1]
        i = y[2]
        r = y[3]
        return np.array([self.model.dSdt(s, i), self.model.dEdt(s, i, e),
                         self.model.dIdt(i, e), self.model.dRdt(t, i)])


class ReportedError:
    def __init__(self, reported_data, max_day):
        self.reported_data = reported_data
        self.max_day = max_day

    def evaluate(self, reported_simulation):
        error = 0
        for index in range(0, self.max_day):
            error += (self.reported_data[index] - reported_simulation[index]) ** 2
        error /= self.max_day
        return error


class Optimizable:
    def __init__(self, adapter, reported_data, initial_population, max_day):
        self.adapter = adapter
        self.reported_data = reported_data
        self.initial_population = initial_population
        self.max_day = max_day

    def cost(self, x):
        s_0 = self.initial_population - x[0] - x[1] - x[2]
        e_0 = x[0]
        i_0 = x[1]
        r_0 = x[2]
        solution = solve_ivp(self.adapter.rhs,
                             (0, 100),  # max_day [dia] # TODO Change 100 to something meaningful
                             np.array([s_0, e_0, i_0, r_0]),
                             'RK45')
        error_measurement = ReportedError(self.reported_data, self.max_day)
        error = error_measurement.evaluate(solution.y[3])
        return error


def main_proto():
    data = pd.read_csv('DatosMazzolenni.csv')
    data = data.fillna(0)
    print(data)

    data_ext = data['Exterior']
    data_tot = data['total confirmados']
    print(f'len(total confirmados): {len(data_tot)}')

    alpha = 1 / 3
    beta = 0.25
    gamma = 1 / 10
    initial_population = 7e6

    model = Model(alpha, beta, gamma, initial_population, data_ext)
    adapter = Adapter(model)
    to_minimize = Optimizable(adapter, data_tot, initial_population, 13)

    optimized_initial_state = brute(to_minimize.cost, [slice(0, 10, 0.5), slice(0, 10, 0.5), slice(0, 10, 0.5)])
    print(optimized_initial_state)

    solution = solve_ivp(adapter.rhs,
                         (0, 100),  # max_day [dia] # TODO Change 100 to something meaningful
                         np.concatenate(([initial_population], optimized_initial_state)),
                         'RK45')

    pylab.plot(solution.t, solution.y[0], 'o')
    pylab.plot(solution.t, solution.y[1], '*')
    pylab.plot(solution.t, solution.y[2], '+')
    pylab.plot(solution.t, solution.y[3], '-')

    pylab.show()


def main():
    data = pd.read_csv('DatosMazzolenni.csv')
    data = data.fillna(0)
    print(data)

    data_ext = data['Exterior']
    data_tot = data['total confirmados']
    print(f'len(total confirmados): {len(data_tot)}')

    alpha = 1 / 3
    beta = 0.25
    gamma = 1 / 10
    initial_population = 7e6
    s_0 = initial_population - 1
    e_0 = 0
    i_0 = 1
    r_0 = 0
    model = Model(alpha, beta, gamma, initial_population, data_ext)
    adapter = Adapter(model)

    start_time = time.time()  # Timer start time
    solution = solve_ivp(adapter.rhs,
                         (0, 100),  # t [dia]
                         np.array([s_0, e_0, i_0, r_0]),
                         'RK45')
    stop_time = time.time()  # Timer stop time
    print(f'Total Time: {stop_time - start_time}')  # ~0.005s
    print(solution)

    error_measurement = ReportedError(data_tot, 13)

    print(f'Reported Real: {data_tot[0:13]}')
    print(f'Reported Simulation: {solution.y[3][0:13]}')
    print(f'Error: {error_measurement.evaluate(solution.y[3])}')

    pylab.plot(solution.t, solution.y[3], '-')
    pylab.plot(data_tot)
    pylab.show()


if __name__ == '__main__':
    main()

# Current usage: $python ./SEIRHUD.py
