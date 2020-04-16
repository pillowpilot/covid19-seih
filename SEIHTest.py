import unittest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from SEIH import SEIH, buildY_0
from DataStructures import SEIHParameters, ODEVariables, TimeIntervals, RealData
from Utilities import read_data_file


class SEIHTest(unittest.TestCase):
    def test_construction_and_evaluation(self):
        x = np.array([3.00, 0.60, 0.30, 0.20, 0.95,
                      0.44, 0.51, 0.42, 0.02, 0.13,
                      0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code

        time_intervals = TimeIntervals(['01/22/2020'])

        param = SEIHParameters()
        param.contact = np.array([1.0, 1.0, 0.3])
        param.severe = 0.2
        param.alpha = 1.0 / 5.0
        param.gamma = np.array([1.0 / 5.0, 1.0 / 14.0, 1.0 / 6.0])
        param.delta = np.array([1.0 / 9.0, 1.0 / 15.0])
        param.N = 100000000
        param.In = 0
        param.hospitalization_limit = np.array([0, 1000000])
        param.ro = x[0]
        param.k = 0.3
        param.beta = x[1:len(time_intervals) + 1]
        param.dead_rate = x[len(time_intervals) + 1:len(time_intervals) * 2 + 1]

        initial_conditions = ODEVariables()
        initial_conditions.time = pd.to_datetime('01/22/2020')
        initial_conditions.susceptible = param.N
        initial_conditions.exposed = 0
        initial_conditions.infected_mild = 0
        initial_conditions.infected_mild_unreported = 0
        initial_conditions.infected_severe = 0
        initial_conditions.in_quarantine = 6
        initial_conditions.hospitalized = 1
        initial_conditions.dead = 0
        initial_conditions.recovered = 0
        initial_conditions.Rh = 0

        y_0 = buildY_0(initial_conditions, param)

        seih: SEIH = SEIH(param, time_intervals, x)

        infected_data = read_data_file('testing_resources/cum_world.dat')
        dead_data = read_data_file('testing_resources/dcm_world.dat')
        recovered_data = read_data_file('testing_resources/rcm_world.dat')
        real_data = RealData(infected_data, dead_data, recovered_data)

        delay_infected: int = initial_conditions.time.toordinal() - real_data.infected['date'][0].toordinal()
        delay_dead: int = real_data.dead['date'][0].toordinal() - initial_conditions.time.toordinal()
        time: pd.Series = real_data.infected['date'][delay_infected:]
        time_span = (time[0].toordinal(), time[len(time) - 1].toordinal())

        sol = solve_ivp(seih.evaluate, time_span, y_0)

        cumulative_infected = sol.y[5] + sol.y[7] + sol.y[8] + sol.y[9]

        plt.figure(1)
        plt.plot(sol.t, cumulative_infected, sol.t, sol.y[0])
        plt.yscale('log')
        plt.show()


if __name__ == '__main__':
    unittest.main()
