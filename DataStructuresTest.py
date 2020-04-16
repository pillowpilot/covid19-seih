import unittest
import numpy as np
import pandas as pd
from Utilities import read_data_file
from DataStructures import SEIHParameters, ODEVariables, TimeIntervals, RealData


class SEIHParameterTest(unittest.TestCase):
    def test_usage(self):
        param = SEIHParameters()
        param.contact = np.array([1, 1, 0.3])
        param.severe = 0.15
        param.alpha = 1 / 5
        param.gamma = 1 / np.array([5, 14, 6])
        param.delta = 1 / np.array([9, 15])
        param.N = 1e8
        param.imported_cases = 0
        param.hospitalization_limit = np.array([0, 10, 50, 100, 500, 1000, 5000])
        param.ro = 2
        param.k = 1
        param.beta = np.array([9.7e-4, 0.2028, 0.6924, 1, 0])
        param.dead_rate = np.array([0.1544, 0.2830, 0, 0, 0, 1, 0])


class InitialConditionsTest(unittest.TestCase):
    def test_usage(self):
        init = ODEVariables()
        init.time = pd.to_datetime('01/22/2020')  # Also to_datetime('01/22/2020', format='%m/%d/%Y')
        init.in_quarantine = 6
        init.hospitalized = 1
        init.dead = 0


class TimeIntervalsTest(unittest.TestCase):
    def test_usage(self):
        time_intervals = TimeIntervals(['01/22/2020', '01/28/2020', '02/10/2020', '03/02/2020', '03/22/2020'])


class RealDataTest(unittest.TestCase):
    def test_usage(self):
        infected_data = read_data_file('testing_resources/cum_world.dat')
        dead_data = read_data_file('testing_resources/dcm_world.dat')
        recovered_data = read_data_file('testing_resources/rcm_world.dat')
        real_data = RealData(infected_data, dead_data, recovered_data)


if __name__ == '__main__':
    unittest.main()
