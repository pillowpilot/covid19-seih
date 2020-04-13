import unittest
import numpy as np
import pandas as pd
from SEIH import SEIH
from DataStructures import SEIHParameters, InitialConditions, TimeIntervals, RealData
from Utilities import read_data_file


class SEIHTest(unittest.TestCase):
    def test_captured_input_on_evaluate(self):
        param = SEIHParameters()
        param.contact = np.array([1, 1, 0.3000])
        param.severe = 0.1500
        param.alpha = 0.2000
        param.gamma = np.array([0.2000, 0.0714, 0.1667])
        param.delta = np.array([0.1111, 0.0667])
        param.N = 100000000
        param.In = 0
        param.Hlimit = np.array([0, 10, 50, 100, 500, 1000, 5000])
        param.ro = 2.3977
        param.k = 0.4832
        param.beta = np.array([0.2220, 0.9773, 0.1514, 0.6609, 0.3833])
        param.drate = np.array([0.6626, 0.3076, 0.1451, 0.1039, 0.4661, 0.6898, 0.6328])

        time_intervals = TimeIntervals(['01/22/2020', '01/28/2020', '02/10/2020', '03/02/2020', '03/22/2020'])

        init = InitialConditions()
        init.t = pd.to_datetime('01/22/2020')  # Also to_datetime('01/22/2020', format='%m/%d/%Y')
        init.Q = 6
        init.H = 1
        init.D = 0

        infected_data = read_data_file('testing_resources/cum_world.dat')
        dead_data = read_data_file('testing_resources/dcm_world.dat')
        recovered_data = read_data_file('testing_resources/rcm_world.dat')
        real_data = RealData(infected_data, dead_data, recovered_data)

        # TODO Verify if this makes sense!
        delay_infected: int = init.t.toordinal() - real_data.infected['date'][0].toordinal() + 1
        delay_dead: int = real_data.dead['date'][0].toordinal() - init.t.toordinal() + 1
        time: pd.Series = real_data.infected['date'][delay_infected:]

        x = np.array([2.3977, 0.4832, 0.2220, 0.9773, 0.1514,
                      0.6609, 0.3833, 0.6626, 0.3076, 0.1451,
                      0.1039, 0.4661, 0.6898, 0.6328])  # TODO Maybe make this a column array

        seih = SEIH(param, time_intervals, init, real_data, x)

        y: np.ndarray = np.array([10, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * 1e7

        t: int = time[1].toordinal()
        seih.evaluate(t, y)


if __name__ == '__main__':
    unittest.main()
