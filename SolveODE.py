import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from DataStructures import SEIHParameters, InitialConditions, TimeIntervals, RealData
from Utilities import read_data_file
from SEIH import SEIH


def main():
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

    x = np.array([3.39, 0.23, 0.57, 0.20, 0.95,
                  0.44, 0.51, 0.42, 0.02, 0.13,
                  0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code

    seih = SEIH(param, time_intervals, init, real_data, x)

    y = np.array([1e8, 169.8547, 18.4871, 33.5583, 3.0812, 6, 0, 1, 0, 0])

    sol = solve_ivp(seih.evaluate, (737700, 737900), y)
    print(sol)

    cumulative_infected = sol.y[5] + sol.y[7] + sol.y[8]

    print(cumulative_infected)
    plt.figure(1)
    plt.plot(sol.t, cumulative_infected)
    plt.yscale('log')
    plt.show()


if __name__ == '__main__':
    main()
