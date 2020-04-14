import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

from DataStructures import SEIHParameters, InitialConditions, TimeIntervals, RealData
from Utilities import read_data_file
from SEIH import SEIH


def main():
    param = SEIHParameters()
    param.contact = np.array([1.0, 1.0, 0.3])
    param.severe  = 0.2
    param.alpha   = 1.0/5.0
    param.gamma   = np.array([1.0/5.0, 1.0/14.0, 1.0/6.0])
    param.delta   = np.array([1.0/9.0, 1.0/15.0])
    param.N       = 100000000
    param.In      = 0
    param.Hlimit  = np.array([0, 1000000])
    param.ro      = 3.0
    param.k       = 0.3
    param.beta    = np.array([0.6])
    param.drate   = np.array([0.5])

    time_intervals = TimeIntervals(['01/22/2020'])

    init   = InitialConditions()
    init.t = pd.to_datetime('01/22/2020')
    init.S = param.N
    init.E = 0
    init.I1 = 0
    init.I2 = 0
    init.I3 = 0
    init.Q = 6
    init.H = 1
    init.D = 0
    init.R = 0
    init.Rh = 0

    infected_data  = read_data_file('testing_resources/cum_world.dat')
    dead_data      = read_data_file('testing_resources/dcm_world.dat')
    recovered_data = read_data_file('testing_resources/rcm_world.dat')
    real_data      = RealData(infected_data, dead_data, recovered_data)


    delay_infected: int = init.t.toordinal() - real_data.infected['date'][0].toordinal()
    delay_dead: int = real_data.dead['date'][0].toordinal() - init.t.toordinal()
    time: pd.Series = real_data.infected['date'][delay_infected:]

    # print(delay_infected)
    # print(delay_dead)
    # print(real_data.infected['date'][delay_dead])
    # como se accede al ultimo elemento de time?
    tspan = (time[0].toordinal(), time[len(time)-1].toordinal())
    # print(tspan)

    x = np.array([3.00, 0.60, 0.30, 0.20, 0.95,
                  0.44, 0.51, 0.42, 0.02, 0.13,
                  0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code

    seih = SEIH(param, time_intervals, init, real_data, x)

    #y = np.array([init.S, init.E, init.I1, init.I2, init.I3, init.Q, init.Rh, init.H, init.R, init.D])

    #print(y)

    # print(time.map(pd.Timestamp.toordinal))

    #sol = solve_ivp(seih.evaluate, tspan, y)
    # print(sol)

    #cumulative_infected = sol.y[5] + sol.y[7] + sol.y[8] + sol.y[9]

    # print(cumulative_infected)
    #plt.figure(1)
    #plt.plot(sol.t, cumulative_infected)
    #plt.yscale('log')
    #plt.show()


if __name__ == '__main__':
    main()
