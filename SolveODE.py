import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from DataStructures import SEIHParameters, InitialConditions, TimeIntervals, RealData
from Utilities import read_data_file
from SEIH import SEIH, buildY_0


def ode(x: np.ndarray, *args):
    time_intervals = TimeIntervals(['01/22/2020'])

    param = SEIHParameters()
    param.contact = np.array([1.0, 1.0, 0.3])
    param.severe = 0.2  #
    param.alpha = 1.0 / 5.0
    param.gamma = np.array([1.0 / 5.0, 1.0 / 14.0, 1.0 / 6.0])
    param.delta = np.array([1.0 / 9.0, 1.0 / 15.0])
    param.N = 100000000
    param.In = 0  # casos importados
    param.Hlimit = np.array([0, 1000000])  # limite de hospitalizaciones
    # param.ro = 3.0
    param.k = 0.3  # porcentaje de los que no se reportan
    # param.beta = np.array([0.6])
    # param.drate = np.array([0.5])

    # This gets overwritten!
    param.beta = x[1:len(time_intervals) + 1]  # entre 0 y (2 o 3)
    param.drate = x[len(time_intervals) + 1:]  # entre 0 y 1 (dead rate)

    # print(x)
    # print(x[0])
    # print(x[1])
    param.ro = x[0]  # entre 0 y (10 o 20)
    # param.k = x[1]

    initial_conditions = InitialConditions()
    initial_conditions.t = pd.to_datetime('01/22/2020')
    initial_conditions.S = param.N  # suceptible (puden infectarse)
    initial_conditions.E = 0  # expuestos
    initial_conditions.I1 = 0  # infectados sintomas leves
    initial_conditions.I2 = 0  # infectados sintomas leves sin reportes
    initial_conditions.I3 = 0  # infectados sintomas severos
    initial_conditions.Q = 6  # en sus casas (I1, reportan -> Q)
    initial_conditions.H = 1  # internados (I3, reportan -> H) [para saber la capacidad del sistema sanitario]
    initial_conditions.D = 0  # dead
    initial_conditions.R = 0  # recuperados
    initial_conditions.Rh = 0  # (I2, despues cuarentena -> Rh)

    infected_data = read_data_file('testing_resources/cum_world.dat')
    dead_data = read_data_file('testing_resources/dcm_world.dat')
    recovered_data = read_data_file('testing_resources/rcm_world.dat')
    real_data = RealData(infected_data, dead_data, recovered_data)

    y_0 = buildY_0(initial_conditions, param)

    seih = SEIH(param, time_intervals, x)

    delay_infected: int = initial_conditions.t.toordinal() - real_data.infected['date'][0].toordinal()
    delay_dead: int = real_data.dead['date'][0].toordinal() - initial_conditions.t.toordinal()
    time: pd.Series = real_data.infected['date'][delay_infected:]
    # convertir cada entry de time con toordinal para calcular los valores SOLO en esos puntos!
    time_span = (time[0].toordinal(), time[len(time) - 1].toordinal())

    solution = solve_ivp(seih.evaluate, time_span, y_0)  # Ver doc sobre punto anterior.

    cumulative_infected = solution.y[5] + solution.y[7] + solution.y[8] + solution.y[9]  # Q, H, R, D

    # log y dist euclidanea


    return solution


def main():
    # x = np.array([3.00, 0.60, 0.30, 0.20, 0.95,
    #               0.44, 0.51, 0.42, 0.02, 0.13,
    #               0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code

    number_of_variables = 4
    bounds = [(2, 4)] * number_of_variables
    result = differential_evolution(ode, bounds)
    print(result)

    # solution = ode(x)
    #
    # cumulative_infected = solution.y[5] + solution.y[7] + solution.y[8] + solution.y[9]
    #
    # plt.figure(1)
    # plt.plot(solution.t, cumulative_infected, solution.t, solution.y[0])
    # plt.yscale('log')
    # plt.show()

    # optimo ~ > 10, no es probable que pase 20 0 30


if __name__ == '__main__':
    main()
