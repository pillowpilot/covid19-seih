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

    parameters = SEIHParameters()
    parameters.contact = np.array([1.0, 1.0, 0.3])
    parameters.severe = 0.2  #
    parameters.alpha = 1.0 / 5.0
    parameters.gamma = np.array([1.0 / 5.0, 1.0 / 14.0, 1.0 / 6.0])
    parameters.delta = np.array([1.0 / 9.0, 1.0 / 15.0])
    parameters.N = 100000000
    parameters.In = 0  # casos importados
    parameters.Hlimit = np.array([0, 1000000])  # limite de hospitalizaciones
    parameters.k = 0.3  # porcentaje de los que no se reportan
    parameters.beta = x[1:len(time_intervals) + 1]  # entre 0 y (2 o 3)
    parameters.drate = x[len(time_intervals) + 1:]  # entre 0 y 1 (dead rate)
    parameters.ro = x[0]  # entre 0 y (10 o 20)

    ode_modeling_variables = InitialConditions()
    ode_modeling_variables.t = pd.to_datetime('01/22/2020')
    ode_modeling_variables.S = parameters.N  # suceptible (puden infectarse)
    ode_modeling_variables.E = 0  # expuestos
    ode_modeling_variables.I1 = 0  # infectados sintomas leves
    ode_modeling_variables.I2 = 0  # infectados sintomas leves sin reportes
    ode_modeling_variables.I3 = 0  # infectados sintomas severos
    ode_modeling_variables.Q = 6  # en sus casas (I1, reportan -> Q)
    ode_modeling_variables.H = 1  # internados (I3, reportan -> H) [para saber la capacidad del sistema sanitario]
    ode_modeling_variables.D = 0  # dead
    ode_modeling_variables.R = 0  # recuperados
    ode_modeling_variables.Rh = 0  # (I2, despues cuarentena -> Rh)

    infected_data = read_data_file('testing_resources/cum_world.dat')
    dead_data = read_data_file('testing_resources/dcm_world.dat')
    recovered_data = read_data_file('testing_resources/rcm_world.dat')
    real_data = RealData(infected_data, dead_data, recovered_data)

    y_0 = buildY_0(ode_modeling_variables, parameters)

    seih = SEIH(parameters, time_intervals, x)

    infected_offset: int = ode_modeling_variables.t.toordinal() - real_data.infected['date'][0].toordinal()
    dead_offset: int = real_data.dead['date'][0].toordinal() - ode_modeling_variables.t.toordinal()
    time: pd.Series = real_data.infected['date'][infected_offset:]
    time_span = (time[0].toordinal(), time[len(time) - 1].toordinal())

    solution = solve_ivp(seih.evaluate, time_span, y_0, t_eval=np.arange(time_span[0], time_span[1] + 1))
    in_quarantine = solution.y[5]
    hospitalized = solution.y[7]
    recovered = solution.y[8]
    dead = solution.y[9]

    # TODO cumI = sum(y(:,[6 8:10]),2); Q+H+R+D? infected_theoretical es un BUEN NOMBRE?
    infected_theoretical: np.ndarray = in_quarantine + hospitalized + recovered + dead  # Q, H, R, D
    transformed_infected_theoretical: np.ndarray = np.log(infected_theoretical)
    infected_real_data: np.ndarray = real_data.infected['infected'][infected_offset:].values  # pd.Series -> np.ndarray
    transformed_infected_real_data: np.ndarray = np.log(infected_real_data)
    infected_prediction_error = np.linalg.norm(transformed_infected_real_data - transformed_infected_theoretical)

    dead_theoretical: np.ndarray = dead[dead_offset:]
    transformed_dead_theoretical: np.ndarray = np.log(dead_theoretical)
    dead_real_data: np.ndarray = real_data.dead['dead']
    transformed_dead_read_data: np.ndarray = np.log(dead_real_data)
    dead_prediction_error = np.linalg.norm(transformed_dead_read_data - transformed_dead_theoretical)

    prediction_error = infected_prediction_error + dead_prediction_error

    # plt.figure(44)
    # plt.plot(transformed_dead_theoretical)
    # plt.plot(transformed_dead_read_data)
    # plt.figure(55)
    # plt.plot(transformed_infected_theoretical)
    # plt.plot(transformed_infected_real_data)
    # plt.show()

    return prediction_error


def main():
    x = np.array([3.00, 0.60, 0.30, 0.20, 0.95,
                  0.44, 0.51, 0.42, 0.02, 0.13,
                  0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code

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
