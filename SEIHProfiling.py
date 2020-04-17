import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from SEIH import SEIH, buildY_0
from DataStructures import SEIHParameters, ODEVariables, TimeIntervals, RealData
from Utilities import read_data_file

# run in console: python -m cProfile -s tottime SEIHProfiling.py
def experiment():
    x = np.array([1.76592537, 0.64402601, 0.22258755, 0.38885279])

    infected_data = read_data_file('testing_resources/cum_world.dat')
    dead_data = read_data_file('testing_resources/dcm_world.dat')
    recovered_data = read_data_file('testing_resources/rcm_world.dat')
    real_data: RealData = RealData(infected_data, dead_data, recovered_data)

    time_intervals: TimeIntervals = TimeIntervals(['01/22/2020'])

    parameters: SEIHParameters = SEIHParameters()
    parameters.contact = np.array([1.0, 1.0, 0.3])
    parameters.severe = 0.2
    parameters.alpha = 1.0 / 5.0
    parameters.gamma = np.array([1.0 / 5.0, 1.0 / 14.0, 1.0 / 6.0])
    parameters.delta = np.array([1.0 / 9.0, 1.0 / 15.0])
    parameters.N = 100000000
    parameters.imported_cases = 0
    parameters.hospitalization_limit = np.array([0, 1000000])
    parameters.k = 0.3

    parameters.ro = x[0]
    parameters.beta = x[1:len(time_intervals) + 1]
    parameters.dead_rate = x[len(time_intervals) + 1:]

    # Load ODE variables initial values
    ode_modeling_variables = ODEVariables()
    ode_modeling_variables.time = pd.to_datetime('01/22/2020')
    ode_modeling_variables.susceptible = parameters.N
    ode_modeling_variables.exposed = 0
    ode_modeling_variables.infected_mild = 0
    ode_modeling_variables.infected_mild_unreported = 0
    ode_modeling_variables.infected_severe = 0
    ode_modeling_variables.in_quarantine = 6
    ode_modeling_variables.hospitalized = 1
    ode_modeling_variables.dead = 0
    ode_modeling_variables.recovered = 0
    ode_modeling_variables.Rh = 0

    infected_offset: int = ode_modeling_variables.time.toordinal() - real_data.infected['date'][0].toordinal()
    dead_offset: int = real_data.dead['date'][0].toordinal() - ode_modeling_variables.time.toordinal()

    infected_time_series: pd.Series = real_data.infected['date'][infected_offset:]
    dead_time_series: pd.Series = real_data.dead['date']

    time_span = (infected_time_series[0].toordinal(), infected_time_series[len(infected_time_series) - 1].toordinal())

    y_0 = buildY_0(ode_modeling_variables, parameters)
    seih = SEIH(parameters, time_intervals, x)

    solution = solve_ivp(seih.evaluate, time_span, y_0, t_eval=np.arange(time_span[0], time_span[1] + 1))
    in_quarantine = solution.y[5]
    hospitalized = solution.y[7]
    recovered = solution.y[8]
    dead = solution.y[9]

    infected_theoretical: np.ndarray = in_quarantine + hospitalized + recovered + dead  # Q, H, R, D
    transformed_infected_theoretical: np.ndarray = np.log(infected_theoretical)
    infected_real_data: np.ndarray = real_data.infected['infected'][infected_offset:].values
    transformed_infected_real_data: np.ndarray = np.log(infected_real_data)
    infected_prediction_error = np.linalg.norm(transformed_infected_real_data - transformed_infected_theoretical)

    dead_theoretical: np.ndarray = dead[dead_offset:]
    transformed_dead_theoretical: np.ndarray = np.log(dead_theoretical)  # TODO Sometimes: log(0)
    dead_real_data: np.ndarray = real_data.dead['dead']
    transformed_dead_real_data: np.ndarray = np.log(dead_real_data)
    dead_prediction_error = np.linalg.norm(transformed_dead_real_data - transformed_dead_theoretical)

    prediction_error = infected_prediction_error + dead_prediction_error

    print(prediction_error)


if __name__ == '__main__':
    experiment()
