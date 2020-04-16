import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution

from DataStructures import SEIHParameters, ODEVariables, TimeIntervals, RealData
from Utilities import read_data_file
from SEIH import SEIH, buildY_0


class SolveODE:
    def __init__(self):
        # Initialize all static data (data that is *constant* over all calls by optimizer)
        infected_data = read_data_file('testing_resources/cum_world.dat')
        dead_data = read_data_file('testing_resources/dcm_world.dat')
        recovered_data = read_data_file('testing_resources/rcm_world.dat')
        self.real_data: RealData = RealData(infected_data, dead_data, recovered_data)

        self.time_intervals: TimeIntervals = TimeIntervals(['01/22/2020'])

        self.parameters: SEIHParameters = SEIHParameters()
        self.parameters.contact = np.array([1.0, 1.0, 0.3])
        self.parameters.severe = 0.2
        self.parameters.alpha = 1.0 / 5.0
        self.parameters.gamma = np.array([1.0 / 5.0, 1.0 / 14.0, 1.0 / 6.0])
        self.parameters.delta = np.array([1.0 / 9.0, 1.0 / 15.0])
        self.parameters.N = 100000000
        self.parameters.imported_cases = 0  # casos importados
        self.parameters.hospitalization_limit = np.array([0, 1000000])  # limite de hospitalizaciones
        self.parameters.k = 0.3  # porcentaje de los que no se reportan

        self.ode_modeling_variables = ODEVariables()
        self.ode_modeling_variables.time = pd.to_datetime('01/22/2020')
        self.ode_modeling_variables.susceptible = self.parameters.N  # suceptible (puden infectarse)
        self.ode_modeling_variables.exposed = 0  # expuestos
        self.ode_modeling_variables.infected_mild = 0  # infectados sintomas leves
        self.ode_modeling_variables.infected_mild_unreported = 0  # infectados sintomas leves sin reportes
        self.ode_modeling_variables.infected_severe = 0  # infectados sintomas severos
        self.ode_modeling_variables.in_quarantine = 6  # en sus casas (I1, reportan -> Q)
        self.ode_modeling_variables.hospitalized = 1  # internados (I3, reportan -> H) [para saber la capacidad del sistema sanitario]
        self.ode_modeling_variables.dead = 0  # dead
        self.ode_modeling_variables.recovered = 0  # recuperados
        self.ode_modeling_variables.Rh = 0  # (I2, despues cuarentena -> Rh)

        self.infected_offset: int = self.ode_modeling_variables.time.toordinal() - \
                                    self.real_data.infected['date'][0].toordinal()
        self.dead_offset: int = self.real_data.dead['date'][0].toordinal() - \
                                self.ode_modeling_variables.time.toordinal()

        self.infected_time_series: pd.Series = self.real_data.infected['date'][self.infected_offset:]
        self.dead_time_series: pd.Series = self.real_data.dead['date']

        self.time_span = (self.infected_time_series[0].toordinal(),
                          self.infected_time_series[len(self.infected_time_series) - 1].toordinal())

    def ode(self, x: np.ndarray, *args):
        # Initializing non-static parameters
        self.parameters.ro = x[0]  # entre 0 y (10 o 20)
        self.parameters.beta = x[1:len(self.time_intervals) + 1]  # entre 0 y (2 o 3)
        self.parameters.dead_rate = x[len(self.time_intervals) + 1:]  # entre 0 y 1 (dead rate)

        y_0 = buildY_0(self.ode_modeling_variables, self.parameters)
        seih = SEIH(self.parameters, self.time_intervals, x)

        solution = solve_ivp(seih.evaluate, self.time_span, y_0,
                             t_eval=np.arange(self.time_span[0], self.time_span[1] + 1))
        in_quarantine = solution.y[5]
        hospitalized = solution.y[7]
        recovered = solution.y[8]
        dead = solution.y[9]

        # TODO cumI = sum(y(:,[6 8:10]),2); Q+H+R+D? infected_theoretical es un BUEN NOMBRE?
        infected_theoretical: np.ndarray = in_quarantine + hospitalized + recovered + dead  # Q, H, R, D
        transformed_infected_theoretical: np.ndarray = np.log(infected_theoretical)
        infected_real_data: np.ndarray = self.real_data.infected['infected'][self.infected_offset:].values
        transformed_infected_real_data: np.ndarray = np.log(infected_real_data)
        infected_prediction_error = np.linalg.norm(transformed_infected_real_data - transformed_infected_theoretical)

        dead_theoretical: np.ndarray = dead[self.dead_offset:]
        transformed_dead_theoretical: np.ndarray = np.log(dead_theoretical)  # TODO Sometimes: log(0)
        dead_real_data: np.ndarray = self.real_data.dead['dead']
        transformed_dead_real_data: np.ndarray = np.log(dead_real_data)
        dead_prediction_error = np.linalg.norm(transformed_dead_real_data - transformed_dead_theoretical)

        prediction_error = infected_prediction_error + dead_prediction_error

        if 'plot' in args:
            plt.figure(1)
            dt = plt.plot(self.dead_time_series, dead_theoretical, label="Dead Theoretical")
            dr = plt.plot(self.dead_time_series, dead_real_data, label='Dead Real')
            plt.xlabel('Time (days)')
            plt.ylabel('Count')
            plt.yscale('log')
            plt.legend()

            plt.figure(2)
            it = plt.plot(self.infected_time_series, infected_theoretical, label='Infected Theoretical')
            ir = plt.plot(self.infected_time_series, infected_real_data, label='Infected Real')
            plt.xlabel('Time (days)')
            plt.ylabel('Count')
            plt.yscale('log')
            plt.legend()
            plt.show()

        return prediction_error


def main():
    x = np.array([3.00, 0.60, 0.30, 0.20, 0.95,
                  0.44, 0.51, 0.42, 0.02, 0.13,
                  0.48, 0.22, 0.33, 0.80])  # Optimized values from MATLAB code
    x = np.array([1.76592537, 0.64402601, 0.22258755, 0.38885279])  # From diff_evo

    ro_bounds = [(0, 10)]
    beta_bounds = [(0, 2)]
    dead_rate_bounds = [(0, 1)] * 2

    bounds = ro_bounds + beta_bounds + dead_rate_bounds
    number_of_variables = len(bounds)
    solve_ode = SolveODE()
    result = differential_evolution(solve_ode.ode, bounds)
    print(result)
    x = result.x

    # x = np.random.random(4)
    # print('>> ', x)
    solve_ode.ode(x, 'plot')

if __name__ == '__main__':
    main()
