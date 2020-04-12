import numpy as np
import pandas as pd


class SEIHParameters:
    def __init__(self):
        self.contact: np.ndarray = None
        self.severe: float = None
        self.alpha: float = None
        self.beta: np.ndarray = None
        self.gamma: np.ndarray = None
        self.delta: np.ndarray = None
        self.N: float = None
        self.In: float = None
        self.Hlimit: np.ndarray = None
        self.drate: np.ndarray = None
        self.ro: float = None
        self.k: float = None  # TODO Verify at runtime (w/ __set__) that 0<=k<=1


class InitialConditions:
    def __init__(self):
        self.t = None  # TODO Add datetime type (datetime64 or something like that)
        self.Q: float = None
        self.H: float = None
        self.D: float = None


class TimeIntervals:
    def __init__(self, intervals):
        self.intervals = pd.to_datetime(intervals)

    def __len__(self):
        return len(self.intervals)

    def __str__(self):
        return self.intervals.__str__()

    def __repr__(self):
        return self.__str__()


class RealData:
    def __init__(self, infected: pd.DataFrame, dead: pd.DataFrame, recovered: pd.DataFrame):
        self.infected: pd.DataFrame = infected.rename(columns={'count': 'infected'})
        self.dead: pd.DataFrame = dead.rename(columns={'count': 'dead'})
        self.recovered: pd.DataFrame = recovered.rename(columns={'count': 'recovered'})
