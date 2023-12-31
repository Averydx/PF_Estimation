from dataclasses import dataclass,field
import numpy as np
from numpy.typing import NDArray
from numpy import random,int_,float_
from typing import Dict,List
from functools import wraps
from time import perf_counter

@dataclass(frozen=True)
class RunInfo: 
    observation_data: NDArray[int_] #Array of observation data that will be passed to the Algorithm, dimension agnostic 
    forecast_time:int # optional param to indicate the amount of time series to forecast 

class Clock: 
    time: int
    def __init__(self) -> None:
        self.time = 0
    
    def tick(self):
        self.time +=1

@dataclass
class Particle: 
    param: Dict
    state: NDArray
    observation: NDArray[int_]

@dataclass(frozen=True)
class Context: 
    particle_count: int = 1000
    clock: Clock = field(default_factory=lambda: Clock())
    rng:random.Generator = field(default_factory=lambda: np.random.default_rng())
    data_scale:int = 1 #optional param to indicate the scale of the data i.e. the number number of days between each data point
    seed_size: float = 0.01 #estimate of initial percentage of infected out of the total population
    population: int = 100000 #estimate of the total population 
    state_size: int = 4 #number of state variables in the model 
    estimated_params: List[str] = field(default_factory=lambda: []) #number of estimated parameters in the model 


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print('func:%r args:[%r, %r] took: %2.4f sec' % \
          (f.__name__, args, kw, te-ts))
        return result
    return wrap

def quantiles(item:List)->List: 
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
        return list(np.quantile(item, qtlMark))








