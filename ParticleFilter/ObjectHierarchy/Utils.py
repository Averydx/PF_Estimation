from dataclasses import dataclass,field
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
    particle_count: int
    clock: Clock
    rng:random.Generator
    data_scale:int #optional param to indicate the scale of the data i.e. the number number of days between each data point
    seed_size: float #estimate of initial percentage of infected out of the total population
    population: int #estimate of the total population 
    state_size: int #number of state variables in the model 
    estimated_params: List[str] #number of estimated parameters in the model 


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








