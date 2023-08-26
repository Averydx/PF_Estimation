from dataclasses import dataclass,field
from numpy.typing import NDArray
from numpy import int_,float_
from typing import Optional

@dataclass(frozen=True)
class RunInfo: 
    observation_data: NDArray[int_] #Array of observation data that will be passed to the Algorithm, dimension agnostic 
    forecast_time:int # optional param to indicate the amount of time series to forecast 
    particle_count:int  #optional number of particles to use in the algorithm, defaults to 1000
    data_scale:int #optional param to indicate the scale of the data i.e. the number number of days between each data point


class Clock: 
    time: int
    def __init__(self) -> None:
        self.time = 0
    
    def tick(self):
        self.time +=1



@dataclass
class Particle: 
    param: NDArray[float_]
    state: NDArray
    observation: int

@dataclass(frozen=True)
class Context: 
    particle_count: int
    clock: Clock







