import numpy as np
from numpy.typing import NDArray
from typing import List
from dataclasses import dataclass,field

@dataclass
class Output: 
    observation_data: NDArray #holds the time series observation data
    observation_qtls: NDArray = field(init=False) #quantiles of the estimated observations at each time point
    beta_qtls: NDArray = field(init=False) #quantiles of the estimated beta at each time point
    time_series: int = field(init=False) #the length of the time series data
    average_beta: NDArray = field(init=False) #holds the average estimated beta across the time series
    average_state: NDArray = field(init=False)#hold the average estimated state across the time series

    '''A hack to initialize the fields to zeros without explicit initialization'''
    def __post_init__(self):
        object.__setattr__(self, 'time_series', len(self.observation_data))
        object.__setattr__(self,'beta_qtls', np.zeros((23,self.time_series)))
        object.__setattr__(self,'observation_qtls', np.zeros((23,self.time_series)))   
        object.__setattr__(self,'average_beta',np.zeros(self.time_series))  
        object.__setattr__(self,'average_state',np.zeros((self.time_series,4)))


        



    

