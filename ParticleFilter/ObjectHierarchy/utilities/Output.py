import numpy as np
from numpy.typing import NDArray
from typing import List
from dataclasses import dataclass,field

@dataclass
class Output: 
    observation_data: NDArray
    observation_qtls: NDArray = field(init=False)
    beta_qtls: NDArray = field(init=False)
    time_series: int = field(init=False)
    average_beta: NDArray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'time_series', len(self.observation_data))
        object.__setattr__(self,'beta_qtls', np.zeros((23,self.time_series)))
        object.__setattr__(self,'observation_qtls', np.zeros((23,self.time_series)))   
        object.__setattr__(self,'average_beta',np.zeros(self.time_series))     


        



    

