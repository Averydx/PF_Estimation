from abc import ABC,abstractmethod
from typing import Tuple,List
from numpy.typing import NDArray
from numpy import int_
from ObjectHierarchy.Utils import Particle

class Integrator(ABC): 

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    @abstractmethod
    def propagate(self,particleArray:List[Particle])->List[Particle]: 
        pass
    '''Note: observations are forced to be integers, if using a deterministic integrator round off values before returning them from propagate'''

    