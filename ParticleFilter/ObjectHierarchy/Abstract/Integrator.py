from abc import ABC,abstractmethod
from typing import Tuple,List
from numpy.typing import NDArray
from numpy import int_
from Utils import Particle

class Integrator(ABC): 

    @abstractmethod
    def propagate(self,particleArray:List[Particle])->List[Particle]: #Propagates the state forward one step and returns an array of states and observations across the the integration period
        pass
    #Note: observations are forced to be integers, if using a deterministic integrator round off values before returning them from propagate

    