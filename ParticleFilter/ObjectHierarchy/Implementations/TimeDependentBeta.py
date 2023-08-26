from types import FunctionType

from numpy import float_, int_
from numpy.typing import NDArray
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import *
from typing import Tuple,List,Dict
import numpy as np

from ObjectHierarchy.Utils import Context

class TimeDependentAlgo(Algorithm): 

    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler) -> None:
        super().__init__(integrator, perturb, resampler)

    def run(self,info:RunInfo) ->Output:
        return Output()
    


class Euler(Integrator): 
    def __init__(self) -> None:
        super().__init__()

    def propagate(self,particleArray:List[Particle])->Tuple[NDArray,NDArray[int_]]: #Propagates the state forward one step and returns an array of states and observations across the the integration period
        return (np.array(0),np.array(0))
    



class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,particleArray:List[Particle]):
        return super().randomly_perturb(particleArray)
    




class PoissonResample(Resampler): 
    def __init__(self, likelihood) -> None:
        super().__init__(likelihood)


    def compute_weights(self, observation: int, particle_observation: int) -> NDArray[float_]:
        return np.array(0)
    
    def resample(self, weights: NDArray[float_], ctx: Context) -> NDArray[int_]:
        return super().resample(weights, ctx)

    