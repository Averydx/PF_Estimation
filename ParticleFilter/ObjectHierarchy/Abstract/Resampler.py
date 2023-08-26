from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from ObjectHierarchy.Utils import Context

class Resampler(ABC): 

    _likelihood:object #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 

    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self._likelihood = likelihood

    @abstractmethod
    def compute_weights(self,observation:int,particle_observation:int)->NDArray[float_]: #implementations of compute_weights call the Resamplers _likelihood function in the computation
        pass

    def resample(self,weights:NDArray[float_],ctx:Context) ->NDArray[int_]: #Resamples based on the weights returned from compute_weights and returns the new indexes
        indexes = np.arange(ctx.particle_count)
        new_particle_indexes = np.random.choice(a=indexes, size=ctx.particle_count, replace=True, p=weights)
        return new_particle_indexes




    