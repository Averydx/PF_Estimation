from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List
from ObjectHierarchy.Utils import Context,Particle

class Resampler(ABC): 

    likelihood:FunctionType | BuiltinFunctionType #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 

    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    @abstractmethod
    def compute_weights(self,observation:int,particleArray:List[Particle])->NDArray[float_]: #implementations of compute_weights call the Resamplers _likelihood function in the computation
        pass

    @abstractmethod
    def resample(self,weights:NDArray[float_],ctx:Context,particleArray:List[Particle]) ->List[Particle]: #Resamples based on the weights returned from compute_weights and returns the new indexes
        
        indexes = np.arange(ctx.particle_count)
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=weights)
        particle_copy = particleArray.copy()

        for i,_ in enumerate(particleArray):
            particleArray[i] = particle_copy[int(new_particle_indexes[i])]

        return particleArray




    