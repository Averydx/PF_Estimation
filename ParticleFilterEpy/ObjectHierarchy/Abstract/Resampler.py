from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List,Dict
from ObjectHierarchy.utilities.Utils import Context,Particle
from copy import deepcopy

class Resampler(ABC): 

    likelihood:FunctionType | BuiltinFunctionType #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 
    Flags: Dict[str,int]

    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    @abstractmethod
    
    def compute_weights(self,observation:NDArray,particleArray:List[Particle])->float: #implementations of compute_weights call the Resamplers _likelihood function in the computation
        '''Returns a float which is the average log probability of particles '''
        pass

    @abstractmethod
    def resample(self,ctx:Context,particleArray:List[Particle]) ->List[Particle]: #Resamples based on the weights returned from compute_weights and returns the new indexes
        
        weights = [particle.weight for particle in particleArray]
        indexes = np.arange(ctx.particle_count)
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=weights)


        particleCopy = particleArray.copy()
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),particleCopy[new_particle_indexes[i]].state.copy(),particleCopy[new_particle_indexes[i]].observation,particleCopy[new_particle_indexes[i]].weight.copy())


        

        return particleArray




    