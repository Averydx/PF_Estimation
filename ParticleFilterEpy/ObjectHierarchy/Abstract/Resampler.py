from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List,Dict
from ObjectHierarchy.utilities.Utils import Context,Particle




class Resampler(ABC): 
    '''Resamplers take a likelihood function as an argument and compute the weights based on a user defined implementation of compute_weights, if making an implementation of the class, call
super().resample(args) to resample the particles, there's no difference in resampling between implementations'''

    likelihood:FunctionType | BuiltinFunctionType #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 
    Flags: Dict[str,int]

    '''Protects against invalid constructor arguments, i.e not a function'''
    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    '''No base implementation, user defined for child implementation'''
    @abstractmethod
    def compute_weights(self,observation:NDArray,particleArray:List[Particle])->NDArray[float_]: #implementations of compute_weights call the Resamplers _likelihood function in the computation
        pass

    '''Resamples the particle array based on the weights computed in compute_weights(args)'''
    @abstractmethod
    def resample(self,weights:NDArray[float_],ctx:Context,particleArray:List[Particle]) ->List[Particle]: #Resamples based on the weights returned from compute_weights and returns the new indexes
        
        indexes = np.arange(ctx.particle_count)
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=weights)

        '''Making a shallow copy here will be necessary, the resampling process will shift the indexes while running, leading to slightly inaccurate results'''
        particleCopy = particleArray.copy()

        '''Instantiate new particles in the array will the given data at the new index'''
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),particleCopy[new_particle_indexes[i]].state.copy(),particleCopy[new_particle_indexes[i]].observation)


        

        return particleArray




    