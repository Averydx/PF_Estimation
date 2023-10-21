from abc import ABC,abstractmethod
from numpy.typing import NDArray
import numpy as np
from numpy import float_,int_
from types import FunctionType,BuiltinFunctionType
from typing import List,Callable
from ObjectHierarchy.utilities.Utils import Context,Particle
from copy import deepcopy

class Resampler(ABC): 

    likelihood:Callable #A function that returns a likelihood given a real observation and a simulated observation corresponding to a particle 

    def __init__(self,likelihood) -> None:
        if not isinstance(likelihood,(FunctionType,BuiltinFunctionType)): 
            raise Exception("Likelihood is not a function")
        self.likelihood = likelihood

    @abstractmethod
    def compute_weights(self,observation:NDArray,particleArray:List[Particle])->NDArray[float_]: #implementations of compute_weights call the Resamplers _likelihood function in the computation
        pass

    @abstractmethod
    def resample(self,weights:NDArray[float_],ctx:Context,particleArray:List[Particle]) ->List[Particle]: #Resamples based on the weights returned from compute_weights and returns the new indexes
        '''This is a basic resampling method, more advanced methods like systematic resampling need to override this'''    
    
        indexes = np.arange(ctx.particle_count) #create a cumulative ndarray from 0 to particle_count

        #The numpy resampling algorithm, see jupyter notebnook resampling.ipynb for more details
        new_particle_indexes = ctx.rng.choice(a=indexes, size=ctx.particle_count, replace=True, p=weights)

        particleCopy = particleArray.copy()#copy the particle array refs to ensure we don't overwrite particles

        #this loop reindexes the particles by rebuilding the particles
        for i in range(len(particleArray)): 
            particleArray[i] = Particle(particleCopy[new_particle_indexes[i]].param.copy(),
                                        particleCopy[new_particle_indexes[i]].state.copy(),
                                        particleCopy[new_particle_indexes[i]].observation,
                                        particleCopy[new_particle_indexes[i]].dispersion)


        

        return particleArray




    