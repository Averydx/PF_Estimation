from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Resampler import Resampler
from scipy.stats import poisson
from typing import List
import numpy as np
from numpy.typing import NDArray


def likelihood(observation,particle_observations:NDArray[np.int_])->NDArray: 
    return poisson.pmf(observation,particle_observations)


class PoissonResample(Resampler): 
    def __init__(self) -> None:
        super().__init__(likelihood)


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[np.float_]:

        weights = np.array(self.likelihood(np.round(observation),[particle.observation for particle in particleArray]))


        for j in range(len(particleArray)):  
            if(weights[j] == 0):
                weights[j] = 10**-300 
            elif(np.isnan(weights[j])):
                weights[j] = 10**-300
            elif(np.isinf(weights[j])):
                weights[j] = 10**-300


        weights = weights/np.sum(weights)

        
        return np.squeeze(weights)
    
    def resample(self, weights: NDArray[np.float_], ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        return super().resample(weights, ctx,particleArray)
