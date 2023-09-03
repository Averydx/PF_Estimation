from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Utils import variance
from scipy.stats import poisson,nbinom,norm
from typing import List
import numpy as np
from numpy.typing import NDArray

'''Likelihood functions'''
def likelihood_poisson(observation,particle_observations:NDArray[np.int_])->NDArray: 
    return poisson.pmf(k=observation,mu=particle_observations)

def likelihood_NB(observation,particle_observations:NDArray[np.int_],var: float)->NDArray: 
    X = np.zeros_like(particle_observations)

    for i,P_obv in enumerate(particle_observations): 
       X[i] = nbinom.pmf(k=observation,n=var,p=var/(P_obv + var))

    return X

def likelihood_normal(observation,particle_observations:NDArray[np.int_],var:float)->NDArray: 
    return norm.pdf(x=observation,loc=particle_observations,scale = var)

'''Resampler using the normal probability density function to compute the weights'''
class NormResample(Resampler):


    def __init__(self,var) -> None:
        super().__init__(likelihood_normal)
        self.var = var

    def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[np.float_]:

        weights = np.array(self.likelihood(np.round(observation),[particle.observation for particle in particleArray],self.var))


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
    
#TODO Fix this 
# '''Resampler using the negative binomial probability mass function to compute the weights'''
# class NBResample(Resampler):

#     observations: NDArray[np.int_]

#     def __init__(self,observations:NDArray[np.int_]) -> None:
#         super().__init__(likelihood_NB)
#         self.observations = observations

#     def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[np.float_]:

#         weights = np.array(self.likelihood(np.round(observation),[particle.observation for particle in particleArray],var))


#         for j in range(len(particleArray)):  
#             if(weights[j] == 0):
#                 weights[j] = 10**-300 
#             elif(np.isnan(weights[j])):
#                 weights[j] = 10**-300
#             elif(np.isinf(weights[j])):
#                 weights[j] = 10**-300


#         weights = weights/np.sum(weights)

        
#         return np.squeeze(weights)
    
#     def resample(self, weights: NDArray[np.float_], ctx: Context,particleArray:List[Particle]) -> List[Particle]:
#         return super().resample(weights, ctx,particleArray)


'''Resampler using the poisson probability mass function to compute the weights'''
class PoissonResample(Resampler): 

    def __init__(self) -> None:
        super().__init__(likelihood_poisson)


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
