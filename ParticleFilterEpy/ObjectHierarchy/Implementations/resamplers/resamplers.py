from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Utils import variance
from scipy.stats import poisson,nbinom,norm,multivariate_normal
from typing import List
import numpy as np
from numpy.typing import NDArray

'''Likelihood functions'''
def likelihood_poisson(observation,particle_observations:NDArray[np.int_])->NDArray: 
        return poisson.pmf(k=observation,mu=particle_observations)

def likelihood_NB(observation,particle_observations:NDArray[np.int_],var:float)->NDArray: 
    X = np.zeros(len(particle_observations))

    for i,P_obv in enumerate(particle_observations): 
       X[i] = nbinom.pmf(observation,var,var/(P_obv + var))

    return X

def likelihood_normal(observation,particle_observations:NDArray[np.int_],var)->NDArray: 
    return norm.pdf(observation,loc=particle_observations,scale=var)

def joint_likelihood_normal(observation:NDArray[np.int_],particle_observations:NDArray[np.int_],cov:int): 
    return multivariate_normal.pdf(observation,mean = particle_observations,cov = cov)

'''Resampler using the negative binomial probability mass function to compute the weights'''
class NBResample(Resampler):

    var: float

    def __init__(self,var) -> None:
        super().__init__(likelihood_NB)
        self.var = var
        self.Flags = {"all_size_valid":False}

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


'''Resampler using the poisson probability mass function to compute the weights'''
class PoissonResample(Resampler): 

    def __init__(self) -> None:
        super().__init__(likelihood_poisson)
        self.Flags = {"all_size_valid":False}


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

'''resampler using the multivariate normal distribution for resampling, note-the standard deviation must be very large for high-dimensional probability spaces(for R^6 I set it to 10000000)'''
class MultivariateNormalResample(Resampler):

    cov: int

    def __init__(self,cov) -> None:
        super().__init__(joint_likelihood_normal)
        self.Flags = {"all_size_valid":True}
        self.cov = cov


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float_]:
        p_obvs = np.array([particle.observation for particle in particleArray])
        weights = np.zeros(len(p_obvs))
        for i,particle in enumerate(particleArray):
            weights[i] = joint_likelihood_normal(observation=observation,particle_observations=particle.observation,cov=self.cov)

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
    

    
