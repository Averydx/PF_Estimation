from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Utils import variance,log_norm
from scipy.stats import poisson,nbinom,norm,multivariate_normal
from typing import List
import numpy as np
from numpy.typing import NDArray

'''Likelihood functions'''
def likelihood_poisson(observation,particle_observations:NDArray[np.int_])->NDArray: 
        return poisson.pmf(k=observation,mu=particle_observations)

def log_likelihood_poisson(observation,particle_observations:NDArray[np.int_])->NDArray: 
        return poisson.logpmf(k=observation,mu=particle_observations)

def likelihood_NB(observation,particle_observations:NDArray[np.int_],var:float)->NDArray: 
    X = np.zeros(len(particle_observations))

    for i,P_obv in enumerate(particle_observations): 
       X[i] = nbinom.pmf(observation,var,var/(P_obv + var))

    return X

def likelihood_normal(observation,particle_observations:NDArray[np.int_],var)->NDArray: 
    return norm.pdf(observation,loc=particle_observations,scale=var)

def joint_likelihood_normal(observation:NDArray[np.int_],particle_observations:NDArray[np.int_],cov:int): 
    return multivariate_normal.pdf(observation,mean = particle_observations,cov = cov)

def log_likelihood_normal(observation:NDArray[np.int_],particle_observations:NDArray[np.int_],cov:int): 
    return norm.logpdf(observation,particle_observations,cov)

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
            weights[i] = self.likelihood(observation=observation,particle_observations=particle.observation,cov=self.cov)

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
    
'''log multivariate poisson resampler --not using a poisson model with covariance between the random variables, 
assumes every observation in the observation vector is a poisson random variable'''
class LogMultivariatePoissonResample(Resampler): 
  
    def __init__(self) -> None:
        super().__init__(log_likelihood_poisson)
        self.Flags = {"all_size_valid":True}


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> float:
        p_obvs = np.array([particle.observation for particle in particleArray])
        weights = np.zeros(len(p_obvs))
        for i,particle in enumerate(particleArray):
            for j in range(len(particle.observation)): 
                weights[i] += (self.likelihood(observation=observation[j],particle_observations=particle.observation[j]))

        weights = weights-np.max(weights)
        #weights = log_norm(weights)
        
        weights = np.exp(weights)
        weights /= np.sum(weights)

        for j in range(len(particleArray)):  
            if(weights[j] == 0):
                weights[j] = 0
            elif(np.isnan(weights[j])):
                weights[j] = 0
            elif(np.isinf(weights[j])):
                weights[j] = 0

            particleArray[j].weight = weights[j]

        print(weights)        

        

        return weights
    
    def resample(self, ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        
        # log_cdf = np.zeros(ctx.particle_count)
        # log_cdf[0] = weights[0]
        # for j in range(1,ctx.particle_count): 
        #     log_cdf[j] = max(weights[j],log_cdf[j-1]) + np.log(1 + np.exp(-1*np.abs(log_cdf[j-1] - weights[j])))

        # i = 0
        # indices = np.zeros(ctx.particle_count)
        # u = np.zeros(ctx.particle_count)
        # u[0] = ctx.rng.uniform(0,1/ctx.particle_count)
        # for j in range(0,ctx.particle_count): 
        #     u[j] = np.log(u[0] + 1/ctx.particle_count * j)
        #     while u[j] > log_cdf[i]: 
        #         i += 1
        #     indices[j] = i

        # indices=indices.astype(int)
        # particleCopy = particleArray.copy()
        # for i in range(len(particleArray)): 
        #     particleArray[i] = Particle(particleCopy[indices[i]].param.copy(),particleCopy[indices[i]].state.copy(),particleCopy[indices[i]].observation)

        '''Resample generally calls out to the super to do the actual resampling, although a custom resampler can override the base implementation'''
        return super().resample(ctx,particleArray)
    
'''It's possible a distribution that has a variance parameter might work more nicely for our purposes in the multivariate case'''
class LogNormalResample(Resampler): 
    cov: int

    def __init__(self,cov) -> None:
        super().__init__(log_likelihood_normal)
        self.Flags = {"all_size_valid":True}
        self.cov = cov

#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: NDArray, particleArray:List[Particle]) -> NDArray[np.float_]:
        p_obvs = np.array([particle.observation for particle in particleArray])
        weights = np.zeros(len(p_obvs))
        for i,particle in enumerate(particleArray):
            for j in range(len(particle.observation)): 
                weights[i] += (self.likelihood(observation=observation[j],particle_observations=particle.observation[j],cov = self.cov))

        weights = weights-np.max(weights)
        #weights = log_norm(weights)
       
        weights = np.exp(weights)
        for j in range(len(particleArray)):  
            if(weights[j] == 0):
                weights[j] = 10**-300 
            elif(np.isnan(weights[j])):
                weights[j] = 10**-300
            elif(np.isinf(weights[j])):
                weights[j] = 10**-300

        weights /= np.sum(weights)
        return np.squeeze(weights)
    
    def resample(self, weights: NDArray[np.float_], ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        return super().resample(weights, ctx,particleArray)