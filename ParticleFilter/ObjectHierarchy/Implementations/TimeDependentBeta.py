
from numpy import float_, int_
from numpy.typing import NDArray
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import *
from utilities.utility import multivariate_normal
from typing import Tuple,List,Dict
import numpy as np

from ObjectHierarchy.Utils import Context

class TimeDependentAlgo(Algorithm): 

    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,context:Context) -> None:
        super().__init__(integrator, perturb, resampler)
        self.context = context


    '''Basic initialization function, these functions will always call back to the parent for the basic setup, just initialize the params as a dictionary'''

    #TODO Think about passing all param initialization back to the parent, no reason to have it be user defined, unless theres an elaborate prior, but the prior doesn't matter that much 
    def initialize(self) -> None:
        params = {"beta":-1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0} #Initialize estimated parameters to -1 so the parent knows which params not to touch
        super().initialize(params)
        #initialization of the estimated parameters will be done in the override
        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''

            beta = self.context.rng.uniform(0.,1.)
            self.particles[i].param['beta'] = beta

        
    @timing
    def run(self,info:RunInfo) ->Output:


        '''field initializations for Output'''
        output = Output(observation_data=info.observation_data)
        


        while self.context.clock.time < len(info.observation_data): 
            self.particles = self.integrator.propagate(self.particles)

            weights = self.resampler.compute_weights(info.observation_data[self.context.clock.time],self.particles)
            self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)

            self.particles = self.perturb.randomly_perturb(ctx=self.context,particleArray=self.particles)

            '''output updates, not part of the main algorithm'''
            output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
            output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])

            self.context.clock.tick()
            print(f"iteration: {self.context.clock.time}")

        print(output.beta_qtls)
        return output
    


class Euler(Integrator): 
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle])->List[Particle]: 


        for j,particle in enumerate(particleArray): 
            dt,sim_obv =self.RHS_H(particle)

            particleArray[j].state += dt
            particleArray[j].observation = np.array([sim_obv])

        return particleArray
    

    def RHS_H(self,particle:Particle):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = particle.state
        N = S + I + R + H 

        new_H = ((1/particle.param['D'])*particle.param['gamma']) * I   

        dS = -particle.param['beta']*(S*I)/N + (1/particle.param['L'])*R 
        dI = particle.param['beta']*S*I/N-(1/particle.param['D'])*I
        dR = (1/particle.param['hosp']) * H + ((1/particle.param['D'])*(1-(particle.param['gamma']))*I)-(1/particle.param['L'])*R 
        dH = (1/particle.param['D'])*(particle.param['gamma']) * I - (1/particle.param['hosp']) * H 

        return np.array([dS,dI,dR,dH]),new_H
    

class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,ctx:Context,particleArray:List[Particle]):
        C = np.diag([(self.hyperparameters['sigma1']/ctx.population) ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] **2,
                     self.hyperparameters['sigma2'] ** 2]).astype(float)
        
        
        A = np.linalg.cholesky(C)
        for i,_ in enumerate(particleArray): 

            #variation of the state and parameters

            perturbed = np.log(np.concatenate((particleArray[i].state,[particleArray[i].param['beta']])))

            perturbed = np.exp(multivariate_normal(perturbed,A))
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population


            particleArray[i].state = perturbed[0:ctx.state_size]
            particleArray[i].param['beta'] = perturbed[-1]
            



        return particleArray
    

class PoissonResample(Resampler): 
    def __init__(self, likelihood) -> None:
        super().__init__(likelihood)


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[float_]:

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
    
    def resample(self, weights: NDArray[float_], ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        return super().resample(weights, ctx,particleArray)

    