
from numpy import float_, int_
from numpy.typing import NDArray
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import *
from typing import Tuple,List,Dict
import numpy as np

from ObjectHierarchy.Utils import Context

class TimeDependentAlgo(Algorithm): 

    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler) -> None:
        super().__init__(integrator, perturb, resampler)
        self.context = Context(particle_count=5,clock=Clock(),rng=random.default_rng(),data_scale=1,seed_size=0.01,population=100000,state_size=4,estimated_params=[])


    '''Basic initialization function, these functions will always call back to the parent for the basic setup, just initialize the params as a dictionary'''

    #TODO Think about passing all param initialization back to the parent, no reason to have it be user defined, unless theres an elaborate prior, but the prior doesn't matter that much 
    def initialize(self) -> None:
        params = {"beta": -1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0} #Initialize estimated parameters to -1 so the parent knows which params not to touch
        super().initialize(params)
        #initialization of the estimated parameters will be done in the override
        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''
            beta = self.context.rng.uniform(0,1)
            self.particles[i].param['beta'] = beta
        

    def run(self,info:RunInfo) ->Output:
        return super().run(info=info)
    


class Euler(Integrator): 
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle])->List[Particle]: 


        for j,particle in enumerate(particleArray): 
            dt,sim_obv =self.RHS_H(particle)


            particleArray[j].state += dt
            particleArray[j].observation = sim_obv
        return particleArray
    
    def RHS_H(self,particle:Particle):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = particle.state
        N = S + I + R + H 

        new_H = ((1/particle.param['D'])*particle.param['gamma']) * I   

        dS = -(particle.param['beta']*(S*I))/N + (1/particle.param['L'])*R 
        dI = (particle.param['beta']*S*I)/N-(1/particle.param['D'])*I
        dR = (1/particle.param['hosp']) * H + ((1/particle.param['D']) * (1- particle.param['gamma'])) * I -(1/particle.param['L']) * R
        dH = (1/particle.param['D']) * particle.param['gamma'] * I - (1/particle.param['hosp']) * H 
        
        return np.array([dS,dI,dR,dH]),new_H
    



class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,ctx:Context,particleArray:List[Particle]):

        state_var = np.array([self.hyperparameters['sigma1'] for _ in range(ctx.state_size)])
        param_var = np.array([self.hyperparameters['sigma2'] for _ in range(len(ctx.estimated_params))])

        state_var[0] = state_var[0]/ctx.population

        cov = np.diag(np.concatenate((state_var,param_var))).astype(float)

        for i in range(len(particleArray)): 
            state = particleArray[i].state
            params = [particleArray[i].param[item] for _,item in enumerate(ctx.estimated_params)]

            total_state = np.concatenate((state,params))
 
            perturbed_total_state = np.exp(ctx.rng.multivariate_normal(np.log(total_state),cov))
            
            perturbed_total_state = perturbed_total_state/np.sum(perturbed_total_state) * ctx.population

            particleArray[i].state = perturbed_total_state[:ctx.state_size]
            for j,param in enumerate(ctx.estimated_params): 
                particleArray[i].param[param] = perturbed_total_state[ctx.state_size+j]

        return particleArray
    

class PoissonResample(Resampler): 
    def __init__(self, likelihood) -> None:
        super().__init__(likelihood)


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[float_]:

        weights = np.array(self.likelihood(observation,[particle.observation for particle in particleArray]))

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

    