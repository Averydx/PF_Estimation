from types import FunctionType

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
        self.context = Context(particle_count=10,clock=Clock(),rng=random.default_rng(),data_scale=1,seed_size=100,population=100000,state_size=4)

    def initialize(self) -> None:
        params = {"beta": -1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0} #Initialize estimated parameters to -1 so the parent knows which params not to touch
        super().initialize(params)
        #initialization of the estimated parameters will be done in the override
        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''
            beta = self.context.rng.uniform(0,1)
            self.particles[i].param['beta'] = beta
        
    def run(self,info:RunInfo) ->Output:
        if info.forecast_time > 0 and (info.observation_data - info.forecast_time) > 5:
            pass
        else: 
            while self.context.clock.time < len(info.observation_data): 
                self.particles=self.integrator.propagate(self.particles)
                    
                weights = np.squeeze(self.resampler.compute_weights(info.observation_data[self.context.clock.time],self.particles))

                self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)
                self.particles = self.perturb.randomly_perturb(self.particles)

                self.context.clock.tick()

        return Output()
    


class Euler(Integrator): 
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle])->List[Particle]: 
        return particleArray
    



class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,particleArray:List[Particle]):
        return particleArray
    

class PoissonResample(Resampler): 
    def __init__(self, likelihood) -> None:
        super().__init__(likelihood)


#TODO Debug invalid weights in divide 
    def compute_weights(self, observation: int, particleArray:List[Particle]) -> NDArray[float_]:
        weights = np.array(self.likelihood(observation,[particle.observation for particle in particleArray]))
        weights = weights/np.sum(weights)
        weights = np.nan_to_num(weights,nan= 10e-300,posinf=10e-300,neginf=10e-300)
        return weights
    
    def resample(self, weights: NDArray[float_], ctx: Context,particleArray:List[Particle]) -> List[Particle]:
        return super().resample(weights, ctx,particleArray)

    