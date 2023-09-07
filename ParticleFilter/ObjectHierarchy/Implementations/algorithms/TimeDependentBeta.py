
from numpy import float_, int_
from numpy.typing import NDArray
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import *
from utilities.utility import multivariate_normal
from typing import Tuple,List,Dict
import numpy as np

from ObjectHierarchy.utilities.Utils import Context

class TimeDependentAlgo(Algorithm): 

    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,context:Context) -> None:
        super().__init__(integrator, perturb, resampler,context)
        


    '''Basic initialization function, these functions will always call back to the parent for the basic setup, just initialize the params as a dictionary'''

    #TODO Think about passing all param initialization back to the parent, no reason to have it be user defined, unless theres an elaborate prior, but the prior doesn't matter that much 
    def initialize(self,params) -> None:
        super().initialize(params)
        #initialization of the estimated parameters will be done in the override
        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''

            beta = self.context.rng.uniform(0.,1.)
            self.particles[i].param['beta'] = beta

        
    @timing
    def run(self,info:RunInfo) ->Output:


        '''field initializations for Output'''
        self.output = Output(observation_data=info.observation_data)
        self.output_flags = info.output_flags

        while self.context.clock.time < len(info.observation_data): 
            
            self.particles = self.integrator.propagate(self.particles,self.context)

            weights = self.resampler.compute_weights(info.observation_data[self.context.clock.time],self.particles)
            self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)

            self.particles = self.perturb.randomly_perturb(ctx=self.context,particleArray=self.particles)

            '''output updates, not part of the main algorithm'''
            self.output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
            self.output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])
            self.output.average_beta[self.context.clock.time] = np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)])
            print(([particle.state for _,particle in enumerate(self.particles)]))



            self.context.clock.tick()
            #print(f"iteration: {self.context.clock.time}")

        self.clean_up()
        return self.output
    


    