from typing import Dict
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.context import Compartments, SimDType
from epymorph.util import check_ndarray
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Context,Particle,quantiles,timing
import matplotlib.pyplot as plt

import numpy as np



class Epymorph_PF(Algorithm): 
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler, ctx: Context) -> None:
        super().__init__(integrator, perturb, resampler, ctx)

    def initialize(self, params: Dict) -> None:

        '''Type and shape checking to prevent against invalid data'''
        check_ndarray(value=self.ctx.observation_data[0,:],dtype=[np.int64,np.int32,np.int16],shape=(self.ctx.geo.nodes,))

        '''call the super to perform checking for estimated parameters '''
        super().initialize(params=params)

        for _ in range(self.ctx.particle_count): 
            for param in self.ctx.estimated_params:
                params[param] = self.ctx.rng.uniform(0.,1.)

            '''Draw a random int to represent the initial infected'''
            pops = self.ctx.geo.data['population'] 
            initial_infected = self.ctx.rng.integers(0,np.round(self.ctx.seed_size*pops[0]))
            state = []

            for index,pop in enumerate(pops): 
                if(index == 0):
                    substate = [pop-initial_infected,initial_infected]
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                else: 
                    substate = [pop,0]
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                state.append(substate)

            state = np.array(state)      
            observation = np.array([0 for _ in range(self.ctx.geo.nodes)])

            self.particles.append(Particle(param=params.copy(),state=state.copy(),observation=observation))

    def run(self)->Output:
        '''field initializations for Output'''
        self.output = Output(observation_data=self.ctx.observation_data)

        '''epymorph requires that we store the tick index'''
        tick_index = 0

        '''main run loop'''
        while self.ctx.clock.time < int((self.ctx.observation_data[:,0].size)): 
                
            self.particles = self.integrator.propagate(self.particles,self.ctx,tick_index)



    #             weights = self.resampler.compute_weights(info.observation_data[:,self.context.clock.time],self.particles)
    #             self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)
    #             self.particles = self.perturb.randomly_perturb(ctx=self.context,particleArray=self.particles)

    #             # '''output updates, not part of the main algorithm'''
    #             #self.output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
    #             print(np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)],axis=0))
    #             # self.output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])

    #             self.context.clock.tick()
    #             print(f"iteration: {self.context.clock.time}")





    # @timing
    # def run(self,info:RunInfo) ->Output:

    #         average_beta = np.zeros(300)

  
    
    #         self.clean_up()

    #         return self.output
    



        