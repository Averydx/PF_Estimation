from typing import Dict
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.context import Compartments, SimDType
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Context,Particle,quantiles,timing
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
import matplotlib.pyplot as plt
import numpy as np



class Epymorph_PF(Algorithm): 
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler, ctx: Context) -> None:
        super().__init__(integrator, perturb, resampler, ctx)

    def initialize(self, params: Dict,infection_location:int) -> None:

        '''Error handling for infection_location'''
        if infection_location < 0 or infection_location > self.ctx.geo.nodes-1:
            raise Exception("Infection location is out of bounds")


        '''Note, it is required to invoke the super initializer to get the list of estimated params'''
        super().initialize(params=params)



        '''Initializes the estimated parameters to via a uniform prior '''
        for _ in range(self.ctx.particle_count): 
            '''initializes the estimated parameters'''
            for param in self.ctx.estimated_params:
                params[param]= np.array([self.ctx.rng.uniform(0.,1.) for _ in range(len(params[param]))])


            '''Draw a random int to represent the initial infected'''
            pops = self.ctx.geo.data['population'] 
            initial_infected = self.ctx.rng.integers(0,np.round(self.ctx.seed_size*pops[0]))
            state = []

            '''Constructs the state array, if the index is 0 place the infected seed there, else append the population in S and every other compartment to zeroes'''
            for index,pop in enumerate(pops): 
                if(index == infection_location):
                    substate = [pop-initial_infected,initial_infected]
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                else: 
                    substate = [pop,0]
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                state.append(substate)

            state = np.array(state)   

            '''Initialize observations as an array of zeroes'''   
            observation = np.array([0 for _ in range(self.ctx.geo.nodes)])

            '''Create the Particle with the computed fields'''
            self.particles.append(Particle(param=params.copy(),state=state.copy(),observation=observation))

    @timing
    def run(self)->Output:
        '''field initializations for Output'''
        self.output = Output(observation_data=self.ctx.observation_data)

        beta=[]

        '''main run loop- note ctx.observation_data has been enforced to shape TxN(time by nodes), and therefore axis 0 is used for loop iteration'''
        while self.ctx.clock.time < int((self.ctx.observation_data[:,0].size)): 

            self.particles = self.integrator.propagate(self.particles,self.ctx)
            weights = self.resampler.compute_weights(self.ctx.observation_data[self.ctx.clock.time,:],self.particles)
            self.particles = self.resampler.resample(weights=weights,ctx=self.ctx,particleArray=self.particles)
            self.particles = self.perturb.randomly_perturb(ctx=self.ctx,particleArray=self.particles)


            '''output updates, not part of the main algorithm'''
            beta.append(np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)],axis=0))
            print(np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)],axis=0))
            self.ctx.clock.tick()
            print(f"iteration: {self.ctx.clock.time}")  

        plt.plot(beta)
        plt.show()

        return self.output    





        