from typing import Dict
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.context import Compartments, SimDType
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Context,Particle,quantiles,timing,jacob
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
import matplotlib.pyplot as plt
from matplotlib.cm import plasma
import multiprocessing as mp
import numpy as np



class Epymorph_PF(Algorithm): 
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler, ctx: Context) -> None:
        super().__init__(integrator, perturb, resampler, ctx)

    def initialize(self, params: Dict) -> None:
        '''Note, it is required to invoke the super initializer to instantiate the particles'''
        super().initialize(params=params)

    @timing
    def run(self)->Output:
        '''field initializations for Output'''
        self.output = Output(observation_data=self.ctx.observation_data)

        beta=[]
        observations = []
        LL=0
        '''main run loop'''
        while self.ctx.clock.time < int(self.ctx.observation_data[:,0].size): 

            self.particles = self.integrator.propagate(self.particles,self.ctx)
            '''With this new aggregation approach the observations must be explicitly zeroed, see Callables in Abstract.Algorithm'''
            if(self.ctx.clock.time % self.ctx.estimation_scale == 0): 
                LL_t = self.resampler.compute_weights(self.ctx.observation_data[self.ctx.clock.time,:],self.particles)
                LL+= np.log(LL_t)
                self.particles = self.resampler.resample(ctx=self.ctx,particleArray=self.particles)

                self.particles = self.perturb.randomly_perturb(ctx=self.ctx,particleArray=self.particles)
                
                self.zero_observations()
                print(self.ctx.clock.time)
            
            '''output updates, not part of the main algorithm'''
            beta.append(np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)],axis=0))
            
            print(beta[-1])
            
            self.ctx.clock.tick()
            #print(f"iteration: {self.ctx.clock.time}")  

        plt.title("Beta Over Time")
        plt.xlabel("Time(days)")
        plt.ylabel("Beta")
        #plt.plot(beta)
        plt.plot(beta)
        print(f"Log Likelihood: {LL}");
        plt.show()



       
        return self.output    




        