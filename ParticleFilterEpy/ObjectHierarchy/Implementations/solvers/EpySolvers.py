
from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.utilities.particle_simulation import ParticleSimulation
from ObjectHierarchy.Abstract.Integrator import Integrator
from epymorph.data import geo_library,ipm_library,mm_library
from epymorph.context import Compartments, SimDType
from typing import List,Tuple
import numpy as np
import multiprocessing as mp

class EpymorphSolver(Integrator): 

    def propagate(self, particleArray: List[Particle], ctx: Context,pool) -> List[Particle]:

        args = [(ctx,particle) for particle in particleArray]
        particleArray = pool.starmap(self.sub,args)

        # for j,particle in enumerate(particleArray): 
        #     param = particle.param
        #     compartments = particle.state

        #     sim = ParticleSimulation(
        #         geo=ctx.geo,
        #         ipm_builder=ctx.ipm_builder,
        #         mvm_builder=ctx.mvm_builder,
        #         tick_index = 0,
        #         param=param, 
        #         compartments=compartments
        #     )

        #     incidence = sim.step() + sim.step()
            
        #     particleArray[j].observation = incidence[:,0]
        #     particleArray[j].state = np.array(sim.get_compartments())

        return particleArray


    def sub(self,ctx:Context,particle:Particle):
        sim = ParticleSimulation(
                geo=ctx.geo,
                ipm_builder=ctx.ipm_builder,
                mvm_builder=ctx.mvm_builder,
                tick_index = 0,
                param=particle.param, 
                compartments=particle.state
            )
    
        incidence = sim.step() + sim.step()
        particle.state = sim.get_compartments()
        particle.observation = incidence[:,0]
        return particle

