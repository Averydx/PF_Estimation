
from ObjectHierarchy.utilities.Utils import Particle,Context,timing
from ObjectHierarchy.utilities.particle_simulation import ParticleSimulation
from ObjectHierarchy.Abstract.Integrator import Integrator
from epymorph.data import geo_library,ipm_library,mm_library
from epymorph.context import Compartments, SimDType
from epymorph.geo import Geo
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from typing import List,Tuple
import numpy as np
import multiprocessing as mp

class EpymorphSolver(Integrator): 
    @timing
    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:

        args = [(ctx.geo,ctx.ipm_builder,ctx.mvm_builder,particle) for particle in particleArray]
        particleArray = ctx.process_pool.starmap(self.sub,args)

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


    def sub(self,geo:Geo,ipm_builder:IpmBuilder,mvm_builder:MovementBuilder,particle:Particle):
        sim = ParticleSimulation(
                geo=geo,
                ipm_builder=ipm_builder,
                mvm_builder=mvm_builder,
                tick_index = 0,
                param=particle.param, 
                compartments=particle.state
            )
        #print(mp.current_process())
        incidence = sim.step() + sim.step()
        particle.state = sim.get_compartments()
        particle.observation = incidence[:,0]
        return particle

