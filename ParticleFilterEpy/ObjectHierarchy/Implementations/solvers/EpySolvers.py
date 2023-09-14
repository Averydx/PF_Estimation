
from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.utilities.particle_simulation import ParticleSimulation
from ObjectHierarchy.Abstract.Integrator import Integrator
from epymorph.data import geo_library,ipm_library,mm_library
from epymorph.context import Compartments, SimDType
from typing import List
import numpy as np

class EpymorphSolver(Integrator): 
    def propagate(self, particleArray: List[Particle], ctx: Context) -> List[Particle]:
        for j,particle in enumerate(particleArray): 
            param = particle.param
            compartments = particle.state

            sim = ParticleSimulation(
                geo=ctx.geo,
                ipm_builder=ctx.ipm_builder,
                mvm_builder=ctx.mvm_builder,
                tick_index = 0,
                param=param, 
                compartments=compartments
            )

            incidence = sim.step() + sim.step()
            
            particleArray[j].observation = incidence[:,0]
            particleArray[j].state = np.array(sim.get_compartments())

        return particleArray

