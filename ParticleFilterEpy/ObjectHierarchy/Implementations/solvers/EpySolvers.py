'''Stochastic analog to the euler solver for Alex and Kayodes SIRH model'''
from ObjectHierarchy.utilities.Utils import Particle,Context
import ObjectHierarchy.utilities.particle_simulation
from ObjectHierarchy.Abstract.Integrator import Integrator
from epymorph.data import geo_library,ipm_library,mm_library
from epymorph.context import Compartments, SimDType
from typing import List
import numpy as np

class EpymorphSolver(Integrator): 
    def propagate(self, particleArray: List[Particle], ctx: Context,tick_index) -> List[Particle]:
        geo=geo_library['pei']()
        ipm_builder = ipm_library['sirs']()
        mvm_builder = mm_library['pei']()
        for j,particle in enumerate(particleArray): 
            param = particle.param
            compartments = particle.state

            sim = ParticleSimulation(
                geo=geo,
                ipm_builder=ipm_builder,
                mvm_builder=mvm_builder,
                tick_index = tick_index,
                param=param, 
                compartments=compartments
            )

            incidence = sim.step() + sim.step()

            particleArray[j].observation = incidence[:,0]
            particleArray[j].state = np.array(sim.get_compartments())

        return particleArray

