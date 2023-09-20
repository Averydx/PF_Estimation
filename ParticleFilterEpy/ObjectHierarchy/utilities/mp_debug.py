from particle_simulation import ParticleSimulation
import multiprocessing as mp
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from epymorph.data import geo_library,ipm_library,mm_library
import concurrent.futures as cf
from Utils import Particle
import numpy as np
from time import perf_counter


def propagate(particle):
    sim = ParticleSimulation(
                geo=geo_library['pei'](),
                ipm_builder=ipm_library['sirs'](),
                mvm_builder=mm_library['pei'](),
                tick_index = 0,
                param={"beta":0.4,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control":0.9}, 
                compartments=np.array(particle.state)
            )
    

    incidence = sim.step() + sim.step()
    particle.state = sim.get_compartments()
    particle.observation = incidence
    return particle


if __name__ == '__main__':
    particles = []
    for i in range(10000):
        particles.append(Particle(param={},state=np.array([[10000,100000,100001],
                                                          [10000,100000,100001],
                                                          [10000,100000,100001],
                                                          [10000,100000,100001],
                                                          [10000,100000,100001],
                                                          [10000,100000,100001]]),observation=np.array([0 for _ in range(geo_library['pei']().nodes)])))
    t0 = perf_counter()
    print(mp.cpu_count())
    with mp.Pool(mp.cpu_count()) as executor:
        particles =executor.map(propagate, particles)

    t1 = perf_counter()
    print(f"mp time:{t1-t0}")

    t2 = perf_counter()
    for particle in particles:
        propagate(particle=particle)
    t3 = perf_counter()

    print(f"single thread:{t3-t2}")
