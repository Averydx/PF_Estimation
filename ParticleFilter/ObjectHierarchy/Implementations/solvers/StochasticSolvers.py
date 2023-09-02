'''Stochastic analog to the euler solver for Alex and Kayodes SIRH model'''
from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Integrator import Integrator
from typing import List
import numpy as np




class PoissonSolver(Integrator): 
    def propagate(self, particleArray: List[Particle],ctx:Context) -> List[Particle]:
        for j,particle in enumerate(particleArray): 
            S,I,R,H = particle.state
            N = S+I+R+H
            new_susceptibles = ctx.rng.poisson((1/particle.param['L'])*R)
            new_infected = ctx.rng.poisson((particle.param['beta'] * S * I)/N)
            new_recovered_from_H = ctx.rng.poisson((1/particle.param['hosp']) * H)
            new_recovered_from_I = ctx.rng.poisson(((1/particle.param['D']) * (1-particle.param['gamma']))*I) 
            new_hospitalized = ctx.rng.poisson((1/particle.param['D']) * particle.param['gamma'] * I)
            

            state = np.zeros_like(particle.state)
            state[0] = max(0.,S-new_infected + new_susceptibles)
            state[1] = max(0.,I + new_infected - (new_hospitalized + new_recovered_from_I))
            state[2] = max(0.,R + new_recovered_from_H + new_recovered_from_I - new_susceptibles)
            state[3] = max(0.,H + new_hospitalized - new_recovered_from_H)

            particleArray[j].state = state
            particleArray[j].observation = new_hospitalized

        return particleArray
