from ObjectHierarchy.utilities.Utils import Particle,Context
from ObjectHierarchy.Abstract.Integrator import Integrator
from typing import List
import numpy as np


class EulerSolver(Integrator): 
    def __init__(self) -> None:
        super().__init__()

    '''Propagates the state forward one step and returns an array of states and observations across the the integration period'''
    def propagate(self,particleArray:List[Particle],ctx:Context)->List[Particle]: 


        for j,particle in enumerate(particleArray): 
            dt,sim_obv =self.RHS_H(particle)

            particleArray[j].state += dt
            particleArray[j].observation = np.array([sim_obv])

        return particleArray
    

    def RHS_H(self,particle:Particle):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = particle.state
        N = S + I + R + H 

        new_H = ((1/particle.param['D'])*particle.param['gamma']) * I   

        dS = -particle.param['beta']*(S*I)/N + (1/particle.param['L'])*R 
        dI = particle.param['beta']*S*I/N-(1/particle.param['D'])*I
        dR = (1/particle.param['hosp']) * H + ((1/particle.param['D'])*(1-(particle.param['gamma']))*I)-(1/particle.param['L'])*R 
        dH = (1/particle.param['D'])*(particle.param['gamma']) * I - (1/particle.param['hosp']) * H 

        return np.array([dS,dI,dR,dH]),new_H
    
