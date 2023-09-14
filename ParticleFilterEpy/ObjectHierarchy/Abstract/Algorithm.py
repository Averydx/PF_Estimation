from abc import ABC,abstractmethod
from numpy.typing import NDArray
from numpy import random,array,concatenate
from typing import List,Dict
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Particle,Context,Clock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Algorithm(ABC): 

    integrator: Integrator
    perturb: Perturb
    resampler: Resampler
    particles: List[Particle]
    ctx: Context
    output: Output

    def __init__(self,integrator:Integrator,perturb:Perturb,resampler:Resampler,ctx:Context)->None:
        self.integrator = integrator
        self.perturb = perturb
        self.resampler = resampler
        self.particles = []
        self.ctx = ctx
        self.output = Output(np.array([]))


    '''Abstract Methods''' 
    @abstractmethod
    def initialize(self,params:Dict)->None:

        for _,(key,val) in enumerate(params.items()): 
            if val == -1: 
                self.ctx.estimated_params.append(key)

    @abstractmethod
    def run(self) ->Output:
        pass
        
    '''Callables'''

    '''Prints the particle swarm in a human readable format'''
    def print_particles(self): 
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

    '''Verifies the field all compatible with the underlying model'''
    def verify_fields(self)->None:
        '''Perturber and resampler verification'''
        if(self.ctx.geo.nodes > 1 and self.resampler.Flags['all_size_valid'] is False):
            raise Exception("Resampler is incompatible with geos of dim > 1")
        
        if(self.ctx.geo.nodes > 1 and self.perturb.Flags['all_size_valid'] is False):
            raise Exception("Perturber is incompatible with geos of dim > 1")




    



    







    

