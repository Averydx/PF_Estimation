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

    def print_particles(self): 
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

    def clean_up(self): 
        if self.output_flags['write'] is True: 
            pd.DataFrame(self.output.average_beta).to_csv('./output/average_beta.csv')



    



    







    

