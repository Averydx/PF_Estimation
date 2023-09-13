from abc import ABC,abstractmethod,abstractproperty
from numpy.typing import NDArray
from numpy import random,array,concatenate
from typing import List,Dict
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import RunInfo,Particle,Context,Clock
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from copy import deepcopy

class Algorithm(ABC): 

    integrator: Integrator
    perturb: Perturb
    resampler: Resampler
    particles: List[Particle]
    context: Context
    output_flags: Dict
    output: Output

    def __init__(self,integrator:Integrator,perturb:Perturb,resampler:Resampler,context:Context)->None:
        self.integrator = integrator
        self.perturb = perturb
        self.resampler = resampler
        self.particles = []
        self.context = context
        self.output = Output(np.array([]))
        self.output_flags = {}


    '''Abstract Methods''' 
    @abstractmethod
    def initialize(self,params:Dict)->None: #method to initialize all fields of the 

        for _,(key,val) in enumerate(params.items()): 
            if val == -1: 
                self.context.estimated_params.append(key)

        for _ in range(self.context.particle_count): 
            initial_infected = self.context.rng.uniform(0,self.context.seed_size*self.context.population)
            state = concatenate((array([self.context.population-initial_infected,initial_infected]),[0 for _ in range(self.context.state_size-2)])) #SIRH model 
            self.particles.append(Particle(param=params.copy(),state=state.copy(),observation=array([0])))

        

    @abstractmethod
    def run(self,info:RunInfo) ->Output:
        pass
        

    '''Callables'''

    def print_particles(self): 
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

    def clean_up(self): 
        if self.output_flags['write'] is True: 
            pd.DataFrame(self.output.average_beta).to_csv('./output/average_beta.csv')



    



    







    

