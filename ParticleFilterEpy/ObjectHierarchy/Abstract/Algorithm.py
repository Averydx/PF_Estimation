from abc import ABC,abstractmethod
from numpy.typing import NDArray
from numpy import random,array,concatenate
from typing import List,Dict
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Particle,Context,Clock
from epymorph.util import check_ndarray,NumpyTypeError
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Algorithm(ABC): 

    '''The basic fields of an algorithm implementation
    Integrator,perturb, and resampler are extensions of their abstract base classes as defined in the Abstract directory
    
    Context and Particle are defined in utilities/Utils.py
    '''
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

        '''Builds the list of estimated parameters'''
        for _,(key,val) in enumerate(params.items()): 
            if np.all([item == -1 for item in val]): 
                self.ctx.estimated_params.append(key)

        '''Type and shape checking to prevent against invalid data'''
        try:
            check_ndarray(value=self.ctx.observation_data[0,:],dtype=[np.int64,np.int32,np.int16],shape=(self.ctx.geo.nodes,))
        except NumpyTypeError:
            raise Exception("Observation data did not match the specified shape, check data dimensionality, must be (TxN)")
        
        '''Verify the resampler and perturber are functional for the epymorph model at the requested scale'''
        self.verify_fields()

        '''Initializes the default values of the estimated params, based on the length passed in at object instantiation'''
        for _ in range(self.ctx.particle_count): 
            for param in self.ctx.estimated_params:
                params[param] = [self.ctx.rng.uniform(0.,1.) for _ in range(len(params[param]))]
                if (len(params[param]) != len(self.ctx.observation_data[0,:])):
                    if(len(params[param])== 1):
                        break
                    raise Exception(f"estimated parameter:{param} shape and data shape mismatch!")

            '''Draw a random int to represent the initial infection node and the initial_infected amount'''
            infection_location = self.ctx.rng.integers(0,self.ctx.geo.nodes)
            pops = self.ctx.geo.data['population'] 
            initial_infected = self.ctx.rng.integers(0,np.round(self.ctx.seed_size*pops[infection_location]))
            state = []

            '''Create initial compartment values for both the population with the initial infection and the ones without dynamically'''
            for index,pop in enumerate(pops): 
                
                if(index == infection_location):
                    substate = [pop-initial_infected,initial_infected]
                    '''Appends a list of zeros after instantiating the desired compartments of the state vector
                      to represent the initial seed'''
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                else: 
                    substate = [pop,0]
                    for _ in range(self.ctx.ipm_builder.compartments-2):
                        substate.append(0)

                state.append(substate)

            state = np.array(state)      
            self.particles.append(Particle(param=params.copy(),state=state.copy(),observation=np.array([])))
            self.zero_observations(); 

            

    


        

    '''Entirely implementation defined-requires only an Output return at the end, 
    general template is defined elsewhere in Documentation'''
    @abstractmethod
    def run(self) ->Output:
        pass
        

    '''Callables'''
    

    '''Prints the particle swarm in a human readable format'''
    def print_particles(self)->None: 
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

    '''Verifies the field all compatible with the underlying model'''
    def verify_fields(self)->None:
        '''Perturber and resampler verification'''
        if(self.ctx.geo.nodes > 1 and self.resampler.Flags['all_size_valid'] is False):
            raise Exception("Resampler is incompatible with geos of dim > 1")
        
        if(self.ctx.geo.nodes > 1 and self.perturb.Flags['all_size_valid'] is False):
            raise Exception("Perturber is incompatible with geos of dim > 1")
        
    def zero_observations(self)->None: 
        for particle in self.particles: 
            particle.observation = np.array([0 for _ in range(self.ctx.geo.nodes)])




    



    







    

