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

    '''The abstract algorithm class outlines the fields and methods all implementations of the class must define.'''
   
    '''Integrator is the one step transition model, see Integrator.py '''
    integrator: Integrator

    '''Perturb is the perturbation model, see Perturb.py'''
    perturb: Perturb

    '''Resampler is the resampling method and likelihood computations, see Resampler.py'''
    resampler: Resampler

    '''The particle swarm, the basic Particle dataclass is defined in Utils.py'''
    particles: List[Particle]

    '''Metadata about algorithm hyperparameters and fields, necessary for run()'''
    ctx: Context

    '''Output aggregation, only returned from run() see below for more details'''
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

        
        '''Calling this function will setup parameters in the Particle Filtering, 
        in user implementations of the initialization method you can call this to perform setup'''

        '''Type and shape checking to prevent against invalid data'''
        try:
            check_ndarray(value=self.ctx.observation_data[0,:],dtype=[np.int64,np.int32,np.int16],shape=(self.ctx.geo.nodes,))
        except NumpyTypeError:
            raise Exception("Observation data did not match the specified shape, check data dimensionality, must be (TxN)")
    
        '''Verify the resampler and perturber are functional for the epymorph model at the requested scale'''
        self.verify_fields()


        for _,(key,val) in enumerate(params.items()): 
            if np.any(val[val <0]): 
                self.ctx.estimated_params.append(key)







    @abstractmethod
    def run(self) ->Output:
        '''No base implementation of the run method, this is the main loop for the estimation problem, the return type will encapsulate all the data the algorithm accumulated 
        during runtime, see utilities/Output.py for more details on the Output dataclass'''
        pass
        

    
    '''Callables'''

    
    def print_particles(self): 
        '''Prints the particle swarm in a human readable format'''
        for i,particle in enumerate(self.particles): 
            print(f"{i}: {particle}")

    
    def verify_fields(self)->None:
        '''Verifies the field all compatible with the underlying model'''

        '''Perturber and resampler verification'''
        if(self.ctx.geo.nodes > 1 and self.resampler.Flags['all_size_valid'] is False):
            raise Exception("Resampler is incompatible with geos of dim > 1")






    



    







    

