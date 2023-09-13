from abc import ABC,abstractmethod
from typing import Dict,List
from ObjectHierarchy.Utils import Particle,Context

class Perturb(ABC): 
    hyperparameters: Dict

    def __init__ (self,hyper_params:Dict)-> None: 
        self.hyperparameters = hyper_params


    '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''
    @abstractmethod
    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        pass


