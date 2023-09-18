from abc import ABC,abstractmethod
from typing import Dict,List
from ObjectHierarchy.utilities.Utils import Particle,Context

class Perturb(ABC): 
    '''Perturb is the abstract class which is extended to define a perturbation scheme, it's quite barebones, mandates only randomly_perturb() and two fields'''


    '''hyperparameters is the Dict of perturbation parameters, i.e. variance/covariance or cooling rate'''
    hyperparameters: Dict
    '''Flags define whether a perturbation scheme is compatible with different types of geos and algorithms'''
    Flags:Dict[str,int]
    def __init__ (self,hyper_params:Dict)-> None: 
        self.hyperparameters = hyper_params


    '''Implementations of this method will take a list of particles and perturb it according to a user defined distribution'''
    @abstractmethod
    def randomly_perturb(self,ctx:Context,particleArray: List[Particle])->List[Particle]: 
        pass


