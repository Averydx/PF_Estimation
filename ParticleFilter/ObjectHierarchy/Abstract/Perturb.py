from abc import ABC,abstractmethod
from typing import Dict,List
from ObjectHierarchy.Utils import Particle

class Perturb(ABC): 
    hyperparameters: Dict

    def __init__ (self,hyper_params:Dict)-> None: 
        self.hyperparameters = hyper_params

    @abstractmethod
    def randomly_perturb(self,particleArray: List[Particle])->List[Particle]: 
        pass


