from abc import ABC,abstractmethod
from typing import Dict,List
from ObjectHierarchy.Utils import Particle

class Perturb(ABC): 
    hyperparameters: Dict

    def __init__ (self,params:Dict)-> None: 
        self.hyperparameters = params

    @abstractmethod
    def randomly_perturb(self,particleArray: List[Particle])->List[Particle]: 
        pass


