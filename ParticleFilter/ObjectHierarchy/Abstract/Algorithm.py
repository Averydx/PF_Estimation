from abc import ABC,abstractmethod,abstractproperty
from numpy.typing import NDArray
from typing import List
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import RunInfo,Particle,Context,Clock

class Algorithm(ABC): 

    _integrator: Integrator
    _perturb: Perturb
    _resampler: Resampler
    _particles: List[Particle]
    _context: Context

    def __init__(self,integrator:Integrator,perturb:Perturb,resampler:Resampler)->None:
        self._integrator = integrator
        self._perturb = perturb
        self._resampler = resampler
        self._particles = []
        self._context = Context(particle_count=0,clock=Clock())

    @abstractmethod
    def run(self,info:RunInfo) ->Output:
        pass



    







    

