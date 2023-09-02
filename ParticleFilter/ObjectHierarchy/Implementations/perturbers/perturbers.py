from ObjectHierarchy.Abstract.Perturb import Perturb
from typing import List,Dict
import numpy as np
from utilities.utility import multivariate_normal
from ObjectHierarchy.utilities.Utils import Context,Particle

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self,ctx:Context,particleArray:List[Particle]):
        C = np.diag([(self.hyperparameters['sigma1']/ctx.population) ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] ** 2,
                     self.hyperparameters['sigma1'] **2,
                     self.hyperparameters['sigma2'] ** 2]).astype(float)
        
        
        A = np.linalg.cholesky(C)
        for i,_ in enumerate(particleArray): 

            #variation of the state and parameters

            perturbed = np.log(np.concatenate((particleArray[i].state,[particleArray[i].param['beta']])))

            perturbed = np.exp(multivariate_normal(perturbed,A))
            perturbed[0:ctx.state_size] /= np.sum(perturbed[0:ctx.state_size])
            perturbed[0:ctx.state_size] *= ctx.population


            particleArray[i].state = perturbed[0:ctx.state_size]
            particleArray[i].param['beta'] = perturbed[-1]
            



        return particleArray
    
'''IF2 perturbations are only applied to the parameters of interest, not the state itself'''
class DiscretePerturbations(Perturb):

    def __init__(self, params: Dict) -> None:
        super().__init__(params)

    def randomly_perturb(self, ctx: Context, particleArray: List[Particle]) -> List[Particle]:
        
        #var = variance(np.array([particle.param['beta'] for particle in particleArray]))
        for j,particle in enumerate(particleArray): 
            particleArray[j].param['beta'] = ctx.rng.normal(particle.param['beta'],
            self.hyperparameters['sigma2'] * self.hyperparameters['a'] ** ((2*ctx.additional_hyperparameters['m'])/50))

        return particleArray