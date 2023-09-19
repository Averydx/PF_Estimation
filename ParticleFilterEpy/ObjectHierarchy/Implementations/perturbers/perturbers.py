from ObjectHierarchy.Abstract.Perturb import Perturb
from typing import List,Dict
import numpy as np
#from utilities.utility import multivariate_normal
from ObjectHierarchy.utilities.Utils import Context,Particle,multivariate_normal
from epymorph.util import check_ndarray


'''Multivariate geometric perturbations to the parameters only, not the state'''
class ParamOnlyMultivariate(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)
        self.Flags = {"all_size_valid":True}
        if(not 'cov' in self.hyperparameters):
           raise Exception("covariance matrix is not defined -please define the covariance as an scalar in this object's constructor, it will be manually broadcast to a diagional array")

    def randomly_perturb(self,ctx:Context,particleArray:List[Particle]):
        cov = np.diag([self.hyperparameters['cov'] for _ in range(ctx.geo.nodes)])
        A = np.linalg.cholesky(cov)
        for i,_ in enumerate(particleArray): 
            for estimated_param in ctx.estimated_params:
                perturbed = np.log(particleArray[i].param[estimated_param])
                perturbed = np.exp(multivariate_normal(perturbed,A))
                particleArray[i].param[estimated_param] = perturbed

        

        return particleArray

'''Multivariate normal perturbations to all parameters and state variables after log transform'''
class MultivariatePerturbations(Perturb): 
    def __init__(self,params:Dict) -> None:
        super().__init__(params)
        self.Flags = {"all_size_valid":False}

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
        self.Flags = {"all_size_valid":False}

    def randomly_perturb(self, ctx: Context, particleArray: List[Particle]) -> List[Particle]:
        match np.shape(self.hyperparameters['cov']): 
            case (): 
                for j,particle in enumerate(particleArray): 
                    particleArray[j].param[ctx.estimated_params[0]] = ctx.rng.normal(particle.param[ctx.estimated_params[0]],
                    self.hyperparameters['cov'] * self.hyperparameters['a'] ** ((2*ctx.additional_hyperparameters['m'])/50))
            
            case _: 
                for j,particle in enumerate(particleArray): 
                    out = ctx.rng.multivariate_normal([particle.param[x] for x in ctx.estimated_params],
                                                      self.hyperparameters['cov'] * self.hyperparameters['a'] ** ((2*ctx.additional_hyperparameters['m'])/50))
                    
                    for i,name in enumerate(ctx.estimated_params): 
                        particleArray[j].param[name] = out[i]


        return particleArray