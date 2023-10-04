from dataclasses import dataclass,field
import pandas as pd 
import numpy as np
from numpy.typing import NDArray
from numpy import random
from typing import Dict,List,Tuple
from functools import wraps
from time import perf_counter
from epymorph.geo import Geo
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
import multiprocessing as mp
import os.path

'''Wrapper for dataset parsing using pandas, wraps to_csv with protections'''
def get_observations(filePath: str)->NDArray: 
    if not os.path.isfile(filePath):
        raise Exception("The file path specified was not found")
    df = pd.read_csv(filePath)
    
    return df.to_numpy()

'''pass in the cholesky decomposition of the cov matrix, 
important because the cholesky decomp is only calculated once --NOT CALCULATED INSIDE FUNCTION--
Assumes the covariance matrix will be positive definite'''
def multivariate_normal(mean,A):

    x = np.random.standard_normal(mean.shape)
    x = np.dot(x, A)
    x += mean
    x.shape = tuple(mean.shape)
    return x

'''Internal clock for keeping track of the time the algorithm is at in the observation data'''
class Clock: 
    time: int
    def __init__(self) -> None:
        self.time = 0
    
    def tick(self):
        self.time +=1

    def reset(self): 
        self.time = 0

'''The basic particle class-No fields make assumptions about their size, passing in arbitrary sized geos should be possible'''
@dataclass
class Particle: 
    param: Dict
    state: NDArray
    observation: NDArray

'''Metadata about the algorithm'''
@dataclass(frozen=True)
class Context: 

    observation_data: NDArray # TxN NDArray of observation data  
    geo:Geo #information about the population of interest
    ipm_builder: IpmBuilder #Class that builds the ipm 
    mvm_builder: MovementBuilder #Class that builds the movement model
    particle_count: int = 1000
    clock: Clock = field(default_factory=lambda: Clock())
    rng:random.Generator = field(default_factory=lambda: np.random.default_rng()) #uses the default numpy rng   
    seed_size: float = 0.01 #number of initial infected
    estimated_params: List[str] = field(default_factory=lambda: []) #number of estimated parameters in the model 
    estimation_scale: int = 1 #The scale at which to perturb beta, 1 is daily, 7 is weekly, etc.
    process_pool: mp.Pool = field(default_factory=lambda: mp.Pool(mp.cpu_count())) #A process pool that instantiates a number of python processes equal to the core count
    

    
'''Decorator for timing function calls '''
def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = perf_counter()
        result = f(*args, **kw)
        te = perf_counter()
        print('func:%r took: %2.4f sec' % \
          (f.__name__, te-ts))
        return result
    return wrap

'''Returns 23 quantiles of the List passed in'''
def quantiles(items:List)->List: 
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
        return list(np.quantile(items, qtlMark))

'''Returns the sample variance of an array of items'''
def variance(items:NDArray[np.float_])->float: 
    X_bar = np.mean(items)
    return float(np.sum((items-X_bar)**2)/(len(items) - 1))

'''The jacobian logarithm, used in log likelihood normalization and resampling processes
δ will be an array of log-probabilities '''
def jacob(δ:NDArray)->NDArray:
    n = len(δ)
    Δ = np.zeros(n)
    Δ[0] = δ[0]
    for i in range(1,n):
        Δ[i] = max(δ[i],Δ[i-1]) + np.log(1 + np.exp(-1*np.abs(δ[i] - Δ[i-1])))
    return(Δ)

'''normalizes the probability space using the jacobian logarithm as defined in jacob() '''
def log_norm(log_weights:NDArray): 
    norm = (jacob(log_weights)[-1])
    log_weights -= norm
    return log_weights

'''computes the log mean of the parameter of interest'''
def log_mean(particleArray:List[Particle],estimated_param:str):
    return np.sum([np.log(particle.param[estimated_param]) for _,particle in enumerate(particleArray)],axis=0)

'''computes the log covariance matrix of the parameter of interest'''
def log_cov(particleArray:List[Particle],estimated_param:str): 
    eta = log_mean(particleArray,estimated_param)
    vec = np.array([np.log(particle.param[estimated_param]) - eta for _,particle in enumerate(particleArray)])
    sigma = 0
    for i,_ in enumerate(particleArray): 
        sigma += np.outer(vec[i],vec[i].T)

    return sigma







