from ObjectHierarchy.Implementations.TimeDependentBeta import *
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import RunInfo
from scipy.stats import poisson
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

def likelihood(observation,particle_observations:NDArray[int_])->NDArray: 
    return poisson.pmf(observation,particle_observations)
    #return np.array([1 for _ in range(len(particle_observations))])


real_beta = pd.read_csv('./data_sets/beta_test.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
euler = Euler()
perturb = MultivariatePerturbations(params={"sigma1":0.01,"sigma2":0.1})
resample = PoissonResample(likelihood=likelihood)

algo = TimeDependentAlgo(integrator=euler,perturb=perturb,resampler=resample)
algo.initialize()


algo.run(RunInfo(np.array(real_beta),0))

#TODO DEBUG 
#SOLVED GLITCH IN PERTURBATIONS 








