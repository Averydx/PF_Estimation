from ObjectHierarchy.Implementations.algorithms.TimeDependentBeta import *
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo
from ObjectHierarchy.Implementations.solvers.StochasticSolvers import PoissonSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import MultivariatePerturbations
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample
from scipy.stats import poisson
from numpy.typing import NDArray
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 

real_beta = pd.read_csv('./data_sets/beta_test.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
# euler = Euler()
poisson_solver = PoissonSolver()
perturb = MultivariatePerturbations(params={"sigma1":0.01,"sigma2":0.1})
resample = PoissonResample()

algo = TimeDependentAlgo(integrator=poisson_solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=100000,state_size=4))
algo.initialize({"beta":-1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0}) #Initialize estimated parameters to -1 so the parent knows which params not to touch)


out = algo.run(RunInfo(np.array(real_beta),0))
plot(out,0)







