from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_IF2
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo,Context
from ObjectHierarchy.Implementations.solvers.StochasticSolvers import PoissonSolver,EpymorphSolver
from ObjectHierarchy.Implementations.solvers.DeterministicSolvers import EulerSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import DiscretePerturbations,ParamOnlyMultivariate
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample,NormResample,MultivariateNormalResample
from scipy.stats import poisson,norm
from time import perf_counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

real_beta = pd.read_csv('./data_sets/epy_inc.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = ParamOnlyMultivariate({"cov":np.diag([0.01 for _ in range(6)]),"a":0.5})
resample = MultivariateNormalResample()

algo = Epymorph_IF2(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7000000,state_size=4,additional_hyperparameters={"m":1},particle_count=100,beta_length=6))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})


        


out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
# plot(out,1)




