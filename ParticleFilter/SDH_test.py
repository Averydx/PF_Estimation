from ObjectHierarchy.Implementations.algorithms.TB_SDH import TB_SDH
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo,Context
from ObjectHierarchy.Implementations.solvers.StochasticSolvers import PoissonSolver
from ObjectHierarchy.Implementations.solvers.DeterministicSolvers import EulerSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import ParamOnlyMultivariate
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample,NormResample
import numpy as np
import pandas as pd

real_beta = pd.read_csv('./data_sets/beta_test.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
solver = EulerSolver()
perturb = ParamOnlyMultivariate({"cov":np.diag([0.01,0.01,0.01]),"a":0.5})
resample = NormResample(var = 10)

algo = TB_SDH(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7000000,state_size=4,particle_count=10000))
algo.initialize({"beta":0,"a0":-1,"a1":-1,"a2":-1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0,"x1":0.1,"x2":0.7})


out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
#plot(out,0)


