from ObjectHierarchy.Implementations.algorithms.TimeDependentBeta import *
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo
from ObjectHierarchy.Implementations.solvers.DeterministicSolvers import EulerSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import MultivariatePerturbations
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample,NormResample,NBResample
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

real_beta = pd.read_csv('./data_sets/FLU_HOSPITALIZATIONS.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
# euler = Euler()
solver = EulerSolver()
perturb = MultivariatePerturbations(params={"sigma1":0.1,"sigma2":0.1})
resample = NBResample()

algo = TimeDependentAlgo(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7_000_000,state_size=4,particle_count=10000))
algo.initialize({"beta":-1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0}) #Initialize estimated parameters to -1 so the parent knows which params not to touch)

out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
plot(out,1)











