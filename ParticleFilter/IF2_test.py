from ObjectHierarchy.Implementations.algorithms.IF2 import IF2
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo,Context
from ObjectHierarchy.Implementations.solvers.StochasticSolvers import PoissonSolver
from ObjectHierarchy.Implementations.solvers.DeterministicSolvers import EulerSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import DiscretePerturbations
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample,NormResample
import numpy as np
import pandas as pd

real_beta = pd.read_csv('./data_sets/FLU_HOSPITALIZATIONS.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
solver = EulerSolver()
perturb = DiscretePerturbations({"cov":0.02,"a":0.5})
resample = PoissonResample()

algo = IF2(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7_000_000,state_size=4,additional_hyperparameters={"m":200},particle_count=1000))
algo.initialize({"beta":-1,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0})


out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
plot(out,0)



