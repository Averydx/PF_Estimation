from ObjectHierarchy.Implementations.algorithms.epymorph_IF2 import Epymorph_IF2
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.plotting import plot
from ObjectHierarchy.utilities.Utils import RunInfo,Context
from ObjectHierarchy.Implementations.solvers.StochasticSolvers import PoissonSolver,EpymorphSolver
from ObjectHierarchy.Implementations.solvers.DeterministicSolvers import EulerSolver
from ObjectHierarchy.Implementations.perturbers.perturbers import DiscretePerturbations
from ObjectHierarchy.Implementations.resamplers.resamplers import PoissonResample,NormResample,JointPoissonResample
from scipy.stats import poisson,norm
import numpy as np
import pandas as pd

real_beta = pd.read_csv('./data_sets/epymorph_incidence_sum.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = DiscretePerturbations({"cov":0.02,"a":0.5})
resample = JointPoissonResample()

algo = Epymorph_IF2(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7000000,state_size=4,additional_hyperparameters={"m":1},particle_count=1000))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})
algo.particles = algo.integrator.propagate(ctx=algo.context,particleArray=algo.particles)
# for particle in algo.particles: 
#     print(particle.observation)

weights = algo.resampler.compute_weights(real_beta[:,0],particleArray=algo.particles)
algo.resampler.resample(ctx=algo.context,particleArray=algo.particles,weights=weights)
for particle in algo.particles: 
    print(particle.observation)



