from ObjectHierarchy.Implementations.algorithms.epymorph_IF2 import Epymorph_IF2
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
import tracemalloc
import matplotlib.pyplot as plt

real_beta = pd.read_csv('C:/Users/avery/PF_Epymorph/PF_Estimation/data_sets/epy_inc.csv')
real_beta = np.squeeze(real_beta.to_numpy()) 
real_beta = np.delete(real_beta,0,1)

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = ParamOnlyMultivariate({"cov":np.diag([0.1]),"a":0.5})
resample = MultivariateNormalResample()

algo = Epymorph_IF2(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         context=Context(population=7000000,state_size=4,additional_hyperparameters={"m":1},particle_count=1000))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})

# out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
# plot(out,1)



observations = np.zeros((6,300))
beta = np.zeros(300)
tick_index = 0
for i in range(300):
     algo.particles = algo.integrator.propagate(ctx=algo.context,particleArray=algo.particles,tick_index=tick_index)
     weights = algo.resampler.compute_weights(real_beta[:,i],particleArray=algo.particles)
     algo.particles = algo.resampler.resample(ctx=algo.context,particleArray=algo.particles,weights=weights)
     algo.particles = algo.perturb.randomly_perturb(ctx=algo.context,particleArray=algo.particles)

     print(f"iteration: {i}")
     observations[:,i] = np.mean([particle.observation for particle in algo.particles],axis=0)
     beta[i] = (np.mean([particle.param['beta'] for particle in algo.particles],axis=0))
     # print(observations[:,i])
     # print(real_beta[:,i])
     #print(f"{weights}")

#df = pd.DataFrame(observations)
#df.to_csv('./data_sets/custom_epy_obvs')

t = np.arange(300)

for i in range(0,6):
     plt.plot(t,observations[i,:])
     plt.scatter(t,real_beta[i,:],s=0.1)

plt.show()

plt.plot(t,beta)
plt.show()



#algo.particles = algo.integrator.propagate(ctx=algo.context,particleArray=algo.particles)


# for particle in algo.particles: 
#     print(particle.state)



