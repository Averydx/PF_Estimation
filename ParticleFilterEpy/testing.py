
from ObjectHierarchy.Implementations.solvers.EpySolvers import EpymorphSolver
from ObjectHierarchy.Implementations.resamplers.resamplers import MultivariateNormalResample
from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_PF
from ObjectHierarchy.utilities.Utils import Context,get_observations
from epymorph.data import geo_library,ipm_library,mm_library
import numpy as np

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = None
resample = MultivariateNormalResample()

data = get_observations(filePath="C:/Users/avery/PF_Epymorph/PF_Estimation/data_sets/epy_inc.csv")
data = np.delete(data,0,1)
data = data.T
algo = Epymorph_PF(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         ctx=Context(observation_data=data,
                                     particle_count=5,
                                     seed_size=0.01,
                                     geo=geo_library['pei'](),
                                     ipm_builder=ipm_library['sirs'](),
                                     mvm_builder=mm_library['pei']()))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})

for i in range(150):
    algo.particles=algo.integrator.propagate(particleArray=algo.particles,ctx=algo.ctx)
algo.print_particles()
#out = algo.run()
# plot(out,1)

