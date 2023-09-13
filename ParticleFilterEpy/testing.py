
from ObjectHierarchy.Implementations.solvers.EpySolvers import EpymorphSolver
from ObjectHierarchy.Implementations.resamplers.resamplers import MultivariateNormalResample
from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_PF
from ObjectHierarchy.utilities.Utils import Context
from epymorph.data import geo_library,ipm_library,mm_library
import numpy as np

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = None
resample = MultivariateNormalResample()

algo = Epymorph_PF(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         ctx=Context(particle_count=5,seed_size=0.01,geo=geo_library['single_pop'](),ipm_builder=ipm_library['sirh'](),mvm_builder=mm_library['pei']()))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})
algo.print_particles()



# out = algo.run(RunInfo(np.array(real_beta),0,output_flags={'write': True}))
# plot(out,1)

