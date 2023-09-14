
from ObjectHierarchy.Implementations.solvers.EpySolvers import EpymorphSolver
from ObjectHierarchy.Implementations.resamplers.resamplers import MultivariateNormalResample,PoissonResample
from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_PF
from ObjectHierarchy.Implementations.perturbers.perturbers import ParamOnlyMultivariate
from ObjectHierarchy.utilities.Utils import Context,get_observations
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from epymorph.data import geo_library,ipm_library,mm_library
import numpy as np

np.set_printoptions(suppress=True)
solver = EpymorphSolver()
perturb = ParamOnlyMultivariate(params={"cov":0.01})
resample = MultivariateNormalResample(cov=10_000_000)

data = get_observations(filePath="./data_sets/epy_inc.csv")
data = np.delete(data,0,1)
data = data.T
algo = Epymorph_PF(integrator=solver,
                         perturb=perturb,
                         resampler=resample,
                         ctx=Context(observation_data=data,
                                     particle_count=1000,
                                     seed_size=0.01,
                                     geo=geo_library['pei'](),
                                     ipm_builder=ipm_library['sirs'](),
                                     mvm_builder=mm_library['pei'](),
                                     rng = np.random.default_rng(1)))


algo.initialize({"beta":-1,"gamma":0.25,"xi":1/90,"theta":0.1,"move_control": 0.9})
out = algo.run()
# plot(out,1)

