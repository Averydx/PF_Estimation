
from ObjectHierarchy.Implementations.solvers.EpySolvers import EpymorphSolver
from ObjectHierarchy.Implementations.resamplers.resamplers import MultivariateNormalResample,PoissonResample,LogMultivariatePoissonResample
from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_PF
from ObjectHierarchy.Implementations.perturbers.perturbers import ParamOnlyMultivariate
from ObjectHierarchy.utilities.Utils import Context,get_observations,jacob,log_norm
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from epymorph.data import geo_library,ipm_library,mm_library
from ObjectHierarchy.geo.quad_pop import load
import numpy as np



'''Using a __name__ main is required for the multiprocessing module to integrate nicely, 
if wrapping basic functions elsewhere this must be included if using multiprocessing mapping functions
Note: using Multiprocessing is not required, although my basic implementations do use it for speed reasons'''
if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    solver = EpymorphSolver()
    perturb = ParamOnlyMultivariate(params={"cov":0.01})
    resample = LogMultivariatePoissonResample()

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
                                        rng = np.random.default_rng()))


    algo.initialize({"beta":np.array([-1,-1,-1,-1,-1,-1]),"gamma":np.array([0.25]),"xi":np.array([1/90]),"theta":np.array([0.1]),"move_control":np.array([0.9])})
    out = algo.run()



#     # plot(out,1)

