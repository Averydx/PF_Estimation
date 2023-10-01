
from ObjectHierarchy.Implementations.solvers.EpySolvers import EpymorphSolver
from ObjectHierarchy.Implementations.resamplers.resamplers import MultivariateNormalResample,PoissonResample,LogMultivariatePoissonResample,LogNormalResample
from ObjectHierarchy.Implementations.algorithms.epymorph_TDB import Epymorph_PF
from ObjectHierarchy.Implementations.perturbers.perturbers import ParamOnlyMultivariate
from ObjectHierarchy.utilities.Utils import Context,get_observations,jacob,log_norm
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from epymorph.data import geo_library,ipm_library,mm_library
from ObjectHierarchy.geo.single_pop import load
import numpy as np



'''Using a __name__ main is required for the multiprocessing module to integrate nicely, 
if wrapping basic functions elsewhere this must be included if using multiprocessing mapping functions
Note: using Multiprocessing is not required, although my basic implementations do use it for speed reasons'''
if __name__ == '__main__':

    np.set_printoptions(suppress=True)
    solver = EpymorphSolver()
    perturb = ParamOnlyMultivariate(params={"cov":0.01})
    resample = LogMultivariatePoissonResample()

    data = get_observations(filePath="./data_sets/FLU_HOSPITALIZATIONS.csv")
    data = np.delete(data,0,1)
    data = data
    algo = Epymorph_PF(integrator=solver,
                            perturb=perturb,
                            resampler=resample,
                            ctx=Context(observation_data=data,
                                        particle_count=1000,
                                        seed_size=0.01,
                                        geo=load(),
                                        ipm_builder=ipm_library['sirh'](),
                                        mvm_builder=mm_library['pei'](),
                                        estimation_scale = 1,
                                        rng = np.random.default_rng()))



    algo.initialize({"beta":np.array([-1]),"gamma":np.array([0.1]),"xi":np.array([0.1]),"theta":np.array([0.1]),"move_control":np.array([0.9]),'hospitalization_rate': np.array([0.001]),
        'hospitalization_duration':np.array([5.3])})
    out = algo.run()



#     # plot(out,1)

