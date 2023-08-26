from ObjectHierarchy.Implementations.TimeDependentBeta import *
from ObjectHierarchy.Output import Output
from ObjectHierarchy.Utils import RunInfo
from scipy.stats import poisson
from numpy.typing import NDArray

def likelihood(observation,particle_observations:NDArray[int_])->NDArray: 
    return poisson.pmf(observation,particle_observations)


euler = Euler()
perturb = MultivariatePerturbations(params={"sigma1":0.1,"sigma2":0.002})
resample = PoissonResample(likelihood=likelihood)

algo = TimeDependentAlgo(integrator=euler,perturb=perturb,resampler=resample)
algo.initialize()
info = RunInfo(observation_data=np.array([0,2,3,4,6]),forecast_time=0)

Output = algo.run(info=info)
#algo.print_particles()


