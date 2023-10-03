from datetime import date
from functools import partial
import jsonpickle

import numpy as np

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.initializer import single_location
from epymorph.simulation import Simulation
from epymorph.run import plot_event,plot_pop
from epymorph.util import stridesum
import matplotlib.pyplot as plt
from ParticleFilterEpy.ObjectHierarchy.geo.dual_pop import load
import pandas as pd

# Note: the 'library' dictionaries contain functions which load the named component,
# so you have to apply the function to get the _actual_ component.

# The 'pei' model family (IPM/MM/GEO) implement an SIRS model in 6 US states.
# (Remember: it is possible to mix-and-match the models!)



def beta(t):
        
    betaMax1=0.1
    theta=0

    return 0.3+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
    #return 0.1

def beta_2(t):
        
    betaMax1=0.1
    theta=0

    return 0.5+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
    #return 0.1

def beta_3(t):
        
    betaMax1=0.1
    theta=0

    return 0.1+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
    #return 0.1

def beta_4(t):
        
    betaMax1=0.1
    theta=0

    return 0.05+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
    #return 0.1

def beta_5(t):
        
    betaMax1=0.1
    theta=0

    return 0.7+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  

num_days = 343
beta_cos_5 = np.array([beta_5(t) for t in range(343)])
beta_cos_4 = np.array([beta_4(t) for t in range(343)])
beta_cos_3 = np.array([beta_3(t) for t in range(343)])
beta_cos_2 = np.array([beta_2(t) for t in range(343)])
beta_cos = np.array([beta(t) for t in range(343)])
plt.plot(beta_cos)
plt.plot(beta_cos_2)
plt.plot(beta_cos_3)
plt.plot(beta_cos_4)
plt.plot(beta_cos_5)
# plt.show()

betas = np.zeros((343,2))

betas[:,0] = beta_cos
betas[:,1] = beta_cos_2
# betas[:,2] = beta_cos_3
# betas[:,3] = beta_cos_4
# betas[:,4] = beta_cos_3
# betas[:,5] = beta_cos_5


sim = Simulation(
    geo=load(),
    ipm_builder=ipm_library['sirs'](),
    mvm_builder=mm_library['pei']()
)

out = sim.run(
    param={
        'beta':betas,
        'gamma':0.25,
        'xi':1/90,
        'theta': 0.1,
        'move_control': 0.9,
        'infection_duration': 4.0,
        'immunity_duration': 90.0,
    },
    start_date=date(2015, 1, 1),
    duration_days=100,
    initializer=partial(single_location, location=0, seed_size=10_000),
    rng=np.random.default_rng(1)
)

print(out.prevalence[:,:])

incidence = []
for pop_idx in range(out.ctx.nodes):
    values = stridesum( out.incidence[:,pop_idx,0],len(out.ctx.clock.taus))
    incidence.append(values)
incidence = np.array(incidence)

df = pd.DataFrame(incidence)
df.to_csv('./data_sets/epy_inc.csv')

plot_event(out,0)


