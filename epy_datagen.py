from datetime import date
from functools import partial
import jsonpickle

import numpy as np

from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.initializer import single_location
from epymorph.simulation import Simulation
from epymorph.run import plot_event,plot_pop
from epymorph.util import stridesum
from epymorph.geo import CentroidDType, Geo
import matplotlib.pyplot as plt
import pandas as pd

# Note: the 'library' dictionaries contain functions which load the named component,
# so you have to apply the function to get the _actual_ component.

# The 'pei' model family (IPM/MM/GEO) implement an SIRS model in 6 US states.
# (Remember: it is possible to mix-and-match the models!)


label = ['AZ','CA']
geo = Geo(
        nodes=len(label),
        labels=label,
        data={
            'label': np.array(label, dtype=str),
            'geoid': np.array(['04','05'], dtype=str),
            'centroid': np.array([(-111.856111, 34.566667)], dtype=CentroidDType),
            'population': np.array([100_000,100_000], dtype=np.int64),
            'commuters': np.array([[0,1000],
                                   [10000,0]], dtype=np.int64)
        }
    )

def beta(t):
        
    betaMax1=0.1
    theta=0

    return 0.3+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
    #return 0.1

beta_cos = np.array([beta(t) for t in range(300)])

betas = np.zeros((300,2))
betas[:,0] = beta_cos
betas[:,1] = np.array([0.4 for _ in range(300)])

sim = Simulation(
    geo=geo,
    ipm_builder=ipm_library['sirs'](),
    mvm_builder=mm_library['pei']()
)


out = sim.run(
    param={
        'beta': betas,
        'gamma':0.25,
        'xi':1/90,
        'theta': 0.1,
        'move_control': 0.9,
        'infection_duration': 4.0,
        'immunity_duration': 90.0,
    },
    start_date=date(2015, 1, 1),
    duration_days=300,
    initializer=partial(single_location, location=0, seed_size=10_000),
    rng=np.random.default_rng(1)
)

incidence = []
for pop_idx in range(out.ctx.nodes):
    values = stridesum( out.incidence[:,pop_idx,0],len(out.ctx.clock.taus))
    incidence.append(values)
incidence = np.array(incidence)

df = pd.DataFrame(incidence)
df.to_csv('./data_sets/epy_inc.csv')

plot_event(out,0)


