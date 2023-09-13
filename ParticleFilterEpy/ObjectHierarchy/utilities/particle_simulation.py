from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np


from epymorph.clock import Clock
from epymorph.context import (Compartments, Events, SimContext, SimDType,
                              normalize_lists)
from epymorph.geo import Geo
from epymorph.ipm.ipm import Ipm, IpmBuilder
from epymorph.movement.basic import BasicEngine
from epymorph.movement.engine import Movement, MovementBuilder, MovementEngine
from epymorph.simulation import Simulation



class ParticleSimulation: 
    ctx:SimContext
    ipm: Ipm
    mvm: Movement
    mve: MovementEngine
    tick_index: int

    def __init__(self,geo:Geo,ipm_builder:IpmBuilder,mvm_builder:MovementBuilder,tick_index:int,param: dict[str,Any],compartments:Compartments):
        self.ctx = SimContext(nodes = geo.nodes,
                              labels=geo.labels,
                              compartments = ipm_builder.compartments,
                              compartment_tags=ipm_builder.compartment_tags(),
                              events = ipm_builder.events,
                              param=param,
                              clock=Clock(start_date=date(2015,1,1),num_days = 1, taus=mvm_builder.taus),
                              geo = geo.data,
                              rng=np.random.default_rng())
    

        self.tick_index = tick_index
        self.mvm = mvm_builder.build(self.ctx)
        self.mve = BasicEngine(ctx=self.ctx,movement=self.mvm,initial_compartments=compartments)
        self.ipm = ipm_builder.build(self.ctx)

    '''Returns the compartment totals for each population'''
    def get_compartments(self) ->Compartments: 
        return np.array([loc.get_compartments() for loc in self.mve.get_locations()])
    
    '''Advances the simulation by one tau-step. 
    Returns new event counts by population: (N,E)
    '''
    def step(self)->Events: 

        _,N,_,E = self.ctx.TNCE
        events_out = np.empty((N,E),dtype=SimDType)

        tick = self.ctx.clock.ticks[self.tick_index]

        #First do movement
        self.mve.apply(tick)

        #Then for each location
        for n,loc in enumerate(self.mve.get_locations()):
            #calc events by compartment 
            incidence = self.ipm.events(loc,tick)
            #Distribute events 
            self.ipm.apply_events(loc,incidence)
            events_out[n,:] = incidence

        self.tick_index += 1

        return events_out
