from typing import Dict
from epymorph.data import geo_library, ipm_library, mm_library
from epymorph.context import Compartments, SimDType
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Context,Particle, RunInfo,quantiles

import numpy as np



class Epymorph_IF2(Algorithm): 
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler, context: Context) -> None:
        super().__init__(integrator, perturb, resampler, context)

    def initialize(self, params: Dict) -> None:

        '''Initialize list of estimated parameters '''
        for _,(key,val) in enumerate(params.items()): 
            if val == -1: 
                self.context.estimated_params.append(key)


        for i in range(self.context.particle_count): 
            
            '''Populate the initial state of each node using the population key of geo.data'''
            geo = geo_library['pei']()
            pop = geo.data['population'] 

            '''Draw a random int to represent the initial infected'''
            initial_infected = self.context.rng.integers(0,np.round(self.context.seed_size*self.context.population))

            state = np.array([
            [pop[0] - initial_infected, initial_infected, 0],
            [pop[1], 0, 0],
            [pop[2], 0, 0],
            [pop[3], 0, 0],
            [pop[4], 0, 0],
            [pop[5], 0, 0],
        ], dtype=SimDType)
            
            '''Create the particle array '''
            self.particles.append(Particle(param=params.copy(),state=state.copy(),observation=np.array([0 for _ in state])))

        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''
            beta = self.context.rng.uniform(0.,1.)
            self.particles[i].param['beta'] = beta

        '''method to reset the state and observations of the particles after each iteration of M loop, preserving the params'''
    def M_iteration_reset(self): 
        for particle in self.particles:
            particle.observation = np.array([0])
            initial_infected = self.context.rng.uniform(0,self.context.seed_size*self.context.population)
            state = np.concatenate((np.array([self.context.population-initial_infected,initial_infected]),[0 for _ in range(self.context.state_size-2)])) #SIRH model 
            particle.state = state
            self.context.clock.time = 0



    def run(self, info: RunInfo) -> Output:

        '''field initializations for Output'''
        self.output = Output(observation_data=info.observation_data)
        self.output_flags = info.output_flags


        for m in range(0,self.context.additional_hyperparameters['m']): 

            '''operations of M loop are setting the value of m in the hyperparameters, randomly perturbing the params and reseting the state and observation values'''
            self.particles = self.perturb.randomly_perturb(self.context,self.particles)
            self.M_iteration_reset()

            while self.context.clock.time < len(info.observation_data):
                self.particles = self.perturb.randomly_perturb(self.context,self.particles)
                self.particles = self.integrator.propagate(self.particles,self.context)

                weights = self.resampler.compute_weights(info.observation_data[self.context.clock.time],self.particles)
                self.particles = self.resampler.resample(weights=weights,ctx=self.context,particleArray=self.particles)

                '''output updates, not part of the main algorithm'''
                if(m == self.context.additional_hyperparameters['m'] -1): 
                    self.output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
                    self.output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])
                    self.output.average_beta[self.context.clock.time] = np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)])

                #print(f"variance of beta: {variance(np.array([particle.param['beta'] for particle in self.particles]))}")
                self.context.clock.tick()
                #print(f"M:{m} N: {self.context.clock.time}")

            print(f"Iteration:{m} beta: {np.mean([particle.param['beta'] for particle in self.particles])}")

        self.clean_up()
        return Output(np.array([]))


        