from typing import Dict, List
import numpy as np
from ObjectHierarchy.Abstract.Algorithm import Algorithm
from ObjectHierarchy.Abstract.Perturb import Perturb
from ObjectHierarchy.Abstract.Integrator import Integrator
from ObjectHierarchy.Abstract.Resampler import Resampler
from ObjectHierarchy.utilities.Output import Output
from ObjectHierarchy.utilities.Utils import Context, Particle, RunInfo,variance,quantiles


'''Implementation of the IF2 algorithm from Ionides et. al.'''
class IF2(Algorithm): 
    def __init__(self, integrator: Integrator, perturb: Perturb, resampler: Resampler,context:Context) -> None:
        super().__init__(integrator, perturb, resampler,context)


    '''One time initializer, call before the first M loop'''
    def initialize(self, params: Dict) -> None:
        super().initialize(params)
                #initialization of the estimated parameters will be done in the override
        for i in range(self.context.particle_count): 
            '''initialize all other estimated parameters here'''

            beta = self.context.rng.uniform(0.,1.)
            self.particles[i].param['beta'] = beta




    '''Algorithm run implementation, runs the M and N loops'''
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
                if(m == self.context.additional_hyperparameters['m'] -2): 
                    self.output.beta_qtls[:,self.context.clock.time] = quantiles([particle.param['beta'] for _,particle in enumerate(self.particles)])
                    self.output.observation_qtls[:,self.context.clock.time] = quantiles([particle.observation for _,particle in enumerate(self.particles)])
                    self.output.average_beta[self.context.clock.time] = np.mean([particle.param['beta'] for _,particle in enumerate(self.particles)])




                #print(f"variance of beta: {variance(np.array([particle.param['beta'] for particle in self.particles]))}")
                self.context.clock.tick()
                #print(f"M:{m} N: {self.context.clock.time}")

            print(f"Iteration:{m} beta: {np.mean([particle.param['beta'] for particle in self.particles])}")

        self.clean_up()
        return self.output





    '''method to reset the state and observations of the particles after each iteration of M loop, preserving the params'''
    def M_iteration_reset(self): 
        for particle in self.particles:
            particle.observation = np.array([0])
            initial_infected = self.context.rng.uniform(0,self.context.seed_size*self.context.population)
            state = np.concatenate((np.array([self.context.population-initial_infected,initial_infected]),[0 for _ in range(self.context.state_size-2)])) #SIRH model 
            particle.state = state
            self.context.clock.time = 0


