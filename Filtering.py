import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 

class ParticleFilter: 
    def __init__(self,initial_state,beta_prior,num_particles,filePath):

        #Particle and weight initialization
        self.particles = []; 
        self.weights = np.ones(shape=num_particles); 
        self.Propagrator = NumericalPropagator.one_step_propagator(initial_state);
        self.gamma = 0.04; 
        for i in range(len(self.weights)): 
             self.weights[i] /= num_particles;  
        for i in range(num_particles): 
            beta = (
                np.random.uniform(low=beta_prior[0],
                                  high=beta_prior[1])); 
            particle = [initial_state,beta]; 
            self.particles.append(particle); 

        #Obseravtion data initalization
        self.observation_data = pd.read_csv(filePath); 
        self.observation_data =self.observation_data.to_numpy(); 

    def estimate_params(self): 
         for t in range(1): 
             for i in range(len(self.particles)): 
                self.Propagrator.params = [self.particles[i][1],self.gamma];
                print(self.particles[i][0]); 
                self.Propagrator.state = self.particles[i][0]; 
                print(self.Propagrator.propagate()); 

    #internal helper function to compute weights based on observations





                 








