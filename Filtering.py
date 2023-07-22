import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 
from scipy.special import gamma; 

class ParticleFilter: 
    def __init__(self,initial_state,beta_prior,num_particles,filePath):

        #Particle and weight initialization
        self.particles = []; 
        self.weights = np.ones(shape=num_particles); 
        self.Propagrator = NumericalPropagator.one_step_propagator(initial_state);
        self.dailyInfected = np.zeros(shape=num_particles); 
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
            self.propagate(); 
            self.compute_temp_weights(t); 



    #internal helper function to propagate the particle cloud
    def propagate(self): 
        for i in range(len(self.particles)): 
                self.Propagrator.params = [self.particles[i][1],self.gamma];
                self.Propagrator.state = self.particles[i][0]; 
                temp = self.Propagrator.propagate(); 
                self.particles[i][0]= temp[0]; 
                self.dailyInfected[i] = temp[1]; 
        #print((self.particles))



    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t):
        temp_weights = np.zeros(len(self.particles)); 
        for j in range(len(self.observation_data)): 
                a = self.weights[j] * (self.dailyInfected[j] ** self.observation_data[t+1]); 
                a = a/gamma(self.observation_data[t+1]); 
                a = np.exp(-self.dailyInfected[j]); 
                
                temp_weights[j] = a; 


        temp_weights = temp_weights/len(temp_weights); 








                 
