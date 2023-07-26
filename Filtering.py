import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 
from scipy.stats import poisson
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
        self.observation_data = np.delete(self.observation_data,0,1);


#calls all internal functions to estimate the parameters
    def estimate_params(self): 
         for t in range(1): 
            self.propagate(); 
            self.resample_with_temp_weights(t); 



    #internal helper function to propagate the particle cloud
    def propagate(self): 
        for i in range(len(self.particles)): 
                self.Propagrator.params = [self.particles[i][1],self.gamma];
                self.Propagrator.state = self.particles[i][0]; 
                temp = self.Propagrator.propagate(); 
                self.particles[i][0]= temp[0]; 
                self.dailyInfected[i] = temp[1]; 



    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t):
        temp_weights = np.zeros(len(self.particles)); 
        for j in range(len(self.particles)):  
            #temp_weights[j] = (self.weights[j] * (self.dailyInfected[j] ** self.observation_data[t+1])/gamma(self.observation_data[t+1])) * np.exp(-self.dailyInfected[j]);  
            temp_weights[j] = self.weights[j] * poisson.pmf(self.observation_data[t+1],self.dailyInfected[j]);
        temp_weights = temp_weights/sum(temp_weights); 
        #print(self.particles); 

        return temp_weights; 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t): 

        temp_weights = self.compute_temp_weights(t); 
        #print((temp_weights));
        indexes = np.zeros(len(self.particles));
        for i in range(len(self.particles)):
            indexes[i] = i;
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights);
        print(new_particle_indexes);

        for i in range(len(self.particles)):
            self.particles[i] = self.particles[int(new_particle_indexes[i])];

        self.print_particles(); 

    #function to print the particles in a human readable format
    def print_particles(self):
        for i in range(len(self.particles)): 
            print(self.particles[i]); 











                 
















