import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 
from scipy.stats import poisson;

class ParticleFilter: 

    def __init__(self,initial_state,beta_prior,num_particles,filePath):

        #Particle and weight initialization
        self.particles = []; 
        self.weights = np.ones(shape=num_particles); 
        self.Propagrator = NumericalPropagator.one_step_propagator(initial_state);
        self.dailyInfected = np.zeros(shape=num_particles); 
        self.gamma = 0.04; 
        self.population = sum(initial_state); 

        
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
        self.observation_data = self.observation_data.to_numpy(); 
        self.observation_data = np.delete(self.observation_data,0,1);


#calls all internal functions to estimate the parameters
    def estimate_params(self,time): 
        beta_series = np.zeros(time);
        for t in range(time): 
            self.propagate(); 
            temp_weights =  self.resample_with_temp_weights(t); 
            #print(temp_weights);
            self.random_perturbations(); 
            #self.norm_likelihood(t,temp_weights);
            beta_series[t] = np.mean(self.particles[t][1]);
        return beta_series; 

    #internal helper function to propagate the particle cloud
    def propagate(self): 
        for i in range(len(self.particles)): 
                self.Propagrator.params = [self.particles[i][1],self.gamma];
                self.Propagrator.state = self.particles[i][0]; 
                temp = self.Propagrator.propagate(); 
                self.particles[i][0]= np.array(temp[0]); 
                self.dailyInfected[i] = temp[1]; 

    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t):
        temp_weights = np.zeros(len(self.particles)); 
        for j in range(len(self.particles)):  
            #temp_weights[j] = (self.weights[j] * (self.dailyInfected[j] ** self.observation_data[t+1])/gamma(self.observation_data[t+1])) * np.exp(-self.dailyInfected[j]);  
            temp_weights[j] = poisson.pmf(self.observation_data[t+1],self.dailyInfected[j]);
            # if(temp_weights[j] == 0):
            #     temp_weights[j] += 10**-300; 
        
        temp_weights = temp_weights/sum(temp_weights); 

        return temp_weights; 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t): 

        temp_weights = self.compute_temp_weights(t); 
        print((temp_weights));
        indexes = np.zeros(len(self.particles));
        for i in range(len(self.particles)):
            indexes[i] = i;
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights);

        for i in range(len(self.particles)):
            self.particles[i] = self.particles[int(new_particle_indexes[i])];

        return temp_weights;  
    
    #applies the geometric random walk to the particles
    def random_perturbations(self):
        sigma1 = 0.; 
        sigma2 = 0.1; 

        C = np.array([[((sigma1)**2)/self.population,0,0,0],
                      [0,(sigma1)**2,0,0],
                      [0,0,(sigma1)**2,0],
                      [0,0,0,(sigma2)**2],
        ]); 
        
        

        for i in range(len(self.particles)):
            
            print(self.particles[i]);

            temp = []; 
            for  j in range(len(self.particles[i][0])):
                temp.append(self.particles[i][0][j]); 
    
            temp.append(self.particles[i][1]); 
    

            temp = np.log(temp); 
            perturbed = np.random.multivariate_normal(mean = temp,cov = C);

            perturbed = np.exp(perturbed); 
            s = (sum(perturbed[0:3]));
            for j in range(0,3):
                perturbed[j] = perturbed[j] / s; 
                perturbed[j] = perturbed[j] * self.population; 
            
            self.particles[i] = [(perturbed[0:3]).astype(int),perturbed[3]];
            
    
            print(self.particles[i]);
            print("\n");

    def norm_likelihood(self,t,_temp_weights):
        new_temp_weights = np.zeros(len(_temp_weights));
        for i in range(len(_temp_weights)): 
            new_temp_weights[i] = poisson.pmf(self.observation_data[t+1],self.dailyInfected[i]);
        
        
        new_temp_weights = new_temp_weights/sum(new_temp_weights); 


        for i in range(len(_temp_weights)):
            self.weights[i] = new_temp_weights[i]/_temp_weights[i]; 


    #function to print the particles in a human readable format
    def print_particles(self):
        for i in range(len(self.particles)): 
            print(self.particles[i]); 











                 
















