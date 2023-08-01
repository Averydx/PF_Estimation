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
        betas = [];
        dI_average = []; 
        betas.append(self.average_beta())
        dI_average.append(self.average_dI()); 
        for t in range(time): 
            self.propagate(); 
            temp_weights =  self.resample_with_temp_weights(t); 
            self.random_perturbations(); 
            self.norm_likelihood(temp_weights_old=temp_weights,t=t)
            betas.append(self.average_beta());
            dI_average.append(self.average_dI()); 
        return betas,dI_average; 

    #internal helper function to propagate the particle cloud
    def propagate(self): 
        for i in range(len(self.particles)): 
                self.Propagrator.params = [self.particles[i][1],self.gamma];
                self.Propagrator.state = self.particles[i][0]; 
                temp = self.Propagrator.propagate_euler(); 
                self.particles[i][0]= np.array(temp[0]); 
                self.dailyInfected[i] = temp[1]; 

    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t):
        temp_weights = np.ones(len(self.particles)); 
        for j in range(len(self.particles)):    
            temp_weights[j] =  poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected[j]);
            # print(f"Weight: {temp_weights[j]}");
            # print(f"params: {self.particles[j][1]}"); 
            # print(f"Observation: {self.observation_data[t+1]}, Euler prediction: {self.dailyInfected[j]}\n");
        
            if(temp_weights[j] == 0):
                temp_weights[j] = 10**-300;
            elif(np.isnan(temp_weights[j])):
                temp_weights[j] = 10**-300;
            elif(np.isinf(temp_weights[j])):
                temp_weights[j] = 10**-300;

        temp_weights = temp_weights/sum(temp_weights); 
        

        return temp_weights; 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t): 

        temp_weights = self.compute_temp_weights(t); 
        indexes = np.zeros(len(self.particles));
        for i in range(len(self.particles)):
            indexes[i] = i;
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights);

        for i in range(len(self.particles)):
            self.particles[i] = self.particles[int(new_particle_indexes[i])];

        return temp_weights;  
    
    #applies the geometric random walk to the particles
    def random_perturbations(self):

        #sigma1 is the deviation of the state and sigma2 is the deviation of beta
        sigma1 = 0.01; 
        sigma2 = 0.1; 
        
        #for fixed beta use 0.4 and for variable use 0.014 
        C = np.array([[((sigma1)**2)/self.population,0,0,0],
                      [0,(sigma1)**2,0,0],
                      [0,0,(sigma1)**2,0],
                      [0,0,0,(sigma2)**2],
        ]); 
        
        
        for i in range(len(self.particles)):
            

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
            
            self.particles[i] = [perturbed[0:3],perturbed[3]];
            
    #computes the normalized likelihood
    def norm_likelihood(self,temp_weights_old,t):
        temp_weights = np.zeros(len(self.particles)); 
        for j in range(len(self.particles)): 
            temp_weights[j] = poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected[j]);

        temp_weights /= sum(temp_weights);
    
        for j in range(len(self.particles)):
            self.weights[j] = temp_weights[j]/temp_weights_old[j]; 


    #function to print the particles in a human readable format
    def print_particles(self):
        for i in range(len(self.particles)): 
            print(self.particles[i]); 

    #average beta helper function
    def average_beta(self): 
        mean = 0;
        for i in range(len(self.particles)):
            mean += self.particles[i][1]; 
        mean /= len(self.particles);
        return(mean);

    def average_dI(self): 
        return np.mean(self.dailyInfected); 











                 
