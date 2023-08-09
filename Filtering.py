import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 
from scipy.stats import poisson;
from numpy.typing import NDArray; 

class ParticleFilter: 

    particles: list
    weights: NDArray[np.float_]
    Propagator : NumericalPropagator.one_step_propagator
    dailyInfected : NDArray
    static_parameters: NDArray[np.float_]
    population: np.int_
    hyperparameters: NDArray[np.float_]
    SDH: NDArray[np.float_]



    def __init__(self,population,alpha_prior,num_particles,hyperparamters,static_parameters,SDH,filePath):

        #Particle and weight initialization
        self.particles = []; 
        self.weights = np.ones(shape=num_particles); 
        self.Propagator = NumericalPropagator.one_step_propagator();
        self.dailyInfected = np.zeros(shape=num_particles); 
        self.static_parameters = static_parameters; 
        self.population = population; 
        self.hyperparameters = hyperparamters;  
        self.SDH = SDH; 

        #x1 = 0.712593,x2 =  0.86667

        self.weights /= num_particles;  
        

        for i in range(num_particles): 

            initial_infected = np.random.uniform(0,self.population * 0.05); 

            initial_state = [self.population-initial_infected,initial_infected,0,0]; 
            
            alphas = [np.random.uniform(alpha_prior[0],alpha_prior[1]),np.random.uniform(alpha_prior[0],alpha_prior[1]),np.random.uniform(alpha_prior[0],alpha_prior[1])];

            particle = [initial_state,alphas]; 
            self.particles.append(particle); 

        #Obseravtion data initalization
        self.observation_data = pd.read_csv(filePath); 
        self.observation_data = self.observation_data.to_numpy(); 
        self.observation_data = np.delete(self.observation_data,0,1);


#calls all internal functions to estimate the parameters
    def estimate_params(self,time): 
        dI_average = []; 
        qtls = []; 
        dI_average.append(self.average_dI()); 
        for t in range(time): 
            self.propagate(); 
            temp_weights =  self.resample_with_temp_weights(t); 

            self.random_perturbations(); 

            qtls.append(self.quantiles()); 

            self.norm_likelihood(temp_weights_old=temp_weights,t=t)

            dI_average.append(self.average_dI()); 
        return dI_average,np.array(qtls); 

    #internal helper function to propagate the particle cloud
    def propagate(self)->None: 
        for i in range(len(self.particles)): 
                self.Propagator.params = np.concatenate((np.array([self.particles[i][1][0] * np.exp(self.particles[i][1][1] * self.SDH[0] + self.particles[i][1][2] * self.SDH[1])]),self.static_parameters));
                self.Propagator.state = self.particles[i][0];

                temp = self.Propagator.propagate_euler(); 
                self.particles[i][0]= np.array(temp);
                self.dailyInfected[i] = temp[3]; 

    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t) -> np.array:
        temp_weights = np.ones(len(self.particles)); 
        for j in range(len(self.particles)):    
            temp_weights[j] =  poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected[j]);
        
            if(temp_weights[j] == 0):
                temp_weights[j] = 10**-300;
            elif(np.isnan(temp_weights[j])):
                temp_weights[j] = 10**-300;
            elif(np.isinf(temp_weights[j])):
                temp_weights[j] = 10**-300;

        temp_weights = temp_weights/sum(temp_weights); 
        

        return temp_weights; 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t)->np.array: 

        temp_weights = self.compute_temp_weights(t); 
        indexes = np.zeros(len(self.particles));
        for i in range(len(self.particles)):
            indexes[i] = i;
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights);

        for i in range(len(self.particles)):
            self.particles[i] = self.particles[int(new_particle_indexes[i])];

        return temp_weights;  
    
    #applies the geometric random walk to the particles
    def random_perturbations(self)->None:

        #sigma1 is the deviation of the state and sigma2 is the deviation of beta
        sigma1 = self.hyperparameters[0]; 
        sigma2 = self.hyperparameters[1]; 
        
        #for fixed beta use 0.4 and for variable use 0.014 
        C = np.array([[((sigma1)**2)/self.population,0,0,0,0,0,0],
                      [0,(sigma1)**2,0,0,0,0,0],
                      [0,0,(sigma1)**2,0,0,0,0],
                      [0,0,0,(sigma1)**2,0,0,0],
                      [0,0,0,0,(sigma2)**2,0,0],
                      [0,0,0,0,0,(sigma2)**2,0],
                      [0,0,0,0,0,0,(sigma2)**2],
        ]); 
        
        
        for i in range(len(self.particles)):
            

            temp = np.concatenate((self.particles[i][0],self.particles[i][1]));  


            temp = np.log(temp); 
            perturbed = np.random.multivariate_normal(mean = temp,cov = C);
            perturbed = np.exp(perturbed); 
    

            

            s = (np.sum(perturbed[0:4]));
            perturbed[0:4] = (perturbed[0:4] / s) * self.population;            
            # self.particles[i] = [perturbed[0:3],perturbed[3]];
            
    #computes the normalized likelihood
    def norm_likelihood(self,temp_weights_old,t)->None:
        temp_weights = np.zeros(len(self.particles)); 
        for j in range(len(self.particles)): 
            temp_weights[j] = poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected[j]);

        temp_weights /= sum(temp_weights);
    
        for j in range(len(self.particles)):
            self.weights[j] = temp_weights[j]/temp_weights_old[j]; 

    
    #function to print the particles in a human readable format
    def print_particles(self)->None:
        for i in range(len(self.particles)): 
            print(self.particles[i]); 

    #average beta helper function
    def average_beta(self)->np.int_: 
        mean = 0;
        for i in range(len(self.particles)):
            mean += self.particles[i][1]; 
        mean /= len(self.particles);
        return(mean);

    #average betaSI\N helper function
    def average_dI(self): 
        return np.mean(self.dailyInfected); 

    def quantiles(self)->np.array: 
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]);
        return np.quantile(self.dailyInfected, qtlMark);




















                 










                 
