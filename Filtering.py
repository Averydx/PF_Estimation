import numpy as np; 
import pandas as pd; 
import NumericalPropagator; 
from scipy.stats import poisson;
from scipy.stats import nbinom;
from numpy.typing import NDArray
from utility import multivariate_normal

class Output:
    average_betas: NDArray[np.float_]
    average_infected: NDArray[np.float_]
    qtls: NDArray
    observations:NDArray[np.float_]
    time: int

    def __init__(self,time,observations) -> None:
        self.average_betas = (np.zeros(shape=(time)));
        self.average_infected = (np.zeros(shape=time)); 
        self.qtls = np.empty(shape=(time,23)); 
        self.time = time; 
        self.observations = observations; 

     #average beta helper function
    def average_beta(self,particles,t)->None: 
        mean = 0
        for _,particle in enumerate(particles): 
            mean += particle[3]
        mean /= len(particles)
        self.average_betas[t] = mean

    #average betaSI\N helper function
    def average_dI(self,dailyInfected:NDArray[np.float_],t)->None: 
        self.average_infected[t] = np.mean(dailyInfected); 

    def quantiles(self,dailyInfected:NDArray[np.float_],t)->None: 
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990]);
        self.qtls[t]= np.quantile(dailyInfected, qtlMark);


class ParticleFilter: 

    out: Output
    particles: list
    static_parameters: list
    weights: NDArray[np.float_]
    Propagator: NumericalPropagator.one_step_propagator
    dailyInfected: NDArray[np.float_]
    gamma: float
    population: int
    hyperparameters: list
    filePath: str
    estimate_gamma: bool


    def __init__(self,population,beta_prior,num_particles,hyperparamters,static_parameters,init_seed_percent,filePath,estimate_gamma = False):

        #Particle and weight initialization


        #hyperparameters are sigma1,sigma2 and alpha

        self.particles = np.empty((num_particles,4)); 
        self.static_parameters = list;


        self.weights = np.ones(shape=num_particles); 
        self.Propagator = NumericalPropagator.one_step_propagator();
        self.dailyInfected = np.zeros(shape=num_particles); 
        self.gamma = 0.04; 
        self.estimate_gamma = estimate_gamma; 
        self.population = population; 
        self.hyperparameters = hyperparamters; 
        self.static_parameters = static_parameters;    


        #normalize weights
        self.weights /= num_particles;  
        

        for i,_ in enumerate(self.particles): 

            initial_infected = np.random.uniform(0,self.population * init_seed_percent); 
            initial_state = [self.population - initial_infected,initial_infected,0]; 

            if(estimate_gamma):
                initial_static_parameters = [np.random.uniform(0,0.1),self.static_parameters]; 
            else:

                #0.02 is rate at which people return to S
                initial_static_parameters = static_parameters; 

            beta = [
                np.random.uniform(low=beta_prior[0],
                                  high=beta_prior[1])]; 
            

            self.particles[i] = np.concatenate((initial_state,beta)); 

        #Obseravtion data initalization
        self.observation_data = pd.read_csv(filePath); 
        self.observation_data = np.squeeze(self.observation_data.to_numpy()); 
        self.observation_data = np.delete(self.observation_data,0,1);


#calls all internal functions to estimate the parameters
    def estimate_params(self,time)->Output: 

        self.out = Output(time,self.observation_data); 
        
        self.out.average_beta(self.particles,0);
        self.out.average_dI(self.dailyInfected,0);  

        for t in range(time): 
            self.propagate(); 
            temp_weights =  self.resample_with_temp_weights(t); 

            self.random_perturbations(); 
            self.norm_likelihood(temp_weights_old=temp_weights,t=t)

            self.out.average_beta(self.particles,t); 
            self.out.average_dI(self.dailyInfected,t); 
            self.out.quantiles(self.dailyInfected,t); 

            print(f"Iteration:{t} of {time}"); 

        return self.out; 

    #internal helper function to propagate the particle cloud
    def propagate(self)->None: 
        for i,particle in enumerate(self.particles): 
                 
                self.Propagator.params = [particle[3],self.static_parameters[0],self.static_parameters[1]];
                self.Propagator.state = particle[:3]; 
                state,dailyInf = self.Propagator.propagate_euler(); 
                self.particles[i][:3]= np.array(state); 
                self.dailyInfected[i] = dailyInf; 



                if(self.estimate_gamma): 
                    particle[1] = particle[1] * np.exp(self.hyperparameters[2] * np.random.normal(0,1)); 


    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t)->NDArray[np.float_]:
        temp_weights = np.ones(len(self.particles)); 
          
        temp_weights = self.weights *  poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected);
        #temp_weights = self.weights * nbinom.pmf(k=np.round(self.observation_data[t+1]),n=self.dailyInfected,p=0.5,loc=self.dailyInfected); 

        for j,_ in enumerate(self.particles):  
            if(temp_weights[j] == 0):
                temp_weights[j] = 10**-300; 
            elif(np.isnan(temp_weights[j])):
                temp_weights[j] = 10**-300;
            elif(np.isinf(temp_weights[j])):
                temp_weights[j] = 10**-300;


        #normalize temp weights
        temp_weights = temp_weights/np.sum(temp_weights); 
        

        return temp_weights; 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t)->NDArray[np.float_]: 

        temp_weights = self.compute_temp_weights(t); 
        indexes = np.arange(len(self.particles));
        
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights);

        for i,_ in enumerate(self.particles):
            self.particles[i] = self.particles[int(new_particle_indexes[i])];

        return temp_weights;  
    
    #applies the geometric random walk to the particles
    def random_perturbations(self)->None:

        #sigma1 is the deviation of the state and sigma2 is the deviation of beta
        [sigma1,sigma2] = self.hyperparameters;  
        
        C = np.diag([(sigma1)**2/self.population,sigma1**2,sigma1**2,sigma2**2]); 
        A = np.linalg.cholesky(C)
        
        for i,particle in enumerate(self.particles): 
    
            temp = np.log(particle); 
            #perturbed = np.random.multivariate_normal(mean = temp,cov = C,check_valid='ignore');
            perturbed = multivariate_normal(temp,A)
            perturbed = np.exp(perturbed); 

            perturbed[0:3]= (perturbed[0:3] / np.sum(perturbed[0:3])) * self.population; 
            self.particles[i] = perturbed;
            
    #computes the normalized likelihood

    #The hospitalization rate is 0.01

    def norm_likelihood(self,temp_weights_old,t)->None:
        temp_weights = np.zeros(len(self.particles)); 
        temp_weights = poisson.pmf(np.round(self.observation_data[t+1]),self.dailyInfected);

        temp_weights /= np.sum(temp_weights);
    
        self.weights = temp_weights/temp_weights_old; 

    
    #function to print the particles in a human readable format
    def print_particles(self):
        for i in range(len(self.particles)): 
            print(self.particles[i]); 

   



















                 






