import numpy as np 
import pandas as pd 
import ParticleFilter.NumericalPropagator as NumericalPropagator 
from scipy.stats import poisson
from scipy.stats import nbinom
from numpy.typing import NDArray
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 
from ParticleFilter.utilities.utility import *

class Output:
    average_betas: NDArray[np.float_]
    average_infected: NDArray[np.float_]
    qtls: NDArray
    observations:NDArray[np.float_]
    time: int

    def __init__(self,time,observations) -> None:
        self.average_betas = (np.zeros(shape=(time)))
        self.sim_obvs = (np.zeros(shape=time)) 
        self.qtls = np.empty(shape=(time,23)) 
        self.time = time 
        self.observations = observations 


     #average beta helper function
    def average_beta(self,particles,attribs:dict ,t)->None: 
        mean = 0
        for _,particle in enumerate(particles): 
            mean += particle[attribs['compartments']]
        mean /= len(particles)
        self.average_betas[t] = mean

    #average betaSI\N helper function
    def average_dI(self,sim_obvs:NDArray[np.float_],t)->None: 
        self.sim_obvs[t] = np.mean(sim_obvs) 

    def quantiles(self,sim_obvs:NDArray[np.float_],t)->None: 
        qtlMark = 1.00*np.array([0.010, 0.025, 0.050, 0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550, 0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950, 0.975, 0.990])
        self.qtls[t]= np.quantile(sim_obvs, qtlMark)


class ParticleFilter: 

    out: Output
    particles: NDArray
    static_parameters: dict
    weights: NDArray[np.float_]
    Propagator: NumericalPropagator.one_step_propagator
    sim_obvs: NDArray[np.float_]
    population: int
    hyperparameters: dict
    filePath: str
    estimate_gamma: bool
    attribs : dict
    aggregate: int
    aggregatedSimObvs: NDArray[np.float_]
    forecast:bool


    def __init__(self,population,beta_prior,num_particles,hyperparamters,static_parameters,init_seed_percent,filePath,ipm = IPM.SIR,estimate_gamma = False,aggregate = 1,forecast=False):

        #Particle and weight initialization

        #hyperparameters are sigma1,sigma2 and alpha

        self.attribs = ipm_attributes(ipm)
        self.estimate_gamma = estimate_gamma 
        if(not self.estimate_gamma): 
            self.particles = np.empty((num_particles,self.attribs['compartments'] + 1)) 
        else: 
            self.particles = np.empty((num_particles,self.attribs['compartments'] + 2)) 

        self.static_parameters = static_parameters

        self.weights = np.ones(shape=num_particles) 
        self.Propagator = NumericalPropagator.one_step_propagator()
        self.sim_obvs = np.zeros(shape=num_particles) 


        self.population = population 
        self.hyperparameters = hyperparamters 
        self.static_parameters = static_parameters 
        self.aggregate = aggregate
        self.aggregatedSimObvs = np.zeros(shape = num_particles)
        self.forecast = forecast


        #normalize weights
        self.weights /= num_particles  
        

        for i,_ in enumerate(self.particles): 

            match self.attribs["ipm"]: 
                case "SIR": 
                    initial_infected = np.random.uniform(0,self.population * init_seed_percent) 
                    initial_state = [self.population - initial_infected,initial_infected,0] 
                    
            
                case "SIRH": 
                    initial_infected = np.random.uniform(0,self.population * init_seed_percent)
                    initial_state = initial_state = [self.population - initial_infected,initial_infected,0,0] 

                case _:
                    print("Failed to locate IPM")
                    exit(0)
            
                
            beta = [
                np.random.uniform(low=beta_prior[0],
                                  high=beta_prior[1])] 
            
            #beta = [0.4]


            if(not self.estimate_gamma): 
                self.particles[i] = np.concatenate((initial_state,beta)) 
            else:
                gamma = [np.random.uniform(0,0.5)] 
                self.particles[i] = np.concatenate((initial_state,beta,gamma)) 
        #Obseravtion data initalization
        self.observation_data = pd.read_csv(filePath) 
        self.observation_data = np.squeeze(self.observation_data.to_numpy()) 
        self.observation_data = np.delete(self.observation_data,0,1)


#calls all internal functions to estimate the parameters
    def estimate_params(self,time)->Output: 

        self.out = Output(time,self.observation_data) 
        
        self.out.average_beta(self.particles,self.attribs,0)
        self.out.average_dI(self.sim_obvs,0)  

        for t in range(time): 

            for _ in range(self.aggregate): 
                self.propagate()
                self.aggregatedSimObvs += self.sim_obvs 

            if (self.forecast is True) and (t > len(self.observation_data)/3): 
                pass
                
                
                
            else: 
                temp_weights =  self.resample_with_temp_weights(t) 
                
                self.random_perturbations() 
                self.norm_likelihood(temp_weights_old=temp_weights,t=t)
            
            

            self.out.average_beta(self.particles,self.attribs,t) 

            self.out.average_dI(self.aggregatedSimObvs,t) 
            self.out.quantiles(self.aggregatedSimObvs,t) 

                
            self.aggregatedSimObvs = np.zeros(shape = len(self.particles))

            print(f"Iteration:{t} of {time}") 

        return self.out 

    #internal helper function to propagate the particle cloud
    def propagate(self)->None: 
        for i,particle in enumerate(self.particles): 
                 
                if(self.estimate_gamma):  
                    self.Propagator.static_params = self.static_parameters
                    self.Propagator.estimated_params = {"beta":particle[self.attribs['compartments']], "gamma":particle[self.attribs['compartments']+1]}
                else: 
                    self.Propagator.static_params = self.static_parameters
                    self.Propagator.estimated_params = {"beta": particle[self.attribs['compartments']]}
                self.Propagator.state = particle[0:self.attribs['compartments']] 

                match self.attribs['ipm']: 
                    case "SIR": 
                        state,dailyObv = self.Propagator.propagate_euler();  
                    case "SIRH": 
                        state,dailyObv = self.Propagator.propagate_euler_H(); 
                    case _:
                        print("Failed to locate numerical one step propagator for the given IPM")
                        exit(0)
                


                self.particles[i][:self.attribs['compartments']]= np.array(state) 
                self.sim_obvs[i] = dailyObv 
                if(self.estimate_gamma): 
                    particle[self.attribs['compartments']] *= np.exp(self.hyperparameters['alpha'] * np.random.standard_normal()) 






    #internal helper function to compute weights based on observations
    def compute_temp_weights(self,t)->NDArray[np.float_]:
        temp_weights = np.ones(len(self.particles)) 
          
        temp_weights = self.weights * poisson.pmf(np.round(self.observation_data[t]),self.aggregatedSimObvs)
        #temp_weights = self.weights * nbinom.pmf(k=np.round(self.observation_data[t+1]),n=self.sim_obvs,p=0.5,loc=self.sim_obvs) 

        for j,_ in enumerate(self.particles):  
            if(temp_weights[j] == 0):
                temp_weights[j] = 10**-300 
            elif(np.isnan(temp_weights[j])):
                temp_weights[j] = 10**-300
            elif(np.isinf(temp_weights[j])):
                temp_weights[j] = 10**-300


        #normalize temp weights
        temp_weights = temp_weights/np.sum(temp_weights) 
        

        return temp_weights 

    #resample based on the temp weights that were computed
    def resample_with_temp_weights(self,t)->NDArray[np.float_]: 

        temp_weights = self.compute_temp_weights(t) 
        indexes = np.arange(len(self.particles))
        
        new_particle_indexes = np.random.choice(a=indexes, size=len(self.particles), replace=True, p=temp_weights)

        particle_copy = np.copy(self.particles)
        for i,_ in enumerate(self.particles):
            self.particles[i] = particle_copy[int(new_particle_indexes[i])]

        return temp_weights  
    
    #applies the geometric random walk to the particles
    def random_perturbations(self)->None:

        #sigma1 is the deviation of the state and sigma2 is the deviation of beta
        
        match self.attribs['compartments']: 
            case 3: 
                C = [(self.hyperparameters['sigma1'])**2/self.population,self.hyperparameters['sigma1']**2,self.hyperparameters['sigma1']**2,self.hyperparameters['sigma2']**2]
            case 4: 
                C = [(self.hyperparameters['sigma1'])**2/self.population,self.hyperparameters['sigma1']**2,self.hyperparameters['sigma1']**2,self.hyperparameters['sigma1'] **2, self.hyperparameters['sigma2']**2]
            case _: 
                print("Compartment number not matched by preloaded covarinace matrices")
                exit(0)
       
        C = np.diag(C)
        A = np.linalg.cholesky(C)

        
        for i,particle in enumerate(self.particles): 

            if self.estimate_gamma is True: 
                E = self.expectation_loggamma()
                

            

            temp = np.log(particle) 
            #perturbed = np.random.multivariate_normal(mean = temp,cov = C,check_valid='ignore')
            perturbed = multivariate_normal(temp,A)
            perturbed = np.exp(perturbed) 

            perturbed[0:self.attribs['compartments']]= (perturbed[0:self.attribs['compartments']] / np.sum(perturbed[0:self.attribs['compartments']])) * self.population 
            self.particles[i] = perturbed
            
    #computes the normalized likelihood

    #The hospitalization rate is 0.01

    def norm_likelihood(self,temp_weights_old,t)->None:
        temp_weights = np.zeros(len(self.particles)) 
        temp_weights = poisson.pmf(np.round(self.observation_data[t]),self.aggregatedSimObvs)

        temp_weights /= np.sum(temp_weights)
    
        self.weights = temp_weights/temp_weights_old 

    def expectation_loggamma(self):
        #Assumes particles include gamma at particles[0:self.attribs['compartments] + 1]
        E = 0
        for i,particle in enumerate(self.particles): 
            E += self.weights[i] * np.log(particle[self.attribs['compartments']  +1])

        return E
    
    def loggamma_variance(self,E): 
        sigma = np.zeros(len(self.particles))
        for i,particle in enumerate(self.particles): 
            sigma[i] = self.weights[i] * (np.log(particle[self.attribs['compartments']  +1]) - E)**2
        return sigma
            
        
            
    
    #function to print the particles in a human readable format
    def print_particles(self):
        for i in range(len(self.particles)): 
            print(self.particles[i]) 

    



   



















                 






