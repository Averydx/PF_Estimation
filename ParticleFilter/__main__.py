from ParticleFilter.utilities.plotting import plot 
from ParticleFilter.Filtering import ParticleFilter 
import time 
import numpy as np
import pandas as pd
from ParticleFilter.utilities.utility import IPM
from ParticleFilter.utilities.CLI_parsing import parse
from ParticleFilter.utilities.user_data_gen import GenerateSimData

##Run cProfile and snakeviz to extract call stack runtime

def beta(t):
        
      betaMax1=0.1
      theta=0

      return 0.1+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  


def main():
    
    params = {"beta":beta,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0}
    
    initial_state = np.array([100000 ,1000,0,0]) 

    time_series = 343

    start = time.time() 

    #Handles all argument parsing --DONT DELETE--
    args = parse()

    if args.simulate_data: 
       GenerateSimData(params,initial_state,time_series,hospitalization=True)
       file = "./data_sets/beta_test.csv"
    else: 
        file = args.file
        

    if(args.initial_seed): 
        initial_seed = args.initial_seed
    else: 
        initial_seed = 0.01

    if(args.particles): 
        num_particles = args.particles
    else: 
        num_particles = 10000

    if(args.forecast): 
        forecast_bool = True
    else:
        forecast_bool = False



    pf = ParticleFilter(beta_prior=[0.,1.],
                                  population=args.population,
                                  num_particles=num_particles, 
                                  hyperparamters={"sigma1":0.01,"sigma2":0.1,"alpha":0.1},
                                  static_parameters={"gamma":0.1,"eta":0.1,"L":90.0,"D":10.0,"hosp":5.3}, 
                                  init_seed_percent=initial_seed,
                                  filePath=file,
                                  ipm=IPM.SIRH,
                                  estimate_gamma=False,
                                  aggregate=1,
                                  forecast = forecast_bool) 
        

    out = pf.estimate_params(args.iterations if args.iterations and args.iterations < len(pf.observation_data) else len(pf.observation_data))

    if(args.simulate_data): 
        out.real_beta = pd.read_csv('./data_sets/beta_test_beta.csv')
        out.real_beta = np.squeeze(out.real_beta.to_numpy()) 
        out.real_beta = np.delete(out.real_beta,0,1)

    end = time.time()

    print("The time of execution of the program is :",
    (end-start), "s") 



    #plot(out,0)  
    plot(out,1)   
    plot(out,2) 





if __name__ == "__main__":
    main()