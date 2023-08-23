from utilities.plotting import plot 
import pandas as pd
from Filtering import ParticleFilter 
import time 
import numpy as np
import matplotlib.pyplot as plt
from utilities.utility import IPM
from utilities.CLI_parsing import parse
from utilities.user_data_gen import GenerateSimData

##Run cProfile and snakeviz to extract call stack runtime

def beta(t):
        
      betaMax1=0.1
      theta=0

      #return 0.1+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi))  
      return 0.4 



def main():
    
    params = {"beta":beta,"gamma":0.1,"eta":0.1,"hosp":5.3,"L":90.0,"D":10.0}
    
    initial_state = np.array([100000 ,1000,0,0]) 

    time_series = 343

    start = time.time() 

    #Handles all argument parsing --DONT DELETE--
    args = parse()

    if args.simulate_data is not None: 
       GenerateSimData(params,initial_state,time_series,hospitalization=True)
       file = "./data_sets/beta_test.csv"
    else: 
        file = args.file
        

    if(args.initial_seed) is not None: 
        initial_seed = args.initial_seed
    else: 
        initial_seed = 0.01

    if(args.particles) is not None: 
        num_particles = args.particles
    else: 
        num_particles = 10000



    pf = ParticleFilter(beta_prior=[0.,1.],
                                  population=args.population,
                                  num_particles=num_particles, 
                                  hyperparamters={"sigma1":0.01,"sigma2":0.1,"alpha":0.1},
                                  static_parameters={"gamma":0.1,"eta":0.1,"L":90.0,"D":10.0,"hosp":5.3}, 
                                  init_seed_percent=initial_seed,
                                  filePath=file,
                                  ipm=IPM.SIRH,
                                  estimate_gamma=False,
                                  aggregate=1) 
        

    out = pf.estimate_params(args.iterations if args.iterations is not None and args.iterations < len(pf.observation_data) else len(pf.observation_data))


    end = time.time()

    print("The time of execution of the program is :",
    (end-start), "s") 

    plot(out,0)  
    #plot(out,1)   
    plot(out,2) 



if __name__ == "__main__":
    main()