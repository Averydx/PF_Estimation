

import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 
import time; 

def main():
    start = time.time(); 
    state = [10000,100,0]; 
    pf = Filtering.ParticleFilter(beta_prior=[0.,1.],
                                  initial_state=state,
                                  num_particles=10000, 
                                  filePath="observations_beta_t.csv");

    betas = pf.estimate_params(99);
    end = time.time(); 
    print("The time of execution of the program is :",
      (end-start) * 10**3, "ms")
    plt.plot(betas);
    real_betas = pd.read_csv('real_beta_t.csv');
    plt.plot(real_betas.to_numpy());
    
    plt.show();

    
    

if __name__ == "__main__":
    main()
