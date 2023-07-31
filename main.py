

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
                                  num_particles=1000, 
                                  filePath="observations_beta_t.csv");

    betas = pf.estimate_params(99);
    end = time.time(); 
    print("The time of execution of the program is :",
      (end-start), "s")
    
    real_betas = pd.read_csv('real_beta_t.csv');
    real_betas = np.delete(real_betas,0,1);
    t = np.linspace(0,100,num=100);
    plt.scatter(t,real_betas,label="real betas",c="red",s=1.);
    plt.plot(t,betas,label="predicted betas");


    plt.legend(); 

    
    
    plt.show();

    
    

if __name__ == "__main__":
    main()
