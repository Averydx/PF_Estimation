
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 

def main():
    state = [10000,10,0]; 
    pf = Filtering.ParticleFilter(beta_prior=[0.,1.],
                                  initial_state=state,
                                  num_particles=10000, 
                                  filePath="observations_beta_t.csv");

    betas = pf.estimate_params(28);
    plt.plot(betas); 
    plt.show();

    

if __name__ == "__main__":
    main()
