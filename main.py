
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 

def main():
    state = [10000,100,0]; 
    pf = Filtering.ParticleFilter(beta_prior=[0.,1.],
                                  initial_state=state,
                                  num_particles=100, 
                                  filePath="observations_euler.csv");

    betas = pf.estimate_params(1);
    plt.plot(betas);
    plt.show();
    

if __name__ == "__main__":
    main()
