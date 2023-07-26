
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 

def main():
    state = [10000,10,0]; 
    pf = Filtering.ParticleFilter(beta_prior=[0.,0.3],
                                  initial_state=state,
                                  num_particles=1000, 
                                  filePath="observations.csv");

    pf.estimate_params()
    

if __name__ == "__main__":
    main()
