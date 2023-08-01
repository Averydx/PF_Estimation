

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
                                  num_particles=5000, 
                                  filePath="observations_variable_beta.csv");

    aggregate_betas = []; 
    aggregate_dI = []; 
    for _ in range(1): 
        
      betas,dI = pf.estimate_params(99);
      aggregate_betas.append(betas); 
      aggregate_dI.append(dI); 


    end = time.time(); 
    print("The time of execution of the program is :",
      (end-start), "s")
    
    real_betas = pd.read_csv('real_beta_variable.csv');
    real_betas = np.delete(real_betas,0,1);

    t = np.linspace(0,100,num=100);

    plt.figure(figsize=(20, 20))

    fig, (ax1, ax2)  = plt.subplots(1, 2);
    ax1.scatter(t,real_betas,label="real betas",c="red",s=1.);
    for i in range(len(aggregate_betas)): 
      ax1.plot(t,aggregate_betas[i],color="b", alpha=0.1);
    ax1.legend(); 
    ax2.scatter(t,pf.observation_data, label="real observations",c="red",s=1.); 

    for i in range(len(aggregate_dI)): 
      ax2.plot(t,aggregate_dI[i],color="b", alpha=0.1);


    ax2.legend();  

    plt.show();

    
    

if __name__ == "__main__":
    main()
