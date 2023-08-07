from matplotlib import cm
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 
import time; 

def main():
    start = time.time(); 
    state = [100000,100,0]; 
    pf = Filtering.ParticleFilter(beta_prior=[0.,1.],
                                  population=100100,
                                  num_particles=5000, 
                                  hyperparamters=[0.01,0.1],
                                  filePath="observations_variable_beta_poisson.csv");
    aggregate_betas = []; 
    aggregate_dI = []; 
    for _ in range(1): 
        
      betas,dI,qtls = pf.estimate_params(99);
      aggregate_betas.append(betas); 
      aggregate_dI.append(dI); 


    end = time.time(); 
    print("The time of execution of the program is :",
      (end-start), "s")
    
    real_betas = pd.read_csv('real_variable_beta_poisson.csv');
    real_betas = np.delete(real_betas,0,1);

    t = np.linspace(0,99,num=99);

    plt.figure(figsize=(20, 20))

    colors = cm.plasma(np.linspace(0,1,12)); 


    for i in range(11):
      plt.fill_between(t, qtls[:,i], qtls[:,22-i], facecolor = colors[11-i], zorder = i);
      
    
    plt.xlabel("time(days)"); 
    plt.ylabel("Number of Infections"); 
    plt.title("Confidence Intervals"); 



    plt.show();   

if __name__ == "__main__":
    main()
