from matplotlib import cm
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 
import time; 
import Datagen; 

def main():
    start = time.time(); 

    def beta(t):
        
      # betaMax1=0.8
      # theta=0

      # return 0.1+betaMax1*(1.0-np.cos(theta+t/7/52*2*np.pi));  
      return 0.1; 




    dg = Datagen.DataGenerator(beta,0.04,0.02,[10000 ,100,0],100,data_name="beta_test",noise=True); 

    dg.generate_data(); 
    dg.plot_daily_infected(); 

    dg.plot_beta(); 

    #dg.plot_states(); 

    pf = Filtering.ParticleFilter(beta_prior=[0.,1.],
                                  population=10100,
                                  num_particles=5000, 
                                  hyperparamters=[0.01,0.1 ,0.1],
                                  filePath="beta_test.csv",
                                  estimate_gamma=False); 
        

    time_series = 99; 
    betas,dI,qtls = pf.estimate_params(time_series);
    end = time.time();

    print("The time of execution of the program is :",
      (end-start), "s")
    
    t = np.linspace(0,time_series,num=time_series);

    plt.figure(figsize=(20, 20))

    colors = cm.plasma(np.linspace(0,1,12)); 

    plt.plot(t,np.squeeze(pf.observation_data[:time_series]),color = "black",zorder=12); 

    for i in range(11):
      plt.fill_between(t, qtls[:,i], qtls[:,22-i], facecolor = colors[11-i], zorder = i);
    
    plt.xlabel("time(days)"); 
    plt.ylabel("Number of Infections"); 
    plt.title("Confidence Intervals"); 

    plt.show();

    plt.plot(t,dg.beta[0:time_series]); 
    plt.plot(t,betas);

    plt.show();   




if __name__ == "__main__":
    main()
