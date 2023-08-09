from matplotlib import cm
import Filtering; 
import pandas as pd; 
import numpy as np; 
import matplotlib.pyplot as plt; 
import time; 
import NumericalPropagator

def main():


  start = time.time();  
  pf = Filtering.ParticleFilter(alpha_prior=[0.,1.],
                                population=1000000,
                                num_particles=5,
                                hyperparamters=[0.01,0.5],
                                static_parameters=[5,10,0.1,5],
                                SDH=[0.712593,0.86667],
                                filePath="real_beta_variable.csv"); 



  for _ in range(100): 
    pf.propagate(); 
    pf.resample_with_temp_weights(_); 
    pf.random_perturbations(); 
    
    for particle in pf.particles: 
      print(np.sum(particle[0][0:4])); 
  
    print("\n"); 


  








  end = time.time(); 
  print("The time of execution of the program is :",
    (end-start), "s")
    


  # pf.print_particles(); 

    # t = np.linspace(0,99,num=99);

    # plt.figure(figsize=(20, 20))

    # colors = cm.plasma(np.linspace(0,1,12)); 


    # for i in range(11):
    #   plt.fill_between(t, qtls[:,i], qtls[:,22-i], facecolor = colors[11-i], zorder = i);
      
    
    # plt.xlabel("time(days)"); 
    # plt.ylabel("Number of Infections"); 
    # plt.title("Confidence Intervals"); 



    # plt.show();   

if __name__ == "__main__":
    main()
