from Filtering import Output; 
import numpy as np; 
import matplotlib.pyplot as plt; 
from matplotlib import cm; 

def plot(out:Output,graph:np.int_): 
    match graph:
        case 0:
            t = np.linspace(0,out.time,num=out.time);

            plt.figure(figsize=(10, 10))

            colors = cm.plasma(np.linspace(0,1,12)); 

            #plt.plot(t,out.observations,color = "black",zorder=12); 

            for i in range(11):
                 plt.fill_between(t, out.qtls[:,i], out.qtls[:,22-i], facecolor = colors[11-i], zorder = i);
    
            plt.xlabel("time(days)"); 
            plt.ylabel("Number of Infections"); 
            plt.title("Confidence Intervals"); 

            plt.show();
        
        case 1: 
            t = np.linspace(0,out.time,num=out.time); 
            plt.figure(figsize=(10,10))
            plt.plot(t,out.average_infected); 
        
            plt.title("Average Daily Infections"); 
            plt.xlabel("time(days)"); 
            plt.ylabel("Number of Infections"); 


            plt.show(); 
        
        case 2: 
            t = np.linspace(0,out.time,num=out.time); 

            plt.figure(figsize=(10,10))
            plt.plot(t,out.average_betas); 
        
            plt.title("Average Beta over Time"); 
            plt.xlabel("time(days)"); 
            plt.ylabel("Beta"); 
        
            plt.show(); 

        case _:
            pass; 
