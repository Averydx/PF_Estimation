from ParticleFilter.Filtering import Output 
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import cm, ticker 

def plot(out:Output,graph:int): 
    match graph:
        case 0:
            t = np.linspace(0, out.time, num=out.time)
            fig, (ax1, ax2) = plt.subplots(1, 2, dpi=200)
            fig.suptitle('Particle Filter output')
            ax1.set_title("Confidence Intervals")
            ax2.set_title("Average Beta over time")
            colors = cm.plasma(np.linspace(0, 1, 12))
            ax1.set_ylim(0, 300)
            ax1.plot(t, out.observations[:out.time], color="blue", zorder=12)
            for i in range(11):
                ax1.fill_between(t, out.qtls[:, i], out.qtls[:, 22 - i], facecolor=colors[11 - i], zorder=i)
            ax2.set_ylim(0, 1)
            ax2.plot(t, out.average_betas)
            if len(out.real_beta) > 0:
                ax2.plot(t, out.real_beta)
            # To show all the y-ticks
            ax1.yaxis.set_major_locator(ticker.LinearLocator())
            ax2.yaxis.set_major_locator(ticker.LinearLocator())
            plt.tight_layout()
            plt.show()

        case 1: 
            t = np.linspace(0,out.time,num=out.time) 
            plt.figure(figsize=(10,10))
            plt.plot(t,out.sim_obvs) 
        
            plt.plot(t,out.observations[:out.time],color = "black",zorder=12) 

            plt.title("Average Daily Infections") 
            plt.xlabel("time(days)") 
            plt.ylabel("Number of Infections") 


            plt.show() 
        
        case 2: 
            t = np.linspace(0,out.time,num=out.time) 

            plt.figure(figsize=(10,10))
            plt.plot(t,out.average_betas) 
        
            plt.title("Average Beta over Time") 
            plt.xlabel("time(days)") 
            plt.ylabel("Beta") 
        
            plt.show() 

        case _:
            pass 
