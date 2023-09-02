import numpy as np
from matplotlib import cm,ticker
import matplotlib.pyplot as plt
from ObjectHierarchy.utilities.Output import Output



def plot(out:Output,graph:int): 
    match graph:
        case 0:
            t = np.linspace(0, out.time_series, num=out.time_series)
            fig, axs = plt.subplots(2,2,figsize=(10, 10))
            fig.suptitle('Particle Filter output')
            axs[0,0].set_title("Simulated Observations")
            axs[0,0].set_xlabel('Time (Days)')
            axs[0,0].set_ylabel('Number of Hospitalizations')
            axs[0,1].set_title("Beta")
            axs[0,1].set_xlabel("Time (Days)")
            colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore
            axs[0,0].set_ylim(0, 300)
            axs[0,0].plot(t, out.observation_data[:out.time_series], color="black", zorder=12,linewidth =0.8)
            for i in range(11):
                axs[0,0].fill_between(t, out.observation_qtls[i, :], out.observation_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            axs[0,1].set_ylim(0, 1)
            for i in range(11):
                axs[0,1].fill_between(t, out.beta_qtls[i, :], out.beta_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            # # To show all the y-ticks
            axs[0,0].yaxis.set_major_locator(ticker.LinearLocator())
            axs[0,1].yaxis.set_major_locator(ticker.LinearLocator())
            plt.tight_layout()
            plt.show()

        case _:
            pass 




        