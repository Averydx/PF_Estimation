import numpy as np
from matplotlib import cm,ticker
import matplotlib.pyplot as plt
from ObjectHierarchy.Output import Output



def plot(out:Output,graph:int): 
    match graph:
        case 0:
            t = np.linspace(0, out.time_series, num=out.time_series)
            fig, (ax1, ax2) = plt.subplots(1, 2)
            fig.suptitle('Particle Filter output')
            ax1.set_title("Confidence Intervals")
            ax2.set_title("Average Beta over time")
            colors = cm.plasma(np.linspace(0, 1, 12))
            ax1.set_ylim(0, 300)
            ax1.plot(t, out.observation_data[:out.time_series], color="blue", zorder=12)
            for i in range(11):
                ax1.fill_between(t, out.observation_qtls[i, :], out.observation_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            ax2.set_ylim(0, 1)
            for i in range(11):
                ax2.fill_between(t, out.beta_qtls[i, :], out.beta_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            # To show all the y-ticks
            ax1.yaxis.set_major_locator(ticker.LinearLocator())
            ax2.yaxis.set_major_locator(ticker.LinearLocator())
            plt.tight_layout()
            plt.show()

        case _:
            pass 




        