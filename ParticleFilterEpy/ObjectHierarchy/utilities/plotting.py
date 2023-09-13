import numpy as np
from matplotlib import cm,ticker
import matplotlib.pyplot as plt
from ObjectHierarchy.utilities.Output import Output



def plot(out:Output,graph:int): 
    match graph:
        case 0:

            '''Setup for all grids '''
            t = np.linspace(0, out.time_series, num=out.time_series)
            fig, axs = plt.subplots(2,2,figsize=(10, 10))
            fig.suptitle('Particle Filter output')


            axs[0,0].set_title("Simulated Observations")
            axs[0,0].set_xlabel('Time (Days)')
            axs[0,0].set_ylabel('Number of Hospitalizations')


            axs[0,1].set_title("Quantiles of Beta")
            axs[0,1].set_xlabel("Time (Days)")
            axs[0,1].set_ylabel("Beta")

            axs[1,1].set_title("Average Beta")
            axs[1,1].set_xlabel("Time (Days)")
            axs[1,1].set_ylabel("Beta")

            axs[1,0].set_title("FFT of Beta")
            axs[1,0].set_xlabel("Time (Days)")
            axs[1,0].set_ylabel("FFT'd Beta")

            '''Plotting for the quantiles'''
            colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore
            axs[0,0].plot(t, out.observation_data[:out.time_series], color="black", zorder=12,linewidth =0.8)
            for i in range(11):
                axs[0,0].fill_between(t, out.observation_qtls[i, :], out.observation_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            axs[0,1].set_ylim(0, 1)
            for i in range(11):
                axs[0,1].fill_between(t, out.beta_qtls[i, :], out.beta_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)

            '''Plotting for single lines'''
            axs[1,1].plot(t,out.average_beta)

            '''FFT calculations and plotting'''
            f = out.average_beta.squeeze()

            t = len(out.average_beta)
            f_hat = np.fft.fft(f,t)
            PSD = np.abs(f_hat/t)
            freq = 1/t  * np.arange(t)

            indices = PSD>0.02

            f_hat = indices * f_hat
            f_clean = np.fft.ifft(f_hat)

            axs[1,0].plot(np.arange(t),f_clean.real)      

            '''To show all the y-ticks'''
            axs[0,0].yaxis.set_major_locator(ticker.LinearLocator())
            axs[0,1].yaxis.set_major_locator(ticker.LinearLocator())
            axs[1,1].yaxis.set_major_locator(ticker.LinearLocator())
            axs[1,0].yaxis.set_major_locator(ticker.LinearLocator())
            plt.tight_layout()
            plt.show()

        case 1:
            '''Setup for all grids '''
            t = np.linspace(0, out.time_series, num=out.time_series)
            fig, axs = plt.subplots(2,2,figsize=(10, 10))
            fig.suptitle('Particle Filter output')


            axs[0,0].set_title("Simulated Observations")
            axs[0,0].set_xlabel('Time (Days)')
            axs[0,0].set_ylabel('Number of Hospitalizations')


            axs[0,1].set_title("Quantiles of Beta")
            axs[0,1].set_xlabel("Time (Days)")
            axs[0,1].set_ylabel("Beta")

            axs[1,1].set_title("Average Beta")
            axs[1,1].set_xlabel("Time (Days)")
            axs[1,1].set_ylabel("Beta")

            axs[1,0].set_title("Average State")
            axs[1,0].set_xlabel("Time (Days)")
            axs[1,0].set_ylabel("# in State")

            '''Plotting for the quantiles'''
            colors = cm.plasma(np.linspace(0, 1, 12)) # type: ignore
            axs[0,0].plot(t, out.observation_data[:out.time_series], color="black", zorder=12,linewidth =0.8)
            for i in range(11):
                axs[0,0].fill_between(t, out.observation_qtls[i, :], out.observation_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)
            axs[0,1].set_ylim(0, 1)
            for i in range(11):
                axs[0,1].fill_between(t, out.beta_qtls[i, :], out.beta_qtls[22-i, :], facecolor=colors[11 - i], zorder=i)

            '''Plotting for single lines'''
            axs[1,1].plot(t,out.average_beta)

            '''average state plotting'''
            axs[1,0].plot(out.average_state)
            axs[1,0].set_yscale('log')

            '''To show all the y-ticks'''
            axs[0,0].yaxis.set_major_locator(ticker.LinearLocator())
            axs[0,1].yaxis.set_major_locator(ticker.LinearLocator())
            axs[1,1].yaxis.set_major_locator(ticker.LinearLocator())
            plt.tight_layout()
            plt.show()


        case _:
            pass 




        