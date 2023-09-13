from ParticleFilter.Datagen import *


def GenerateSimData(params,initial_state,time_series,data_name="beta_test",noise=True,hospitalization=True):
    dg = DataGenerator(params,initial_state,time_series,data_name,noise,hospitalization,aggregate=1) 

    dg.generate_data() 
    dg.plot_daily_infected() 
    dg.plot_beta() 
    dg.plot_states() 



    
