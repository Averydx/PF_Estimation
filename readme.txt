                                                                    ######### Particle Filter ##########


How to call the filter from the command line: 

Enter python virtual environment and install requirements.txt (using vscode or similar)


Required Arguments: 

--population 
The population of the data set you are estimating over i.e the population of Arizona 





Optional Arguments:

--simulate_data 
true to simulate a data set for testing, the data will then automatically be passed to the algorithm

if you do not want to simulate_data, don't include this argument

--file

specify the file which contains the time series data to estimate over
e.g. ./data_sets/FLU_HOSPITALIZATIONS.csv

if --simulate_data is enabled --file is not used, you can pass a file it just won't do anything

--intial seed 

A value between 0 and 1 that represents the proportion of initial infected out of the total population, the intial infected will be drawn from a uniform distribution  
uniform[0,population * initial_seed]

--particles 
Specifies the number of particles with which to run the simulation, defaults to 10000 but any integer number is valid, performance may vary 

--iterations 
Specify if you want to run the algorithm for less than the length of your data set
i.e. in a forecasting application


Example of execution after entering venv with simulated data:

python main.py --iterations 500 --population 100000 --particles 50000 --simulate_data true --initial_seed 0.1





Example of execution after entering venv with real data: 

python main.py --iterations 223 --population 100000 --particles 50000 --initial_seed 0.1 --file ./data_sets/FLU_HOSPITALIZATIONS.csv






