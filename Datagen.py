import numpy as np; 
import pandas as pd; 
import matplotlib.pyplot as plt; 

class DataGenerator:

    state : list
    results: list 
    dailyInfected: list 
    beta: list
    time: list 
    data_name: str
    noise: bool

    def __init__(self,_beta,_gamma,_eta,_initial_state,time_series,data_name,noise=False):

        self.state = []; 
        self.results = []; 
        self.dailyInfected = []; 
        self.beta = []; 


        self.params = [];

        self.params = [_beta,_gamma,_eta];  


        self.state.append(_initial_state); 
        self.time = time_series;
        self.data_name = data_name; 
        self.noise = noise; 
    
        


    def RHS(self,state,params,t):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R = state;

        N = S + I + R;


        new_I = params[0](t)*S*I/N;

        dS = -params[0](t)*(S*I)/N  + params[2] * R;
        dI = params[0](t)*S*I/N-params[1]*I;
        dR = params[1]*I - params[2] * R;

        self.beta.append(params[0](t)); 

        return np.array([dS,dI,dR]),new_I
    
    def propagate_euler(self,state,params,t):
         #define time step of Euler's method

        NperDay = 1
        sol = np.zeros((3, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        dailyInfected = 0;

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([state[0], state[1], state[2]])

        for i in range(len(tSpanFine)-1):

            dt,new_I = self.RHS(sol[:,i],params,t)
            dailyInfected =  new_I/NperDay;
            sol[:,i+1] = sol[:,i] + dt/NperDay;

        return sol[:,-1],dailyInfected;

    def generate_data(self): 

        betas = [];
        self.dailyInfected = [];
        self.results = [];

        self.results.append(self.state[-1]); 

        for t in range(self.time):
            temp,dI = (self.propagate_euler(self.results[-1],self.params,t));
            betas.append(self.params[0](t)); 

            if self.noise == True:
                self.dailyInfected.append(np.random.poisson(dI)); 
            else:
                self.dailyInfected.append(dI);

            
            self.results.append(temp); 


        df = pd.DataFrame(self.dailyInfected); 
        df2 = pd.DataFrame(self.results); 

        df.to_csv(self.data_name + ".csv"); 

        df2.to_csv(self.data_name + "_states.csv"); 

    

    def plot_states(self): 
        plt.plot(self.results); 
        plt.title("Evolution of the State Variables over Time"); 

        plt.xlabel("Time(days)"); 
        plt.ylabel("Population of State"); 
        plt.show();

    def plot_daily_infected(self): 
        plt.plot(self.dailyInfected); 
        plt.title("Number of New Daily Infections over Time"); 

        plt.xlabel("Time(days)"); 
        plt.ylabel("Number of Infections"); 
        plt.show();

    def plot_beta(self): 
        plt.plot(self.beta); 
        plt.xlabel("Time(days)"); 
        plt.ylabel("Value of Beta");
        plt.show(); 




   

