import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from numpy.typing import NDArray

class DataGenerator:

    state : list
    results: list 
    dailyInfected: list 
    beta: list
    time: list 
    data_name: str
    noise: bool
    params: dict
    hospitalization: bool
    variance: float

    def __init__(self,params_dict,_initial_state,time_series,data_name,noise=False,hospitalization=False,variance = 0.999):

        self.state = [] 
        self.results = [] 
        self.dailyInfected = [] 
        self.beta = [] 


        self.params = params_dict

        self.state.append(_initial_state) 
        self.time = time_series
        self.data_name = data_name 
        self.noise = noise 
        self.hospitalization = hospitalization 
        self.variance = variance 
    
        


    def RHS(self,state:NDArray,params:dict,t):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R = state

        N = S + I + R


        new_I = params["beta"](t)*S*I/N

        dS = -params["beta"](t)*(S*I)/N  + params["eta"] * R
        dI = params["beta"](t)*S*I/N-params["gamma"]*I
        dR = params["gamma"]*I - params["eta"] * R

        self.beta.append(params["beta"](t)) 

        return np.array([dS,dI,dR]),new_I
    
    def propagate_euler(self,state,params,t):
         #define time step of Euler's method

        NperDay = 1
        sol = np.zeros((3, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        dailyInfected = 0

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([state[0], state[1], state[2]])

        for i in range(len(tSpanFine)-1):

            dt,new_I = self.RHS(sol[:,i],params,t)
            dailyInfected =  new_I/NperDay
            sol[:,i+1] = sol[:,i] + dt/NperDay

        return sol[:,-1],dailyInfected

    def RHS_H(self,state:list,params:dict,t:int):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = state
        N = S + I + R + H 

        new_H = ((1/params["D"])*params["gamma"]) * I   

        dS = -params["beta"](t)*(S*I)/N + (1/params["L"])*R 
        dI = params["beta"](t)*S*I/N-(1/params["D"])*I
        dR = (1/params["hosp"]) * H + ((1/params["D"])*(1-params["gamma"])*I)-(1/params["L"])*R 
        dH = ((1/params["D"])*params["gamma"]) * I - (1/params["hosp"]) * H 

        self.beta.append(params["beta"](t)) 

        return np.array([dS,dI,dR,dH]),new_H

    def propagate_euler_H(self,state:list,params:dict,t:int):
         #define time step of Euler's method

        dailyHospitalized = 0

        NperDay = 1
        sol = np.zeros((4, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([state[0], state[1], state[2],state[3]])

        for i in range(len(tSpanFine)-1):

            dt,new_H = self.RHS_H(sol[:,i],params,t)
            dailyHospitalized =  new_H/NperDay
            sol[:,i+1] = sol[:,i] + dt/NperDay
        
        return sol[:,-1],dailyHospitalized





    def generate_data(self): 

        betas = []
        self.dailyInfected = []
        self.results = []

        self.results.append(self.state[-1]) 

        for t in range(self.time):
            if(not self.hospitalization):
                temp,dI = (self.propagate_euler(self.results[-1],self.params,t))
            
            else: 
               temp,dI = (self.propagate_euler_H(self.results[-1],self.params,t)) 
            
            betas.append(self.params["beta"](t)) 

            if self.noise:
                self.dailyInfected.append(np.random.poisson(dI)) 
            else:
                self.dailyInfected.append(dI)

            
            self.results.append(temp) 


        df = pd.DataFrame(self.dailyInfected) 
        df2 = pd.DataFrame(self.results) 

        df.to_csv("./data_sets/" + self.data_name + ".csv") 

        df2.to_csv("./data_sets/" + self.data_name + "_states.csv") 

    def plot_states(self): 
        plt.yscale('log')
        plt.plot(self.results) 
        plt.title("Evolution of the State Variables over Time") 

        plt.xlabel("Time(days)") 
        plt.ylabel("Population of State") 
        plt.show()

    def plot_daily_infected(self): 
        if(not self.hospitalization): 
            plt.plot(self.dailyInfected) 
            plt.title("Number of New Daily Infections over Time") 

            plt.xlabel("Time(days)") 
            plt.ylabel("Number of Infections") 
            plt.show()

        else: 
            plt.plot(self.dailyInfected) 
            plt.title("Number of New Daily Hospitalizations over Time") 

            plt.xlabel("Time(days)") 
            plt.ylabel("Number of Hospitalizations") 
            plt.show()

    def plot_beta(self): 
        plt.plot(self.beta) 
        plt.xlabel("Time(days)") 
        plt.ylabel("Value of Beta")
        plt.show() 




   

