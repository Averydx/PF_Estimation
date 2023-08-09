#The numerical propagator to go from time t to t+1

import numpy as np; 
from numpy.typing import NDArray; 


class one_step_propagator: 

    state: NDArray[np.float_]; 
    params: NDArray[np.float_]; 

    def __init__(self): 
        self.state = None;
        self.params = None; 

    def RHS(self,state,params):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = state;
        N = S + I + R + H; 

        beta,L,D,gamma,hosp = params; 

        dS = -beta*(S*I)/N + (1/L)*R; 

        dI = beta*S*I/N-(1/D)*I;

        dR = (1/hosp) * H + ((1/D)*(1-gamma)*I)-(1/L)*R; 

        dH = ((1/D)*gamma) * I - (1/hosp) * H; 

        return np.array([dS,dI,dR,dH])

    def propagate_euler(self):
         #define time step of Euler's method

        NperDay = 1
        sol = np.zeros((4, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([self.state[0], self.state[1], self.state[2],self.state[3]])

        for i in range(len(tSpanFine)-1):

            dt = self.RHS(sol[:,i],self.params)
            sol[:,i+1] = sol[:,i] + dt/NperDay;
        
        return sol[:,-1];


    