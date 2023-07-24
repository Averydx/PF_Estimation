#The numerical propagator to go from time t to t+1

import numpy as np; 


class one_step_propagator: 
    def __init__(self,_state): 
        self.state = None;
        self.params = None; 

    def RHS(self,state,params):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R = state; 

        print(state); 

        dS = -params[0]*S*I
        dI = params[0]*S*I-params[1]*I
        dR = params[1]*I

        return np.array([dS,dI,dR])
    
    def propagate_euler(self): 
         #define time step of Euler's method

        NperDay = 10
        sol = np.zeros((3, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([self.state[0], self.state[1], self.state[2]])

        for i in range(len(tSpanFine)-1):
            sol[:,i+1] = sol[:,i] + self.RHS(sol[:,i], self.params)/NperDay; 
            
        return sol[:,-1]



    def propagate(self): 
        beta = self.params[0]; 
        gamma = self.params[1]; 

        state_copy = np.copy(self.state); 
        S = state_copy[0]; 
        I = state_copy[1]; 
        R = state_copy[2]; 

         #accumulators
        tInf = 0;
        tRem = 0;

        if S != 0:
            for i in range(0,S):
                inf = np.random.binomial(1,beta);
                tInf += inf;

        if I != 0:
            for i in range(0,I):
                rem = np.random.binomial(1,gamma);

                tRem += rem;

        self.state = [S - tInf,I + tInf - tRem,R + tRem];
        return [self.state,tInf]; 




        







