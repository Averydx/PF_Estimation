#The numerical propagator to go from time t to t+1

import numpy as np 


class one_step_propagator: 
    state: list
    estimated_params:dict
    static_params:dict

    def __init__(self): 
        self.state = list()
        self.estimated_params = None
        self.static_params = None

    def RHS(self,state,):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R = state

        N = S + I + R 

        new_I = self.estimated_params['beta']*S*I/N

        dS = -self.estimated_params['beta']*(S*I)/N + self.static_params['eta'] * R
        dI = self.estimated_params['beta']*S*I/N-self.static_params['gamma']*I
        dR = self.static_params['gamma']*I - self.static_params['eta'] * R 



        return np.array([dS,dI,dR]),new_I

    def propagate_euler(self):
         #define time step of Euler's method

        NperDay = 1
        sol = np.zeros((3, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        dailyInfected = 0

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([self.state[0], self.state[1], self.state[2]])

        for i in range(len(tSpanFine)-1):

            dt,new_I = self.RHS(sol[:,i])
            dailyInfected =  new_I/NperDay
            sol[:,i+1] = sol[:,i] + dt/NperDay
        
        return sol[:,-1],dailyInfected
    
    def propagate_euler_H(self):
         #define time step of Euler's method

        dailyHospitalized = 0

        NperDay = 1
        sol = np.zeros((4, NperDay+1))
        tSpanFine = np.linspace(0, 1, NperDay+1)

        #initiate the values of SIR compartments at the first time point
        sol[:,0] = np.array([self.state[0], self.state[1], self.state[2],self.state[3]])

        for i in range(len(tSpanFine)-1):
            
            dt,new_H = self.RHS_H(sol[:,i])
            dailyHospitalized =  new_H/NperDay
            sol[:,i+1] = sol[:,i] + dt/NperDay
        
        return sol[:,-1],dailyHospitalized
    

    def RHS_H(self,state):
    #params has all the parameters – beta, gamma
    #state is a numpy array

        S,I,R,H = state
        N = S + I + R + H 

        new_H = ((1/self.static_params['D'])*self.static_params['gamma']) * I   

        dS = -self.estimated_params['beta']*(S*I)/N + (1/self.static_params['L'])*R 
        dI = self.estimated_params['beta']*S*I/N-(1/self.static_params['D'])*I
        dR = (1/self.static_params['hosp']) * H + ((1/self.static_params['D'])*(1-(self.static_params['gamma'] if 'gamma' in self.static_params else self.estimated_params['gamma']))*I)-(1/self.static_params['L'])*R 
        dH = (1/self.static_params['D'])*(self.static_params['gamma'] if 'gamma' in self.static_params else self.estimated_params['gamma']) * I - (1/self.static_params['hosp']) * H 
        
        return np.array([dS,dI,dR,dH]),new_H

    
