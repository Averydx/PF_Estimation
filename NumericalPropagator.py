#The numerical propagator to go from time t to t+1

import numpy as np; 


class one_step_propagator: 
    def __init__(self,_state): 
        self.state = None;
        self.params = None; 

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




        







