import numpy as np
from enum import Enum




class IPM(Enum):
    SIR = 1
    SIRS = 2
    SIRH = 3

def ipm_attributes(ipm:IPM): 
    match ipm: 
        case IPM.SIR:
            attribs = {"ipm":"SIR","compartments":3,"static_params": 1}
            return attribs
        case IPM.SIRS: 
            attribs = {"ipm":"SIRS","compartments":3,"static_params": 2}  
            return attribs
        case IPM.SIRH: 
            attribs = {"ipm":"SIRH","compartments":4,"static_params": 5}  
            return attribs
        case _: 
            print("IPM is not supported") 




