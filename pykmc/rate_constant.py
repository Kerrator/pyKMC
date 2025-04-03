from .config import Parameters
import math as m 

def compute_rate_Eyring(dE, config) : 
    p = Parameters() 
    T = config['EventSearch']['T'] 
    k0 = config['EventSearch']['k0'] 
    return k0*((p.kb*T)/p.h)*m.exp(-dE/(p.kb*T))

def compute_htst(dE, engine) : 
    pass
