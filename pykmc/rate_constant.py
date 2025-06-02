from .config import PhysicalConstants
import math as m 

def compute_rate_Eyring(dE, config) : 
    p = PhysicalConstants() 
    T = config.rateconstant.T
    k0 = config.rateconstant.k0
    return k0*((p.kb*T)/p.h)*m.exp(-dE/(p.kb*T))

def compute_htst(dE, engine) : 
    pass
