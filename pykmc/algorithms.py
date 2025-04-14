import random 
import numpy as np
import math as m

def rejection_free(l_k) : 
    """ 
    l_k : list array possible event constant rate 
    """ 
    
    #compute cumulative rate constant
    k_cumulative = [np.sum(l_k[:i]) for i in range(1,len(l_k)+1)] 
    #get random number [0,1[ 
    rand1 = random.random()
    #find event index satisfy ki-1<rand1ktot<ki 
    idx_selected_event = np.searchsorted(k_cumulative, rand1*k_cumulative[-1], side='left') 
    #compute associated delta t : 
    delta_t = -m.log(random.random())/k_cumulative[-1] 
    return idx_selected_event, delta_t
