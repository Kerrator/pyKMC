import numpy as np


class FTPASelector(): 
    """
    """

    def __init__(self) : 
        self.M_abs = None #Absorbing Markov chain transition matrix

    def build_absorbing_matrix_from_connectivity(self, connectivity_table) : 
        """"""
    
        #Build empty Absorbin markoc chain transition matrix
        n_states = max(set(self.df["state"]) | set(self.df["state_connexion"]))
        self.M_abs = np.zeros((n_states, n_states))

        
