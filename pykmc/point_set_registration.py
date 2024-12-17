import ira_mod 
import random
import numpy as np
import pandas as pd 

class PointSetRegistration() : 
    """ 
    """ 
    def __init__(self,  system, psr_style, dimension, nprocs, backend) : 
        """ 
         
        """
        self.psr_style = psr_style
        self.system = system
        self.dimension = dimension
        self.backend = backend
        self.nprocs = nprocs 

    def run(self) : 
        """ 
        """ 
        if self.psr_style == 'ira' : 
            self.ira(0)
        else : 
            print(ERROR)
    
    def ira(self, idx_cat) : 
        """ 
        Use IRA to extract rotation, translation, permutation matrix to apply on generic event
        idx_cat : index in catalog of the selected event 
        """ 
        ira = ira_mod.IRA() 

        #Event informations : 
        id = self.system.catalog.loc[idx_cat].at['event_id']
        coords2 = self.system.catalog.loc[idx_cat].at["initial_positions"] 
        nat2 = len(coords2)
        typ2 = nat2*['Ni']

        #system structure : 
            #find in environment one atom with the event ID 
        #TODO better
        for dic in self.system.environment : 
            if dic['ID'] == id : 
                atom_index_list = dic['atom index']
        #random atom : 
        atom_index = random.choice(atom_index_list)
        rcutevent = 8.0
        ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
        dist = self.system.get_distances(atom_index, ind, mic=True)
        neighbor_list = np.where(dist<rcutevent)[0]

        coords1 = self.system.get_positions()[neighbor_list] 
        nat1 = len(coords1)
        typ1 = typ2

        kmax_factor = 2.0
        rmat, tr, perm, dh = ira.match( nat1, typ1, coords1, nat2, typ2, coords2, kmax_factor )

        a = [[rmat, tr, perm, dh, atom_index, idx_cat]]
        results = pd.DataFrame(a, columns=['R', 
                                        'T', 
                                        'P', 
                                        'dh', 'atom index', 'n event'])
        results.to_pickle('psr_event_'+str(idx_cat)+'.pickle')



