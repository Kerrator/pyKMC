import pandas as pd
from .rate_constant import *
import numpy as np 


class Catalog : 

    def __init__(self, config) : 
        self.config = config
        if self.config['Control']['catalog'] is None : 
            self._initialize_catalog() 
        else : 
            self.catalog = pd.read_pickle(self.config['Control']['catalog'])


    def add_event(self, min1positions, saddlepositions, min2positions, move_atom_idx, dE_forward, dE_backward, neighbors_list_environment) : 
        """ 
        """
        #Energy bounds 
        emin = self.config['EventSearch']['emin_event']
        emax = self.config['EventSearch']['emax_event']
        #Get environment of move_atom_idx 
        neighbors = neighbors_list_environment[move_atom_idx]+[int(move_atom_idx)]

        if self.config['Control']['reconstruction'] : 
            self._add_event_with_reconstruction(min1positions[neighbors], saddlepositions[neighbors], min2positions[neighbors], move_atom_idx, dE_forward, dE_backward)
        else : 
            if emin < dE_forward < emax : 
                is_new = self._add_event_no_reconstruction(min2positions[neighbors], move_atom_idx, dE_forward)
                in_e_bounds = True 
            else : 
                is_new = True
                in_e_bounds = False  
            return is_new, in_e_bounds


    def _add_event_with_reconstruction(self, min1positions, saddlepositions, min2positions )  :
        pass
    def _add_event_no_reconstruction(self, final_positions, move_atom_idx, dE) :



        dfevent = pd.Series({'atom_index' : move_atom_idx, 
                            'final_positions' : final_positions, 
                            'energy_barrier' : dE,
                            'k' : compute_rate_Eyring(dE, self.config)})  

        if len(self.catalog) > 0 : 
            #Check if event alread in catalog : 
            atol = 1e-3 
            rtol = 1e-3 

            #Only select rows with same atom index 
            subset = self.catalog[self.catalog["atom_index"] == dfevent['atom_index']]

            #Check if we have final positions of the event close to at least one final positions in the subset 
            if not subset["final_positions"].apply(lambda pos : np.allclose(pos, dfevent["final_positions"], atol=atol, rtol=rtol)).any() : 
                #if not add event to the catalog : 
                self.catalog = pd.concat([self.catalog, dfevent.to_frame().T], ignore_index=True)
                return True 
            else :
                return False
            
        else : 
            self.catalog = pd.concat([self.catalog, dfevent.to_frame().T], ignore_index=True)
            return True

    def _initialize_catalog(self) : 
        if self.config['Control']['reconstruction'] : 
            self.catalog = pd.DataFrame(columns=['event_id', 
                                                 'initial_positions', 
                                                 'saddle_positions', 
                                                 'final_positions', 
                                                 'energy_barrier', 
                                                 'k', 
                                                 'id_saddle',
                                                 'id_final', 
                                                 'move_atom_idx'])
        else : 
            self.catalog = pd.DataFrame(columns = ['atom_index', 
                                                   'final_positions',
                                                   'energy_barrier',
                                                   'k'])
            
    def save(self, outfile='catalog.pickle') : 
        self.catalog.to_pickle(outfile)