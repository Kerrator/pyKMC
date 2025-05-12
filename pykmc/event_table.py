from abc import ABC, abstractmethod
import pandas as pd
from .rate_constant import *
import numpy as np 
from ase import Atoms
from .environments.graph_nauty import graph
from .system import System
from .neighbors_list import NeighborsList
from .symmetries import unique_symmetries
import sys



class CatalogRef : 

    def __init__(self, config) : 
        self.config = config
        if self.config['Control']['catalog'] is None : 
            self._initialize_catalog() 
        else : 
            self.catalog = pd.read_pickle(self.config['Control']['catalog'])


    def add_event(self, min1positions, saddlepositions, min2positions, move_atom_idx, dE_forward, dE_backward, neighbors_list_environment, cell) : 
        """ 
        """
        #Energy bounds 
        emin = self.config['EventSearch']['emin_event']
        emax = self.config['EventSearch']['emax_event']

        if self.config['Control']['reconstruction'] :
            if emin < dE_forward < emax : 
                is_new = self._add_event_with_reconstruction(min1positions, saddlepositions, min2positions, move_atom_idx, dE_forward, dE_backward, cell)
                in_e_bounds = True 
            else : 
                is_new = True 
                in_e_bounds = False
        else : 
            if emin < dE_forward < emax : 
                #Get environment of move_atom_idx 
                neighbors = neighbors_list_environment[move_atom_idx]
                is_new = self._add_event_no_reconstruction(min2positions[neighbors], move_atom_idx, dE_forward)
                in_e_bounds = True 
            else : 
                is_new = True
                in_e_bounds = False  
        return is_new, in_e_bounds


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
        

    def _add_event_with_reconstruction(self, min1positions, saddlepositions, min2positions, move_atom_idx, dE_forward, dE_backward, cell)  :
        dfevent_forward, dfevent_backward = self._event_series_with_reconstruction(min1positions, saddlepositions, min2positions, move_atom_idx, dE_forward, dE_backward, cell) 
        #Only select rows with same event_id as dfenvent : 
        subset = self.catalog[self.catalog["event_id"] == dfevent_forward["event_id"]] 
        #subset of subset with rows with the same saddle_id : 
        subset = subset[subset["id_saddle"] == dfevent_forward["id_saddle"]]
        #subset of subset of subset with rows with the same final_id : 
        subset = subset[subset["id_final"] == dfevent_forward["id_final"]]
        #if there is no event with same IDs
        if len(subset) == 0 : 
            #add to the catalog foward reaction  
            self.catalog = pd.concat([self.catalog, dfevent_forward.to_frame().T], ignore_index=True)
            #Check if backward reaction is not the same as the forward one    
            if dfevent_forward["event_id"] != dfevent_forward["id_final"] :  
                self.catalog = pd.concat([self.catalog, dfevent_backward.to_frame().T], ignore_index=True)
            return True
        else : 
            return False
            
    def _event_series_with_reconstruction(self, min1_positions, saddle_positions, min2_positions, index_move, dE_forward, dE_backward, cell) : 
        """
        """
        #compute neighbors list for initial, saddle and final positions -> to compute graphs 
        min1system = System() 
        min1system.positions = min1_positions 
        min1system.cell = cell
        min1neighbors_list = NeighborsList(min1system, self.config) 

        saddlesystem = System() 
        saddlesystem.positions = saddle_positions
        saddlesystem.cell = cell
        saddleneighbors_list = NeighborsList(saddlesystem, self.config) 
        
        min2system = System() 
        min2system.positions = min2_positions 
        min2system.cell = cell
        min2neighbors_list = NeighborsList(min2system, self.config) 
        

        #Compute all needed topology ID : 
        id_min1 = graph(min1neighbors_list.neighbors_list['rnei'], min1neighbors_list.neighbors_list['rcut'], atom_idx=[index_move])[0]
        id_saddle = graph(saddleneighbors_list.neighbors_list['rnei'], saddleneighbors_list.neighbors_list['rcut'], atom_idx=[index_move])[0]
        id_min2 = graph(min2neighbors_list.neighbors_list['rnei'], min2neighbors_list.neighbors_list['rcut'], atom_idx=[index_move])[0]
        
        neighbor_list_forwward = min1neighbors_list.neighbors_list['rcut'][index_move]
        neighbor_list_backward = min2neighbors_list.neighbors_list['rcut'][index_move]

        #Symmetries : 
        sym_matrix, sym_perm = unique_symmetries(min1_positions[neighbor_list_forwward],min2_positions[neighbor_list_forwward], self.config['EventSearch']['sym_thr'])
        dfevent_forward = pd.Series({'event_id' : id_min1 , 
                                     'initial_positions' : min1_positions[neighbor_list_forwward], 
                                     'saddle_positions': saddle_positions[neighbor_list_forwward], 
                                     'final_positions': min2_positions[neighbor_list_forwward], 
                                     'energy_barrier': dE_forward, 
                                     'k' : compute_rate_Eyring(dE_forward, self.config), 
                                     'id_saddle' : id_saddle, 
                                     'id_final': id_min2, 
                                     'move_atom_idx': np.where(neighbor_list_forwward == index_move)[0][0], 
                                     'sym_matrix' : sym_matrix, 
                                     'sym_perm' : sym_perm})

        sym_matrix, sym_perm = unique_symmetries(min2_positions[neighbor_list_backward], min1_positions[neighbor_list_backward], self.config['EventSearch']['sym_thr'])
        dfevent_backward = pd.Series({'event_id' : id_min2 , 
                                     'initial_positions' : min2_positions[neighbor_list_backward], 
                                     'saddle_positions': saddle_positions[neighbor_list_backward], 
                                     'final_positions': min1_positions[neighbor_list_backward], 
                                     'energy_barrier': dE_backward, 
                                     'k' : compute_rate_Eyring(dE_backward, self.config), 
                                     'id_saddle' : id_saddle, 
                                     'id_final': id_min1, 
                                     'move_atom_idx': np.where(neighbor_list_backward == index_move)[0][0],
                                     'sym_matrix' : sym_matrix, 
                                     'sym_perm': sym_perm })
        
        return dfevent_forward, dfevent_backward

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
                                                 'move_atom_idx', 
                                                 'sym_matrix', 
                                                 'sym_perm'])
        else : 
            self.catalog = pd.DataFrame(columns = ['atom_index', 
                                                   'final_positions',
                                                   'energy_barrier',
                                                   'k'])
            
    def save(self, outfile='catalog.pickle') : 
        self.catalog.to_pickle(outfile)



class ActiveEventTable() : 

    def __init__(self, event_dataframe = None ) : 
        if event_dataframe is not None : 
            self.active_events = event_dataframe
        else : 
            self.active_events = pd.DataFrame(columns = ['atom_index', 'final_positions', 'energy_barrier', 'k'])

    def add_event(self, dfevent) : 
        self.active_events = pd.concat([self.active_events, dfevent.to_frame().T], ignore_index=True)







