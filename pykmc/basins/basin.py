from .detection import Detector
from .exploration import Explorer, BasinGenericEventExplorer
from .connectivity import BasinStatesConnectivity
from .selection import FTPASelector
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pykmc import System, Config, NeighborsList, AtomicEnvironment, ReferenceEventTable, PointSetRegistration
from typing import Optional
from ..utils import geometry
import pandas as pd
import copy
import numpy as np
from scipy.spatial import cKDTree

#TODO : After merge do minimization and refinement with manager

@dataclass
class StateData:
    system: Optional[System]
    environment: Optional[AtomicEnvironment]
    neighbors_list: Optional[NeighborsList] 
    transient: bool = False
    visited: bool = False

class BasinsGenericEvents() : 

    def __init__(self, config: Config, reference_table,known_environments, manager ) : 
        self.config = config #Config object with basins parameters
        self.explorer = None #object to explore a state in the basin 
        self.reference_table = reference_table #Object with reference generic events
        self.manager = manager #object to do external task (minimize, refine)

        self.connectivity_table = None #Dataframe of basin connexion state
        self.selected_event = None #The selected event after basin exploration
        self.current_state = None #Current state where we're at 
        self.states_to_explore = None #List of state to explore 
        self.explored_states = None #List of state that we already explored
        self.states: dict[int, StateData] = {}  #Dictionnary of StateDate
        self.known_environments = known_environments 

    def detection(self, params) -> bool : 
        """ 
        """
        return self.detector.detection(**params) 

    def _initialize(self, system) : 
        """ 
        Initialize necessary component after entering in basin
        """
        self.current_state = 0
        self.states_to_explore = [0] 
        self.explored_states = [] 
        self.connectivity_table = BasinStatesConnectivity()
        self.explorer = BasinGenericEventExplorer(config=self.config, reference_table=self.reference_table)
        self.selector = FTPASelector()
        self._add_state(state_index=0, system=system)  #add current state 0 to self.states


    def construct_connexion_table(self) : 
        """ 
        explore the basin and construct the connextion table
        """
        #Loop over state to explore 
        while len(self.states_to_explore) != 0 : 
            #next state to explore : 
            to_explore = self.states_to_explore[0]

            if to_explore not in self.states : #always true except at the start (to_explore = 0) 
                #We need to create the state 
                    #find state to state transition
                    #NOTE : HERE NEED TO CHANGE TO HAVE LIST OF TRANSITION TO NOT SAVE EVERY SYSTEMS POSITIONS 
                from_state, event_idx, central_atom, sym_idx, is_transient = self.connectivity_table.get_transition_to_state(target_state=to_explore)
                    #Create new system 
                new_system = self.system_from_state(from_state, event_idx, central_atom, sym_idx) 

                    #Check if it is a new_system or already in states 
                is_new_state = self.is_new_state(new_system) 
                if is_new_state != -1 : #It already exists 
                    #update table
                    self.connectivity_table.change_state_index(current_index=to_explore, new_index=is_new_state)
                    self.explored_states.append(to_explore)
                    self.states_to_explore.remove(to_explore)
                    continue #Skip the rest

                #add state
                self._add_state(state_index=to_explore, system=new_system, transient=is_transient)

                #Check if unknown atomic environments
                if self.is_states_has_unknown_environments(self.states[to_explore]) : 
                    #We consider that this state is an absorbing one because we need to search new events (in main KMC loop) 
                    #Need to update the connectivity table 
                    self.connectivity_table.change_state_to_absorbing(to_explore) 
                    self.states[to_explore].transient = False
                    is_transient = False
                
                if not is_transient : 
                    self.states_to_explore.remove(to_explore)
                    self.explored_states.append(to_explore)
                    continue #We dont explore/skip the rest


            #Explore state 
            self.current_state = to_explore
            print(self.current_state, to_explore)
            print(self.states)
            last_state_connectivity = self.get_last_state_index()
            self.explorer.explore(state=self.states[to_explore], state_index=self.current_state, start_index=last_state_connectivity)
            
            #to_explore has been explored : 
            self.states_to_explore.remove(to_explore)
            self.explored_states.append(to_explore)

            #Merge state connectivity table to basin connectivity table 
            self.connectivity_table.merge(self.explorer.connectivity_table)
            #Clrean explorer connectivity table
            self.explorer.clear()
            self.update_to_explore()



            
    def run(self, system) : 
        """ 
        run the basin exploration and select an event
        """
        #initialize the basin
        self._initialize_basins(system)
        #explore the basin
        self.construct_connexion_table()
        #reorder states index 
        mapping = self.connectivity_table.reorder_states_index()
        self.states = {mapping[old]: val for old, val in self.basin.states.items()}


    def select_event(self) : 
        """ 
        select an event base on the selector algorithm
        """
        pass

    def get_seletec_event(self) : 
        """ 
        convinient method
        """
        pass

    def get_last_state_index(self) : 
        if self.current_state == 0 : #connextion table is empty
            new_state_connexion = 1 
        else : #last state connexion +1
            new_state_connexion = int(self.connectivity_table.get_table().tail(1)['state_connexion']+1)
        return new_state_connexion
    
    def update_to_explore(self) : 
        #Find all state index in the connexion table : 
        unique_states = set(self.connectivity_table.get_table()['state']).union(set(self.connectivity_table.get_table()['state_connexion']))
        self.states_to_explore =  list(unique_states.difference(set(self.explored_states)))


    def system_from_state(self, from_state, event_idx, central_atom, sym_idx) : 
        """ Move to state index
        """

        ref_event = self.reference_table.table.iloc[event_idx].copy()

        #Apply the generic event to the current state 

        #TODO : When merged with lammps manager minimize from the SADDLE POINT and check
        new_system = copy.deepcopy(self.states[from_state].system)

        psr_output = PointSetRegistration(self.config, new_system, ref_event , self.states[from_state].neighbors_list, central_atom).match()
        if psr_output.is_ok():
            psr_output = psr_output.ok_value()

        else : 
            raise ValueError("Basin: PSR failed")

        # Check if PointSetRegistration finds a match
        if psr_output.matching_score < self.config.psr.matching_score_thr :

            # Apply PSR to generic event
            final_positions = ref_event['final_positions']
            
            # Apply symmetry matrix if sym != 0
            if sym_idx != 0 :
                sym_matrices = ref_event['sym_matrix']
                sym_matrix = sym_matrices[sym_idx]
                final_positions = geometry.transform_positions(final_positions, sym_matrix,0, ref_event["sym_perm"][sym_idx])
            final_positions = geometry.transform_positions(final_positions, psr_output.rotation_matrix, psr_output.translation_matrix, psr_output.permutation_matrix)

            

            # Move system do final positions
            neighbors = self.states[from_state].neighbors_list.get_neighbors('rcut', central_atom)

            ####################
            ####################
            ####################
            new_system.update_positions(final_positions, atom_idx = neighbors)
            future = self.manager.minimize_with_results(self.config, positions=new_system.positions)
            min_pos, _ = future.result()
            new_system.update_positions(min_pos)


            return new_system

        else : 
            raise ValueError("Basin: PSR matching score > matching score threshold")


    def is_new_state(self, system) : 
        #Loop over all other system in self.states to see if system is already known 

        for state_index, state_data in self.states.items():
            are_equivalent = self.are_structures_equivalent(system.positions, state_data.system.positions, cell = system.cell) 
            if are_equivalent : 
                return state_index
        return -1 


    def are_structures_equivalent(self, pos1, pos2, cell, tol=0.1):

        if len(pos1) != len(pos2):
            return False

        box = np.diag(cell).tolist()
        tree2 = cKDTree(pos2, boxsize=box)
        distances, _ = tree2.query(pos1, k=1)

        return np.max(distances) < tol

    def is_states_has_unknown_environments(self, state: StateData) : 
        if set(state.environment.atomic_environment_list).difference(self.known_environments) != set() :
            return True 
        else : 
            return False

    def _add_state(self, state_index, system=None, transient=True, applicable_events=None, visited=False ) :
        """Add a new state in the `self.states` dictionnary."""
        #to fit typing 
        neighbors_list  = []
        atomic_environment = []

        if system is not None : 
            neighbors_list = NeighborsList(system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut)  
            atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, neighbors_list.neighbors_list['rnei'], neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add)
            
        new_state =  StateData(system=system, environment=atomic_environment, neighbors_list=neighbors_list, transient=transient,  visited=visited)

        self.states[state_index]= new_state