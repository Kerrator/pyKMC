from pykmc import NeighborsList, AtomicEnvironment, ActiveEventTable
import random 
from .result import EventSearchOutput, KMCLoopInfo, ErrorInfo, ErrorType, Result, AtomicEnvironmentInfo, ReferenceEventSearchInfo, ReferenceValidEventsInfo, RefinementsInfo
import numpy as np
from ase.io import write
from ase import Atoms
from .algorithms import * 
import sys
import pandas as pd
import pickle
from .initializer import Initializer
from .info_simulation import info_atomic_environments, info_reference_event_searches, info_is_valid_reference_events, info_refinements
from .eventsearch import EventSearch
from .refinement import Refinement
from .log import Colors


#TODO fix reconstruction = False
#NOTE can maybe reimplment tries if empty catalog
#TODO add select histo refinement

class KMC() : 

    def __init__(self, config) : 
        self.config = config 
        self.loggers = None
        self.system = None 
        self.engine = None 
        self.neighbors_list = None 
        self.atomic_environment = None 
        self.reference_table = None
        self.visited_environments = None 
        self.total_energy = None
            
    def run(self) : 
        
        #Initialize the simulation, KMC attributes and minimize the system
        self._initialize()
        #Write initial step to file  
        self._append_snapshot_to_trajectory()

        #LOOP KMC PARAMETERS
        nkmc_steps = self.config.control.n_steps
        time = 0.0 #in seconds
        nsearch = self.config.eventsearch.nsearch

        #KMC LOOP
        for step in range(nkmc_steps) :

            self.loggers.info('log', '{}{}Step : {}{}'.format(Colors.BOLD.value, Colors.YELLOW.value, step, Colors.RESET.value))

        #== Find Current atomic environments that has not been visited == 
            new_environments = self.get_new_environments()

        # == FIND NEW GENERIC EVENTS == 
                ##=>List of atoms(central) on which we gonna perfom an event search
            central_atom_research_list = self.central_atoms_research(new_environments, nsearch)

                ##=>Perform event search on each atom in central_atom_research_list
            event_search = self.execute_event_searches(central_atom_research_list)

        # == ADD NEW GENERIC EVENTS TO REFERENCE EVENT TABLE == 
                ##=>Check if the event is valid, ie if not already present and has a valid energy barrier if yes add it to the reference table
            results_is_valid_events = self.add_reference_events(event_search.get_successes_results()) 

                ##=>Close simulation if no events in the reference table  
            if len(self.reference_table.table) == 0 : 
                self.loggers.error('log', "No events have been found, empty reference events table. \n \tTry to increase nsearch or saddle point search algorithm's parameters. \n \tClosing the simulation.")
                self._close()

        # == Refinement == 
            ##=>Subset of reference_event_table with generic event that can be apply to the current step (ie event_id in atomic environment)
            subset_reference_event_table = self.reference_table.has_id_subset_table(self.atomic_environment.atomic_environment_list) 

            ##=>Refines all event in subset
            refinement = self.execute_refinements(subset_reference_event_table)

        # == ADD ACTIVE EVENT TO ACTIVE EVENT TABLE == 

            ##=>Construct active event table 
            active_table = ActiveEventTable(self.config) 
            active_table.add_events(refinement.get_successes_results())

        # == Update System == 
            ##=>Select event 
            idx_selected_event, delta_t, ktot = self._select_event(active_table)
            time += delta_t*10**-12 #time is in seconds

            ##=>Move system
            self._apply_event(idx_selected_event, active_table)

            ##=>Minimize 
            self.minimize_system()

        # == Log informations == 
            atomic_environment_info = self.get_info_atomic_environments(new_environments)
            reference_event_searches_info = self.get_info_reference_event_searches(event_search.results)
            is_valid_events_info = self.get_info_is_valid_reference_events(results_is_valid_events)
            refinements_info = self.get_info_refinements(refinement.results)
            kmc_loop_info = KMCLoopInfo(step = step, 
                                        atomic_environment_info=atomic_environment_info,
                                        reference_event_searches_info=reference_event_searches_info,
                                        valid_event_info=is_valid_events_info, 
                                        refinements_info=refinements_info 
                                        )
            self.loggers.info('info', kmc_loop_info.output_msg())

            self.loggers.table_line_info_kmc('output', step+1, delta_t*10**-12, time, active_table.table.loc[idx_selected_event].at['num_reference_event'], active_table.table.loc[idx_selected_event].at['energy_barrier'], active_table.table.loc[idx_selected_event].at['k'], ktot, self.total_energy )

        # == Update variables == 
            l_ids = list(set(self.atomic_environment.atomic_environment_list)) 
            self.visited_environments.update(set(l_ids).difference(self.visited_environments))
            self.neighbors_list = NeighborsList(self.system, self.config.atomicenvironment.rnei, self.config.atomicenvironment.rcut) 
            self.atomic_environment = AtomicEnvironment(self.config.atomicenvironment.style, self.neighbors_list.neighbors_list['rnei'], self.neighbors_list.neighbors_list['rcut'], self.config.atomicenvironment.neighbors_add)

       
        # == Save Reference Table and List visited environment : 
            self._save()
            self._append_snapshot_to_trajectory()


        # == Check if only cristalline environments == 
            if set(list(self.atomic_environment.atomic_environment_list)) == {"crystal"} : 
                self.loggers.info('log',':=> Only atoms with cristalline environment')
                self._close()
        
    def get_new_environments(self) : 
            new_environments = self.atomic_environment.get_new_environments(self.visited_environments)
            self.loggers.info('log', '\t :=> {} new atomic environments found'.format(len(new_environments)))
            return new_environments

    def central_atoms_research(self, new_environments: list[str | bytes], nsearch: int) -> list[int]: 
        """Generate list of central atoms on which we gonna perform generic event searches for the reference table

        For each new environment it adds nseach atoms having that environment to the list.

        Parameters
        ----------
        new_environment : list[str  |  bytes]
            List of atomic environment ID.
        nsearch : int
            Number of searches per atomic environment. 

        Returns
        -------
        list[int]
            List of central atoms

        Raises
        ------
        IndexError
            If no atoms are found for a given environment, random.choice will raise an IndexError.

        """
        central_atom_research_list = []
        #for each atomic environment hash in new_environment 
        for env in new_environments :
            #find all index having that hash
            tmp1 = [i for i,e in enumerate(self.atomic_environment.atomic_environment_list) if e == env] 
            #Randomly choose nsearch atoms that have that environment 
            tmp2 = [random.choice(tmp1) for _i in range(nsearch)]
            central_atom_research_list += tmp2
        return central_atom_research_list

    def execute_event_searches(self, central_atom_research_list) : 
        event_search = EventSearch(self.system, self.engine, self.loggers)
        event_search.execute(central_atom_research_list)
        return event_search


    def add_reference_events(self, events: list[EventSearchOutput] )  : 
        results_is_valid_events = self.reference_table.add_events(events)  
        self.loggers.info('log', "\t :=> Adding {} events to the reference table".format(len([e for e in results_is_valid_events if e.is_ok()])))
        return results_is_valid_events
    
    def execute_refinements(self, df_reference_events) : 
        refinement = Refinement(self.config, self.loggers, self.system, self.neighbors_list, self.atomic_environment, self.engine)
        refinement.execute(df_reference_events)
        return refinement

    def _select_event(self, active_table) : 
        """
        """
        #list of rate constant
        l_k = np.array([active_table.table.loc[i].at['k'] for i in range(len(active_table.table))])
        idx_selected_event, delta_t, ktot = rejection_free(l_k)
        return idx_selected_event, delta_t, ktot

    def _apply_event(self, idx_selected_event, active_table ) : 
        """ 
        """
        new_positions = active_table.table.loc[idx_selected_event].at['final_positions'] 
        self.system.update_positions(new_positions)

    def minimize_system(self) -> None : 
        """Minimize the system and update its positions"""
        self.loggers.info('log',':=> Minimizing the system')
        new_positions, total_energy = self.engine.minimize(self.system)
        self.system.update_positions(new_positions)
        self.total_energy = total_energy

    def get_info_atomic_environments(self, new_environments: list[str|bytes]) -> AtomicEnvironmentInfo : 
        return info_atomic_environments(self, new_environments)

    def get_info_reference_event_searches(self, results_reference_event_searches: list[Result[EventSearchOutput, ErrorInfo]]) -> ReferenceEventSearchInfo : 
        return info_reference_event_searches(results_reference_event_searches)

    def get_info_is_valid_reference_events(self, results_is_valid_events: list[Result[pd.DataFrame, ErrorInfo]]) -> ReferenceValidEventsInfo :
        return info_is_valid_reference_events(results_is_valid_events) 
    
    def get_info_refinements(self,  results_refinements: list[Result[EventSearchOutput, ErrorType]]) -> RefinementsInfo:
        return info_refinements(results_refinements) 

    def _initialize(self) : 
        Initializer(self).initialize()

    def _append_snapshot_to_trajectory(self) : 
        atoms = Atoms(self.system.types, positions=self.system.positions, cell=self.system.cell, pbc=self.system.pbc)
        write(self.config.control.trajectory_output, atoms, append=True)

    def _save(self) : 
        self.reference_table.save('reference_table.pickle')
        with open(self.config.control.visited_environments_output, 'wb') as file : 
            pickle.dump(self.visited_environments, file)

    def _close(self) : 
        self.loggers.info('log', ':=> End of simulation')
        sys.exit()