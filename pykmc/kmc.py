import random
from ase import Atoms
import numpy as np

#TODO Could add a write_log() file to call at each steps


class KMC() : 
    """class to execute kmc simulation 
    """ 
    def __init__(self, system, kmc_parameters, minimization_params, atomenv_params, eventsearch_params,  potential, dimension, backend) : 
        """ 
         
        """
        self.system = system
        self.potential = potential
        self.kmc_parameters = kmc_parameters 
        self.minimization_params = minimization_params 
        self.atomenv_params = atomenv_params
        self.eventsearch_params = eventsearch_params
        self.dimension = dimension
        self.backend = backend 

    def run(self) : 
        """
        Execute nkmc steps
        """
        nkmc_steps = self.kmc_parameters["nkmc_steps"]
        traj = [Atoms(symbols=self.system.get_chemical_symbols(),
                         positions=self.system.get_positions(),
                         cell=self.system.get_cell(),
                         pbc=self.system.get_pbc())]
        for step in range(nkmc_steps) : 
            self.system.minimize('lammps', self.minimization_params, self.potential, nprocs=1, backend='local')
            self.system.find_environment('cna/graph', self.atomenv_params, dimension=3, nprocs=1)
            self.system.event_search('pARTn', self.eventsearch_params, self.potential)
            self.system.point_set_registration('ira')
            #if len(self.system.catalog)>1 : 
            #    idx_cat = self.select_event() 
            #    self.update_positions(idx_cat) 
            #traj.append(Atoms(symbols=self.system.get_chemical_symbols(),
            #             positions=self.system.get_positions(),
            #             cell=self.system.get_cell(),
            #             pbc=self.system.get_pbc()))

        self.system.kmc_traj = traj

    def select_event(self) : 
        """ 
        return index in system.catalog
        """  
        #TODO Algo : pour le moment random
        #find list of event in catalog that have id in system.environment : 
        l_env = [dict['ID'] for dict in self.system.environment]
        l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]

        return random.choice(l_catalog) 

    def update_positions(self, idx_cat) : 
        """ 
        update positions based on selected event 
        """ 
        #TODO Need to fix this shit
        self.system.wrap(eps=1e-2)
        positions = self.system.catalog.loc[idx_cat].at["final_positions"]
        positions[positions < 0] = 0
        positions[positions > self.system.cell[0][0]-0.1] = self.system.cell[0][0]-0.1
        self.system.set_positions(positions)
        #self.system.set_positions(self.system.catalog.loc[idx_cat].at["final_positions"])
