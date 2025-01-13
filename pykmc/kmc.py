import random
from ase import Atoms
import numpy as np
import pandas as pd 
from ase.io import write
from ase.calculators.lammpslib import LAMMPSlib

#TODO Could add a write_log() file to call at each steps
#TODO Add kmc algo
#TODO Check saddle point topo consistency
#TODO Logs
#TODO Comments/Doc
#TODO should check saddle topo and energy at the same time (some computation are done 2 times)

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
            self.system.logger.logger.info('NEW STEP N°{}'.format(step))
            if step == 0 :  
                self.system.minimize('lammps', self.minimization_params, self.potential, nprocs=1, backend='local')
            self.system.find_environment('cna/graph', self.atomenv_params, dimension=3, nprocs=1)
            self.system.event_search('pARTn', self.eventsearch_params, self.potential)
            if len(self.system.catalog)>1 : 
                self.system.logger.logger.info('Applying Event')
                idx_cat = self.select_event() 
                self.system.logger.logger.info('event n° {} have been chosen'.format(idx_cat))
                central_atom_index = self.select_central_atom(idx_cat)
                self.system.point_set_registration('ira', idx_cat, central_atom_index, self.atomenv_params['rcut'])
                self.update_positions(idx_cat, central_atom_index)
                #read psr_event file 
                #TODO better than writing file 
                self.system.minimize('lammps', self.minimization_params, self.potential, nprocs=1, backend='local')
            #    self.update_positions(idx_cat) 
            traj.append(Atoms(symbols=self.system.get_chemical_symbols(),
                         positions=self.system.get_positions(),
                         cell=self.system.get_cell(),
                         pbc=self.system.get_pbc()))
            write('trajkmc.xsf', traj) 

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

    def select_central_atom(self, idx_cat) : 
        """ 
        Find a central atom with same ID than the event at idx_cat in catalog
        """ 
        id = self.system.catalog.loc[idx_cat].at['event_id'] 
        for dic in self.system.environment : 
            if dic['ID'] == id : 
                atom_index_list = dic['atom index']
        #random atom : 
        central_atom_index = random.choice(atom_index_list)
        return central_atom_index

    def update_positions(self, idx_cat, central_atom_index) : 
        """ 
        update positions based on selected event 
        """ 
        #read psr_event file 
        psr = pd.read_pickle('psr_event_'+str(idx_cat)+'.pickle') 
        rmat = psr.loc[0].at['R']
        tr = psr.loc[0].at['T']
        perm = psr.loc[0].at['P']
        dh = psr.loc[0].at['dh']


        if dh < 0.05 :

            #Check if E_barrier is consistent : 
            is_saddle_e_consistent = self.check_saddle_energy(idx_cat, central_atom_index)
            #is_saddle_topo_consitent = self.check_saddle_topo(idx_cat, central_atom_index) 
            if is_saddle_e_consistent : 

                #transform event position 
                coords = self.system.catalog.loc[idx_cat].at['final_positions']
                for i in range(len(coords)) : 
                    coords[i] = np.matmul(rmat, coords[i]) + tr 
                coords[:] = coords[perm]
                    #find neighbor list 
                rcutevent = 7.0
                ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
                dist = self.system.get_distances(central_atom_index, ind, mic=True)
                neighbor_list = np.where(dist<rcutevent)[0]
                    #replace positions of the neighbors 
                #ugly
                c = 0 
                newpos = self.system.get_positions()
                for i in range(len(newpos)) : 
                    if i in neighbor_list : 
                        newpos[i] = coords[c]
                        #self.system.positions[i] = coords[c]
                        c +=1

                self.system.set_positions(newpos)
                self.system.set_positions(self.system.get_positions(wrap=True))
            else : 
                self.system.logger.logger.info('Energy barrier not consistent')
        else : 
            self.system.logger.logger.info('PSR: dh distance > {}, could not update positions '.format(0.2))


    def check_saddle_energy(self, idx_cat, central_atom_index) :
        """ 
        """ 
        #get energy of the system : 
        cmds = []
        for key, val in self.potential.items() : 
            cmds.append('{} {}'.format(key,val))
        lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
        atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
        atoms.calc = lammps 
        #Current potential energy of the system 
        Eini = atoms.get_potential_energy() 

        #get saddle point energy : 
            #update atoms positions to the saddle point : 
        psr = pd.read_pickle('psr_event_'+str(idx_cat)+'.pickle') 
        rmat = psr.loc[0].at['R']
        tr = psr.loc[0].at['T']
        perm = psr.loc[0].at['P']
        dh = psr.loc[0].at['dh']
        coords = self.system.catalog.loc[idx_cat].at['initial_positions']
        print(coords)
        coords = self.system.catalog.loc[idx_cat].at['saddle_positions']
        print(coords)
        #for i in range(len(coords)) : 
        #    coords[i] = np.matmul(rmat, coords[i]) + tr 
        #coords[:] = coords[perm]
        #rcutevent = 7.0
        #ind = np.linspace(0, atoms.get_global_number_of_atoms()-1, atoms.get_global_number_of_atoms()).astype(int)
        #dist = atoms.get_distances(central_atom_index, ind, mic=True)
        #neighbor_list = np.where(dist<rcutevent)[0]
        neighbor_list = psr.loc[0].at['neighbor_list']
            #replace positions of the neighbors 
        #ugly
        c = 0 
        newpos = atoms.get_positions()
        self.system.logger.logger.info('DEBUG: {} atoms in event and {} in central atom neigh'.format(len(coords), len(neighbor_list)))
        for i in range(len(newpos)) : 
            if i in neighbor_list : 
                #newpos[i] = coords[c] 
                newpos[i] = np.matmul(rmat, coords[c]) +tr
                c +=1

        atoms.set_positions(newpos)
        Esad = atoms.get_potential_energy() 
        dE = Esad-Eini
        self.system.logger.logger.info('Checking barrier energy = {} '.format(dE))
        write('saddlepoint_psr.xyz', atoms)
        return abs(dE-self.system.catalog.loc[idx_cat].at['energy_barrier']) < 0.1

    def check_saddle_topo(self, idx_cat, central_atom_index) :
        """ 
        check topo id consitency between event saddle point and new saddle point 
        """ 
        return True

