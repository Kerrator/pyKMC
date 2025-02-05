import random
from ase import Atoms
import numpy as np
import pandas as pd 
from ase.io import write
from ase.calculators.lammpslib import LAMMPSlib
from .atomic_environment import make_graph
import pynauty
import math as m

#TODO should extract energy system after minimization
#TODO using our own build in function to write outpufile traj file may be less time consuming than creating an Atoms and use ase.io.write

class KMC() : 
    """class to execute kmc simulation 
    """ 
    def __init__(self, system) :
        """
        """
        self.system = system
        self.control = self.system.inputs['Control']
        self.potential = self.system.inputs['Potential']
        self.minimization = self.system.inputs['Minimization']
        self.atomicenvironment_parameters = self.system.inputs['AtomicEnvironment']
        self.event_parameters = self.system.inputs['EventSearch']
        self.psr_parameters = self.system.inputs['PSR']
        self.time = None

    def run(self) : 
        """
        Execute nkmc steps
        """

        #=====#
        #Start#
        #=====#
        self.system.logger.logger.info('===========================')
        self.system.logger.logger.info('= Starting KMC simulation =')
        self.system.logger.logger.info('===========================')

        nkmc_steps = self.system.inputs['Control']['nkmc_steps']
        self.time = 0

        #====================#
        #Loop over nkmc_steps#
        #====================#
        for step in range(nkmc_steps) :
            #==============# 
            #Initialization#
            #==============# 
            if step == 0 : 
                self.initial_step_kmc()
            #Searching atomic environments : 
            self.system.find_environment(self.atomicenvironment_parameters['style'], self.atomicenvironment_parameters)
            #Searching Events : 
            self.system.event_search(self.event_parameters['style'], self.event_parameters, self.atomicenvironment_parameters, self.potential, self.system.reconstruction, self.control['dimension'], self.control['nprocs'], self.control['backend'])

            #add visited environment : 
            if self.system.reconstruction : 
                lids = [d['ID'] for d in self.system.environment]
                self.system.visited_environment.update(set(lids).difference(self.system.visited_environment)) 

            #If at leas one event in the catalog : 
            if len(self.system.catalog) > 0 : 
                #Select an event in the catalog : 
                idx_cat, delta_t = self.select_event_rejection_free(self.system.reconstruction) 
            else : 
                idx_cat = None

            if idx_cat is not None : #In case we have all atomic environments that do not have event
                if self.system.reconstruction : 
                #Select a central atom on which we will reconstruct the event : 
                    central_atom_index = self.select_central_atom(idx_cat)
                else : 
                    central_atom_index = self.system.catalog.loc[idx_cat].at['atom_index']

                if central_atom_index is not None : #Shoudl not happen ? 
                    if self.system.reconstruction : 
                        #Shape Matching
                        rmat, tr, perm, dh = self.system.point_set_registration(self.psr_parameters['style'], self.psr_parameters, idx_cat, central_atom_index, self.event_parameters['rcutenv'])
                        if rmat is not None : #in case ira did not find a match 
                        #Update positions of the system if recontruction success
                            reconstruction_de, reconstruction_topo = self.update_positions(idx_cat, central_atom_index, rmat, tr, perm, dh)
                            if reconstruction_de and reconstruction_topo : 
                                self.time += delta_t
                            self.system.logger.logger.info('{:<10n} {:<10e} {:<10n} {:<10n} {:<13n} {:<10e} {:<10e} {:<18s} {:<18s}'.format(step, self.time, len(self.system.environment), len(self.system.catalog), idx_cat, self.system.catalog.loc[idx_cat].at['energy_barrier'], dh, str(reconstruction_de), str(reconstruction_topo)))
                    else : 
                        #directly go to final position
                        rcutevent = self.event_parameters['rcutenv']
                        coords = self.system.catalog.loc[idx_cat].at['final_positions']
                        ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
                        dist = self.system.get_distances(central_atom_index, ind, mic=True)
                        neighbor_list = np.where(dist<rcutevent)[0]
                        #ugly
                        c = 0 
                        newpos = self.system.get_positions()
                        for i in range(len(newpos)) : 
                            if i in neighbor_list : 
                                newpos[i] = coords[c]
                                c +=1
                        self.system.set_positions(newpos)
                        self.time += delta_t
                        self.system.logger.logger.info('{:<10n} {:<10e} {:<10n} {:<10n} {:<13n} {:<10e}'.format(step, self.time, len(self.system.environment), len(self.system.catalog), idx_cat, self.system.catalog.loc[idx_cat].at['energy_barrier']))
                        #Destroy the catalog/initialize a new one: 
                        self.system.catalog = pd.DataFrame(columns = ['atom_index', 
                                                                      'final_positions',
                                                                      'energy_barrier',
                                                                      'k'])
                    #Minimize
                    self.system.minimize(self.minimization['style'], self.minimization, self.potential)
                    #Append new cong to trajectory file
                    write(self.system.kmc_traj, Atoms(self.system.get_chemical_symbols(), 
                                                         positions=self.system.get_positions(), 
                                                         cell = self.system.get_cell(),
                                                         pbc=self.system.get_pbc()), append=True)
                    
    def initial_step_kmc(self) : 
        """
        Procedure done at the start of the first step of a KMC simulation
        """
        #Minimization
        self.system.logger.logger.info(':> Minimization of the system')
        self.system.logger.new_line()
        self.system.minimize(self.minimization['style'], self.minimization, self.potential, self.control['dimension'], self.control['nprocs'], self.control['backend'])

        #First log table line 
        self.system.logger.first_line_table(self.system.reconstruction) 

        #Write first configuration to output file
        write(self.system.kmc_traj, Atoms(self.system.get_chemical_symbols(), 
                                          positions=self.system.get_positions(), 
                                          cell = self.system.get_cell(),
                                          pbc=self.system.get_pbc()), append=True)

    def select_event_random(self) : 
        """ 
        return index in system.catalog of a random possible event
        """  
        #find list of event in catalog that have id in system.environment : 
        l_env = [dict['ID'] for dict in self.system.environment]
        l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]
        if len(l_catalog) > 0 : 
            return random.choice(l_catalog)
        else : 
            return None
        
    def select_event_rejection_free(self, reconstruction) : 
        """
        Select an event and return its catalog id based on the rejection free KMC algorithm
        """
        if reconstruction : 
        #1-Find index list of all possible event in the catalog, ie events having IDs that are in the current system.environment
            l_env = [dict['ID'] for dict in self.system.environment]
            l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]
            print(len(l_catalog))
        else : 
        #1- if reconstruction = False all events are possible : 
            l_catalog = [i for i in range(len(self.system.catalog))]
        #2-Get constant rate of possible events
        k = np.array([self.system.catalog.loc[l_catalog[i]].at['k'] for i in range(len(l_catalog))])
        #3-Compute constant rate cummulative sum
        k_cumulative = [np.sum(k[:i]) for i in range(1,len(k)+1)]
        #4-Get random number [0,1[
        rand1 = random.random() 
        #5-Find event index satisfy ki-1<rand1ktot<ki
        idx_selected_event = np.searchsorted(k_cumulative, rand1*k_cumulative[-1], side = 'left')
        #6-Compute associated delta_t
        delta_t = -m.log(random.random())/k_cumulative[-1]
        return idx_selected_event, delta_t 

    def select_central_atom(self, idx_cat) : 
        """ 
        Find a central atom with same ID than the event at idx_cat in catalog
        """ 
        id = self.system.catalog.loc[idx_cat].at['event_id'] 
        atom_index_list = []
        for dic in self.system.environment : 
            if dic['ID'] == id : 
                atom_index_list = dic['atom index']
        #random atom : 
        if len(atom_index_list) > 0 : 
            central_atom_index = random.choice(atom_index_list)
        else : 
            central_atom_index = None
        return central_atom_index

    def update_positions(self, idx_cat, central_atom_index, rmat, tr, perm, dh) : 
        """ 
        update positions based on selected event 
        """ 
        #read psr_event file 
        #psr = pd.read_pickle('psr_event_'+str(idx_cat)+'.pickle') 
        #rmat = psr.loc[0].at['R']
        #tr = psr.loc[0].at['T']
        #perm = psr.loc[0].at['P']
        #dh = psr.loc[0].at['dh']


        if dh < 0.05 :
            #Compute energy of the system : 
            cmds = []
            cmds.append('pair_style {}'.format(self.potential['pair_style']))
            cmds.append('pair_coeff {}'.format(self.potential['pair_coeff']))
#            for key, val in self.potential.items() : 
#                 cmds.append('{} {}'.format(key,val))
            lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
            atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
            atoms.calc = lammps 
            Eini = atoms.get_potential_energy()

            #Save current positions in case reconstruction fail: 
            current_positions = self.system.get_positions()
            
            #Move system to the saddle point : 
                #PSR event saddle point
            #coords = self.system.catalog.loc[idx_cat].at['saddle_positions']
            coords = np.zeros((len(self.system.catalog.loc[idx_cat].at['saddle_positions']), 3))

            #TODO ADD CHECK TRANSLATION

            for i in range(len(coords)) : 
                coords[i] = np.matmul(rmat, self.system.catalog.loc[idx_cat].at['saddle_positions'][i]) + tr 
            coords[:] = coords[perm]
                #modify positions system 
            rcutevent = self.event_parameters['rcutenv']
            ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
            dist = self.system.get_distances(central_atom_index, ind, mic=True)
            neighbor_list = np.where(dist<rcutevent)[0]
            #ugly
            c = 0 
            newpos = self.system.get_positions()
            for i in range(len(newpos)) : 
                if i in neighbor_list : 
                    if c == self.system.catalog.loc[idx_cat].at['move_atom_idx'] : 
                        move_atomsys_idx = i
                    newpos[i] = coords[c]
                    c +=1

            self.system.set_positions(newpos)

            #Check energy barrier :
            #print(move_atomsys_idx)
            #TEST : 
            print('STEP')
            print(move_atomsys_idx)
            dist = (current_positions-newpos)**2  
            dist = dist.sum(axis=-1)
            dist = np.sqrt(dist)
            move_atomsys_idx = np.argmax(dist)
            print(move_atomsys_idx)
            is_energy_saddle_ok = self.check_saddle_energy(idx_cat, Eini)
            is_topo_saddle_ok = self.check_saddle_topo(idx_cat, move_atomsys_idx) 
            
            if not is_energy_saddle_ok or not is_topo_saddle_ok : 
                self.system.set_positions(current_positions)
            else : 
                #move to final positions 
                coords = np.zeros((len(self.system.catalog.loc[idx_cat].at['final_positions']),3))
                for i in range(len(coords)) : 
                    coords[i] = np.matmul(rmat, self.system.catalog.loc[idx_cat].at['final_positions'][i]) + tr 
                coords[:] = coords[perm]
                c=0 
                for i in range(len(newpos)) : 
                    if i in neighbor_list : 
                        newpos[i] = coords[c]
                        c+=1 
                self.system.set_positions(newpos)
            return is_energy_saddle_ok, is_topo_saddle_ok

        else : 
            return None, None

           







           # #Check if E_barrier is consistent : 
           # is_saddle_e_consistent = self.check_saddle_energy(idx_cat, central_atom_index)
           # is_saddle_topo_consistent = self.check_saddle_topo(idx_cat, central_atom_index) 
           # if is_saddle_e_consistent and is_saddle_topo_consistent: 

           #     #transform event position 
           #     coords = self.system.catalog.loc[idx_cat].at['final_positions']
           #     for i in range(len(coords)) : 
           #         coords[i] = np.matmul(rmat, coords[i]) + tr 
           #     coords[:] = coords[perm]
           #         #find neighbor list 
           #     rcutevent = 7.0
           #     ind = np.linspace(0, self.system.get_global_number_of_atoms()-1, self.system.get_global_number_of_atoms()).astype(int)
           #     dist = self.system.get_distances(central_atom_index, ind, mic=True)
           #     neighbor_list = np.where(dist<rcutevent)[0]
           #         #replace positions of the neighbors 
           #     #ugly
           #     c = 0 
           #     newpos = self.system.get_positions()
           #     for i in range(len(newpos)) : 
           #         if i in neighbor_list : 
           #             newpos[i] = coords[c]
           #             #self.system.positions[i] = coords[c]
           #             c +=1

           #     self.system.set_positions(newpos)
           #     self.system.set_positions(self.system.get_positions(wrap=True))
           # else : 
           #     self.system.logger.logger.info('Energy barrier or topo not consistent')
        #else : 
            #self.system.logger.logger.info('PSR: dh distance > {}, could not update positions '.format(0.2))




    def check_saddle_energy(self, idx_cat, E_ini) : 
        #get energy system at saddle point : 
        cmds = []
        cmds.append('pair_style {}'.format(self.potential['pair_style']))
        cmds.append('pair_coeff {}'.format(self.potential['pair_coeff']))
#        for key, val in self.potential.items() : 
#            cmds.append('{} {}'.format(key,val))
        lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
        atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
        atoms.calc = lammps 
        #Current potential energy of the system 
        E_sad = atoms.get_potential_energy()
        dE = E_sad-E_ini
        #self.system.logger.logger.info('Checking barrier energy = {} '.format(dE))

        return abs(dE-self.system.catalog.loc[idx_cat].at['energy_barrier']) < 0.5

    def check_saddle_topo(self, idx_cat, move_atomsys_idx) : 
        #compute topo of moving atom in system : 
        atoms = Atoms(positions=self.system.get_positions(), cell = self.system.get_cell(), pbc=True)
#        pos = atoms.get_positions()
#        pos[pos < 0] = 0
#        atoms.set_positions(pos)
        g_saddle = make_graph(atoms, [move_atomsys_idx], self.atomicenvironment_parameters['rnei'], self.atomicenvironment_parameters['rcut'])[0]
        topo_saddle = pynauty.certificate(g_saddle)

        check = topo_saddle == self.system.catalog.loc[idx_cat].at['id_saddle']
        #self.system.logger.logger.info('Topo saddle reconstruction = {}'.format(check))
        return check 
        #return True


#    def check_saddle_energy(self, idx_cat, central_atom_index) :
#        """ 
#        """ 
#        #get energy of the system : 
#        cmds = []
#        for key, val in self.potential.items() : 
#            cmds.append('{} {}'.format(key,val))
#        lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
#        atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
#        atoms.calc = lammps 
#        #Current potential energy of the system 
#        Eini = atoms.get_potential_energy() 
#
#        #get saddle point energy : 
#            #update atoms positions to the saddle point : 
#        psr = pd.read_pickle('psr_event_'+str(idx_cat)+'.pickle') 
#        rmat = psr.loc[0].at['R']
#        tr = psr.loc[0].at['T']
#        perm = psr.loc[0].at['P']
#        dh = psr.loc[0].at['dh']
#        #coords = self.system.catalog.loc[idx_cat].at['initial_positions']
#        #print(coords)
#        coords = self.system.catalog.loc[idx_cat].at['saddle_positions']
#        #print(coords)
#        #for i in range(len(coords)) : 
#        #    coords[i] = np.matmul(rmat, coords[i]) + tr 
#        #coords[:] = coords[perm]
#        #rcutevent = 7.0
#        #ind = np.linspace(0, atoms.get_global_number_of_atoms()-1, atoms.get_global_number_of_atoms()).astype(int)
#        #dist = atoms.get_distances(central_atom_index, ind, mic=True)
#        #neighbor_list = np.where(dist<rcutevent)[0]
#        neighbor_list = psr.loc[0].at['neighbor_list']
#            #replace positions of the neighbors 
#        #ugly
#        c = 0 
#        newpos = atoms.get_positions()
#        #self.system.logger.logger.info('DEBUG: {} atoms in event and {} in central atom neigh'.format(len(coords), len(neighbor_list)))
#        for i in range(len(newpos)) : 
#            if i in neighbor_list : 
#                #newpos[i] = coords[c] 
#                newpos[i] = np.matmul(rmat, coords[c]) +tr
#                c +=1
#
#        atoms.set_positions(newpos)
#        Esad = atoms.get_potential_energy() 
#        dE = Esad-Eini
#        self.system.logger.logger.info('Checking barrier energy = {} '.format(dE))
#        write('saddlepoint_psr.xyz', atoms)
#        return abs(dE-self.system.catalog.loc[idx_cat].at['energy_barrier']) < 0.5
#
#    def check_saddle_topo(self, idx_cat, central_atom_index) :
#        """ 
#        check topo id consitency between event saddle point and new saddle point 
#        """ 
#        #compute id of the general event saddle point : 
#        atoms = Atoms(positions = self.system.catalog.loc[idx_cat].at['saddle_positions'])
#        g_general_event = make_graph(atoms, [self.system.catalog.loc[idx_cat].at['central_atom']], 3.0, 5.0)[0]
#        write('test1.xyz', atoms)
#        print(self.system.catalog.loc[idx_cat].at['central_atom'])
#
#
#        #compute id of saddle point : 
#        atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
#            #update atoms positions to the saddle point : 
#        psr = pd.read_pickle('psr_event_'+str(idx_cat)+'.pickle') 
#        rmat = psr.loc[0].at['R']
#        tr = psr.loc[0].at['T']
#        perm = psr.loc[0].at['P']
#        dh = psr.loc[0].at['dh']
#        coords = self.system.catalog.loc[idx_cat].at['saddle_positions']
#        neighbor_list = psr.loc[0].at['neighbor_list']
#            #replace positions of the neighbors 
#        central_atom_in_event = self.system.catalog.loc[idx_cat].at['central_atom']
#        #ugly
#        c = 0 
#        newpos = atoms.get_positions()
#        for i in range(len(newpos)) : 
#            if i in neighbor_list : 
#                if c == central_atom_in_event : 
#                    test = i
#                #newpos[i] = coords[c] 
#                newpos[i] = np.matmul(rmat, coords[c]) +tr
#                c +=1
#        atoms.set_positions(newpos)
#        write('test2.xyz', atoms)
#        print(test)
#        g_current = make_graph(atoms, [test], 3.0, 5.0)[0] 
#
#        is_ok = pynauty.certificate(g_general_event) == pynauty.certificate(g_current)
#
#        self.system.logger.logger.info('Checking saddle topo = {} '.format(is_ok))
#        is_ok = True
#        return is_ok 
#        
#







#center of the sphère : 
            #center_min1 = np.mean(min1positions, axis=0)
            #center_min2 = np.mean(min2positions, axis=0)
            #center_saddle = np.mean(saddlepositions, axis=0)

            ##check if there is a translation
            #shift_min2 = center_min2 - center_min1
            #shift_saddle = center_saddle - center_min1
            #print("Shift min2:", shift_min2)
            #print("Shift saddle:", shift_saddle)

            #Correct translation if needed 
            #min2positions = min2positions - shift_min2
            #saddlepositions = saddlepositions - shift_saddle



