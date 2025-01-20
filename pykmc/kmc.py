import random
from ase import Atoms
import numpy as np
import pandas as pd 
from ase.io import write
from ase.calculators.lammpslib import LAMMPSlib
from .atomic_environment import make_graph
import pynauty

#TODO Could add a write_log() file to call at each steps
#TODO Add kmc algo
#TODO Check saddle point topo consistency
#TODO Logs
#TODO Comments/Doc
#TODO should check saddle topo and energy at the same time (some computation are done 2 times)
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

    def run(self) : 
        """
        Execute nkmc steps
        """
        nkmc_steps = self.system.inputs['Control']['nkmc_steps']

        self.system.logger.logger.info('= Starting KMC simulation')
        for step in range(nkmc_steps) : 
            #Initialization
            if step == 0 : 
                self.system.logger.logger.info('Minimization of the system')
                self.system.minimize(self.minimization['style'], self.minimization, self.potential)
                self.system.logger.new_line()
                self.system.logger.logger.info('{:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s} {:<10s}'.format('Step', 'Time', 'Etot', 'Ndiff_env', 'N_event', 'n_select_event', 'cputime'))
                write(self.control['output_file'], Atoms(self.system.get_chemical_symbols(), 
                                                         positions=self.system.get_positions(), 
                                                         cell = self.system.get_cell(),
                                                         pbc=self.system.get_pbc()), append=True)
            #Searching atomic environments : 
             
#        traj = [Atoms(symbols=self.system.get_chemical_symbols(),
#                         positions=self.system.get_positions(),
#                         cell=self.system.get_cell(),
#                         pbc=self.system.get_pbc())]
#        for step in range(nkmc_steps) :
            
#            self.system.logger.logger.info('===============================')
#            self.system.logger.logger.info('======== NEW STEP N°{} ========'.format(step))
#            self.system.logger.logger.info('===============================')
#
#            if step == 0 :  
#                self.system.logger.logger.info(':> Minimization of the system')
#                self.system.minimize('lammps', self.minimization_params, self.potential, nprocs=1, backend='local')
#
#            self.system.logger.logger.info(":> Searching atoms's atomic environments")
#            self.system.find_environment('cna/graph', self.atomenv_params, dimension=3, nprocs=1)
#
#            self.system.event_search('pARTn', self.eventsearch_params, self.potential)
#            if len(self.system.catalog)>0 : 
#
#
#
#                self.system.logger.logger.info('Applying Event')
#                idx_cat = self.select_event() 
#
#
#
#                if idx_cat != None : 
#
#
#                    self.system.logger.logger.info('event n° {} has been chosen'.format(idx_cat))
#                    central_atom_index = self.select_central_atom(idx_cat)
#
#
#                    if central_atom_index != None : 
#                        rmat, tr, perm, dh = self.system.point_set_registration('ira', idx_cat, central_atom_index, 7.0)
#
#
#                        self.update_positions(idx_cat, central_atom_index, rmat, tr, perm, dh)
#
#                        #read psr_event file 
#                        self.system.logger.logger.info(':> Minimization of the system')
#                        self.system.minimize('lammps', self.minimization_params, self.potential, nprocs=1, backend='local')
#                else :
#                    self.system.logger.logger.info('at id not in catalog')
#            traj.append(Atoms(symbols=self.system.get_chemical_symbols(),
#                         positions=self.system.get_positions(),
#                         cell=self.system.get_cell(),
#                         pbc=self.system.get_pbc()))
#            write('trajkmc.xsf', traj) 
#
#       self.system.kmc_traj = traj

    def select_event(self) : 
        """ 
        return index in system.catalog
        """  
        #TODO Algo : pour le moment random
        #find list of event in catalog that have id in system.environment : 
        l_env = [dict['ID'] for dict in self.system.environment]
        l_catalog = [i for i in range(len(self.system.catalog)) if self.system.catalog.loc[i].at['event_id'] in l_env ]
        if len(l_catalog) > 0 : 
            return random.choice(l_catalog)
        else : 
            return None

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
            for key, val in self.potential.items() : 
                 cmds.append('{} {}'.format(key,val))
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
            rcutevent = 7.0 
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
        else : 
            self.system.logger.logger.info('PSR: dh distance > {}, could not update positions '.format(0.2))




    def check_saddle_energy(self, idx_cat, E_ini) : 
        #get energy system at saddle point : 
        cmds = []
        for key, val in self.potential.items() : 
            cmds.append('{} {}'.format(key,val))
        lammps = LAMMPSlib(lmpcmds=cmds, log_file='log.calc_energy.lammps')
        atoms = Atoms(self.system.get_global_number_of_atoms()*['Ni'], positions=self.system.get_positions(), cell=self.system.cell, pbc=True) 
        atoms.calc = lammps 
        #Current potential energy of the system 
        E_sad = atoms.get_potential_energy()
        dE = E_sad-E_ini
        self.system.logger.logger.info('Checking barrier energy = {} '.format(dE))

        return abs(dE-self.system.catalog.loc[idx_cat].at['energy_barrier']) < 0.5

    def check_saddle_topo(self, idx_cat, move_atomsys_idx) : 
        #compute topo of moving atom in system : 
        atoms = Atoms(positions=self.system.get_positions(), cell = self.system.get_cell(), pbc=True)
        pos = atoms.get_positions()
        pos[pos < 0] = 0
        atoms.set_positions(pos)
        g_saddle = make_graph(atoms, [move_atomsys_idx], 3.0, 5.0)[0]
        topo_saddle = pynauty.certificate(g_saddle)

        check = topo_saddle == self.system.catalog.loc[idx_cat].at['id_saddle']
        self.system.logger.logger.info('Topo saddle reconstruction = {}'.format(check))
        return check 


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



