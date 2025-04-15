from pykmc.base_engine import BaseEngine 
import numpy as np 
from ase.data import atomic_numbers, atomic_masses
from lammps import lammps
from pykmc.config import Config
from .partn import pARTn_search

class LammpsEngine(BaseEngine) : 

    def __init__(self, config: Config) :
        self.config_control = config.get('Control') 
        self.config_potential = config.get('Potential')
        self.config_minimization = config.get('Minimization')
        self.config_event_search = config.get('EventSearch')
        self.config_atomic_environment = config.get('AtomicEnvironment')


    def _initialize_default(self,system, lmp_instance) : 

        #parameters
        natoms = len(system.types) 
        cell = system.cell 
        types = system.types
        x = system.positions.flatten() #Lammps format

        xhi, yhi, zhi = cell[0][0], cell[1,1], cell[2,2]

        ind = np.linspace(0, natoms-1, natoms).astype(int)
        ind += 1 #Lammps id start at 1
        #map type to int alphabetic order create a dictionary with atom id and mass, eg {'H' : {'ref': 1, 'mass' : 1.00}, 'Ni': {'ref' : 2, 'mass' : 58.69} }
        map_type = {atom_type: {'ref' :i+1, 'mass' : atomic_masses[atomic_numbers[atom_type]]} for i, atom_type in enumerate(sorted(set(types)))}
        types = [map_type[element]['ref'] for element in types] #map to integer

        #lammps command
        lmp_instance.command('units metal')
        lmp_instance.command('atom_style atomic')
        lmp_instance.command('dimension 3')
        lmp_instance.command('boundary p p p')
        lmp_instance.command('atom_modify sort 0 0.0')
        lmp_instance.command('region box block 0.0 {} 0.0 {} 0.0 {}'.format(xhi, yhi, zhi))
        lmp_instance.command('create_box {} box'.format(len(map_type)))
        lmp_instance.create_atoms(natoms, ind,types, x)
        #Set masses
        for key in map_type.keys() :
            lmp_instance.command('mass {} {}'.format(map_type[key]['ref'], map_type[key]['mass']))
        #Label atoms name to type : 
        lmp_instance.command('labelmap atom '+ " ".join(f"{int(e['ref'])} {key}" for key, e in map_type.items()))

    def _initialize_potential(self, lmp_instance) : 

        pair_style = self.config_potential['pair_style']
        pair_coeff = self.config_potential['pair_coeff']
        lmp_instance.command('pair_style {}'.format(pair_style))
        lmp_instance.command('pair_coeff {}'.format(pair_coeff))

    def minimize(self, system) :

        #Parameters
        min_style = self.config_minimization['min_style']
        etol = self.config_minimization['etol']
        ftol = self.config_minimization['ftol']
        maxiter = self.config_minimization['maxiter']
        maxeval = self.config_minimization['maxeval']
        
        #Lammps instance
        from mpi4py import MPI
        #MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
        lmp = lammps(comm=comm)

        #Lammps default parameters
        self._initialize_default(system, lmp)
        #Initialize potential
        self._initialize_potential(lmp)
        #Minimization 
        lmp.command('min_style {}'.format(min_style))
        lmp.command('minimize {} {} {} {}'.format(etol, ftol, maxiter, maxeval))
        #gather all positions 
        id = lmp.numpy.extract_atom("id")
        positions = lmp.gather_atoms("x", 1, 3)
        if rank == 0 : 
            #convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            return positions
        else : 
            return None
        
    def pARTn(self, system, central_atom) : 
        #Parameters

        lmp = lammps()
        #Lammps default parameters : 
        self._initialize_default(system, lmp)
        #Initialize potential 
        self._initialize_potential(lmp)
        #pARTn search : 
        result = pARTn_search(lmp, self.config_event_search, central_atom, self.config_atomic_environment['rcut'])
        return result

    def compute_distances(self, system) : 
        pass
    def neighbors(self, system) :
        pass