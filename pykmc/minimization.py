from lammps import lammps 
from ase.io.lammpsdata import write_lammps_data
import numpy as np 
from subprocess import run
from mpi4py import MPI
from .utilities import modify_lammps_data_2D
from executorlib import Executor



class Minimization:
    def __init__(self, system, minimization_style, minimization_params, potential, dimension, nprocs, backend) : 
        """ 
        Initialization 
        """
        self.system = system
        self.minimization_style = minimization_style
        self.minimization_params = minimization_params
        self.potential = potential
        self.dimension = dimension
        self.nprocs = nprocs
        self.backend = backend

    def run(self) : 
        """
        Execute minimization based on minimization_style
        return Atoms ASE object with updated positions
        """
        self.system.logger.logger.info('Use {} for minimization'.format(self.minimization_style))
        with Executor(backend=self.backend) as exe :
            match self.minimization_style : 
                case "lammps":
                    fs = exe.submit(self.minimize_lammps, resource_dict={"cores": self.nprocs})
                case _:
                    self.system.logger.logger.error('ERROR:Minimization style not known')
                    raise Exception("Minimization style not known")
        #TODO Need to find a solution for small negative numbers (ie Lammps can gives wrapped positions like 1.0e-10). It mess up with the k-d tree (could replicate positions and not use the box_size option in kdtree)
        #Set new positions : 
        positions = fs.result()
        positions[positions < 0] = 0
        self.system.set_positions(positions)

    def minimize_lammps(self) : 
        """
        Run Lammps minimization
            Parameters : 
                atoms (ASE Atoms Objects) 
                parameters (dict) : dictionary of lammps commands related to the minimization 
                potential (dict) : dictionary of lammps commands related to the potential 
                dimension (int) : dimension (default 3)

            Return : 
                positions (numpy array) : updated positions after minimization
        """ 
        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #TODO check create_atoms : https://docs.lammps.org/Python_module.html#lammps.lammps
        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #initialize lammps :
        lmp = lammps(comm=comm,cmdargs=['-log', 'log_minimize.lammps', '-screen', 'none'])

        #TODO should add posibility to use watherver parameters
        #for the moment default parameters 
        lmp.command('units metal')
        lmp.command('atom_style atomic')
        lmp.command('dimension {}'.format(self.dimension))
        lmp.command('boundary p p p')
        lmp.command('read_data {}'.format(lammps_data_file))
            #Potential
        for key, val in self.potential.items() : 
            lmp.command('{} {}'.format(key, val))
            #Minimization 
        for key, val in self.minimization_params.items() :
            lmp.command('{} {}'.format(key, val)) 
        #gather all positions 
        positions = lmp.gather_atoms("x", 1, 3)

        #Close lammps/MPI
        lmp.close()
        if rank == 0 : 
            #convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            #clean files
            run('rm {}'.format(lammps_data_file), shell=True)
            return positions 
