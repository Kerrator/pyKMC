from lammps import lammps 
from ase.io.lammpsdata import write_lammps_data
import numpy as np 
from subprocess import run
from mpi4py import MPI
from .utilities import modify_lammps_data_2D
from executorlib import Executor



class Minimization:
    """ Class to execute diffent minimization procedure
    """

    def __init__(self, system, minimization_style, minimization_params, potential, dimension, nprocs, backend) : 
        """ 
         
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

        #with Executor(backend=self.backend, max_cores=self.nprocs, cores_per_worker=self.nprocs) as exe : 
        with Executor(backend=self.backend, max_cores=self.nprocs) as exe : 
            match self.minimization_style : 
                case "lammps":
                    fs = exe.submit(self.minimize_lammps)
                    if self.nprocs == 1 : 
                        positions = fs.result()
                    else : 
                        positions = fs.result()[0]
                case _:
                    raise Exception("Minimization style not known")
        #Set new positions : 
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
                atoms (ASE Atoms Objects) : same Atoms Objects as input with updated positions after minimization
        """ 
        #for MPI : 
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()

        #Write lammps data file : 
        lammps_data_file = 'initial_config_minimization.lmp'
        if rank == 0 :
            write_lammps_data(lammps_data_file, self.system, masses=True)
            if self.dimension == 2 : 
                modify_lammps_data_2D(lammps_data_file)

        #initialize lammps :
        lmp = lammps(comm=comm,cmdargs=['-log', 'log_minimize.lammps'])

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

        if rank == 0 : 
            #convert ctype positions into a numpy array
            positions = np.ctypeslib.as_array(positions)
            positions = np.reshape(positions, (-1, 3))
            #clean files
            run('rm {}'.format(lammps_data_file), shell=True)
            #set new positions after minimization
            return positions 
        #close lammps/MPI
        lmp.close()