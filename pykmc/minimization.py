from lammps import lammps 
from ase.io.lammpsdata import write_lammps_data
import numpy as np 
from subprocess import run
from mpi4py import MPI
from .utilities import modify_lammps_data_2D


def minimize_lammps(atoms, parameters, potential, dimension=3) : 
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
        write_lammps_data(lammps_data_file, atoms, masses=True)
        if dimension == 2 : 
            modify_lammps_data_2D(lammps_data_file)

    #initialize lammps :
    lmp = lammps()

    #TODO should add posibility to use watherver parameters
    #for the moment default parameters 
    lmp.command('units metal')
    lmp.command('atom_style atomic')
    lmp.command('dimension {}'.format(dimension))
    lmp.command('boundary p p p')
    lmp.command('read_data {}'.format(lammps_data_file))
        #Potential
    for key, val in potential.items() : 
        lmp.command('{} {}'.format(key, val))
        #Minimization 
    for key, val in parameters.items() :
        lmp.command('{} {}'.format(key, val)) 
    #gather all positions 
    positions = lmp.gather_atoms("x", 1, 3)

    if rank == 0 : 
        #convert ctype positions into a numpy array
        positions = np.ctypeslib.as_array(positions)
        positions = np.reshape(positions, (-1, 3))
        #set new positions after minimization
        atoms.set_positions(positions)
        #clean files
        run('rm {}'.format(lammps_data_file), shell=True)
    #close lammps/MPI
    MPI.Finalize()
    return atoms


    