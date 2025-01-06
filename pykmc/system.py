from ase import Atoms
from ase.io import read
import pandas as pd
from .log import Logger
import os


#TODO : Could get rid of the attributes symboles, positions, cell and pbc, and only use the corresponding ASE methods 
#TODO : Could use a configfile/configfileparser to initialize the minimization, psr, atomic environment and event search parameters

class System(Atoms):
    """
    Extension of the Atoms Ase object with methods to run each needed steps of a kinetic monte carlo simulation 

    The object can use all Atoms Ase object methods and is initialized by using a config file and a catalog file (optional)

    Parameters
    ----------
    file_path : str
        path to the configuration file (format should be readable by the ase.io.read() method)
    catalog : str, optional
        path to the catalog pickle file from a previous simulation, by default None

    Attributes 
    ----------
    logger : Logger object
        to deal with log informations
    symboles : List 
        list of atoms type of the system 
    positions : numpy array 
        current atomic positions 
    cell : Ase cell object 
        cell parameters of the system 
    pbc : bool
        if periodic boundary conditions are used 
    environment : List[dict] 
        each dictionary contains an environment ID ("ID") and a list of atoms index having that ID ("atom index")
    catalog : pandas DataFrame 
        events containing the event ID, initial, saddle point and finale position, the energy barrier and k
    kmc_traj : List[Ase Atoms Object]
        configuration of each step of the kinetic monte carlo simulation

    Methods 
    -------
    minimize(minimization_style, minimization_params, potential, dimension=3, nprocs=1, backend='local')
        call the Minimize.run() method, see the Minimization object 
    find_environment(self, environment_style, environement_params, dimension=3, nprocs=1, backend='local')
        call the atomic_environment.run() method, see the AtomicEnvironment object
    event_search(self, search_style, search_params, potential, dimension=3, nprocs=1, backend='local')
        call the event_search.run() method, see the Event object 
    point_set_registration(self, psr_style, idx_cat,central_atom_index, rcutevent, dimension=3, nprocs=1, backend='local')
        call the psr.run() method, see the PointSetRegistration object 
    kmc(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension=3, backend='local') :
        call the kmc.run() method, see the KMC object

    Examples 
    -------- 
    >>> system = System('config.xyz')
    >>> system.kmc(kmc_parameters, minimization_parameters, search_parameters, potential)
    """

    def __init__(self, file_path, catalog=None):

        #Setup logfile
        try : 
            os.remove('pykmc.log')
        except OSError : 
            pass 
        self.logger = Logger('pykmc.log')
        self.logger.title(self.logger)
        self.logger.logger.info('##############################')
        self.logger.logger.info('#       INITIALIZATION       #')
        self.logger.logger.info('##############################')

        #Read initial config file
        self.logger.logger.info('reading {} configuration file'.format(file_path))
        atoms = read(file_path)  # Load configurations from file 
        super().__init__(symbols=atoms.get_chemical_symbols(),
                         positions=atoms.get_positions(),
                         cell=atoms.get_cell(),
                         pbc=atoms.get_pbc())

        self.environment = None

        #Note :  using Dataframe is usefull to do operation on it, but doing it that way, every time we 
        #add a new event, we have to copy the entire catalog. 
        #Dataframe are not made to be extended
        #could use dict of list 
        if catalog == None : 
            self.catalog = pd.DataFrame(columns=['event_id', 
                                                 'initial_positions', 
                                                 'saddle_positions', 
                                                 'final_positions', 
                                                 'energy_barrier', 
                                                 'k'])
        else : 
            self.logger.logger.info('reading {} catalog file'.format(catalog))
            self.catalog = pd.read_pickle(catalog) #for restart
        
        self.kmc_traj = None #for restart
        self.logger.logger.info('')

    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1, backend='local') : 
        """ 
        Minimize the system and update system positions 
        """
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs, backend)
        minimizer.run()

    def find_environment(self, environment_style, environement_params, dimension=3, nprocs=1, backend='local') : 
        """ 
        Find atomic environment for each atoms in System and create a dictionary 
        """
        from .atomic_environment import AtomicEnvironment 
        atomic_environment = AtomicEnvironment(self, environment_style, environement_params, dimension, nprocs, backend)
        atomic_environment.run()

    def event_search(self, search_style, search_params, potential, dimension=3, nprocs=1, backend='local') : 
        """ 
        For each atomic environment ID that are not in the catalog (except 'crist'), we launch 20 ARTn search 
        and add event to the catalog
        """
        from .event import EventSearch 
        event_search = EventSearch(self, search_style, search_params, potential, dimension, nprocs, backend)
        event_search.run()
    
    def point_set_registration(self, psr_style, idx_cat,central_atom_index, rcutevent, dimension=3, nprocs=1, backend='local') : 
        from .point_set_registration import PointSetRegistration 
        psr = PointSetRegistration(self, psr_style, idx_cat,central_atom_index, rcutevent, dimension, nprocs, backend)
        psr.run()

    def kmc(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension=3, backend='local') :
        """
        """ 
        from .kmc import KMC 
        kmc = KMC(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension, backend)
        kmc.run()

    