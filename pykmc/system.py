from ase import Atoms
from ase.io import read
import pandas as pd
from .log import Logger
from .config import SystemConfig
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
    kmc_traj : List[Ase Atoms Object], optional
        trajectory from a previous simulation (for restart), by default None

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
    _initialize_log()
        initialize the pykmc.log logfile 

    Examples 
    -------- 
    >>> system = System('config.xyz')
    >>> system.kmc(kmc_parameters, minimization_parameters, search_parameters, potential)
    """

    def __init__(self, input_path=None, config_file=None):
        #Setup logfile
        try : 
            os.remove('pykmc.log')
        except OSError : 
            pass 
        self.logger = self._initialize_log()
        self.logger.logger.info('= SYSTEM INITIALIZATION')

        if input_path == None and config_file == None : 
            raise Exception("Need input file or configuration file to initialize the system")

        #Read input file : 
        if input_path is not None : 
            self.inputs = SystemConfig.from_file(input_path)
            config_file = self.inputs['Control'].get('config_file')

        #Read initial config file
        self.logger.logger.info('reading {} configuration file'.format(config_file))
        atoms = read(config_file)  # Load configurations from file 
        super().__init__(symbols=atoms.get_chemical_symbols(),
                         positions=atoms.get_positions(),
                         cell=atoms.get_cell(),
                         pbc=atoms.get_pbc())



        self.environment = None
        self.catalog = None
        self.kmc_traj = None
#        if catalog == None : #Create empty DataFrame
#            self.catalog = pd.DataFrame(columns=['event_id', 
#                                                 'initial_positions', 
#                                                 'saddle_positions', 
#                                                 'final_positions', 
#                                                 'energy_barrier', 
#                                                 'k', 
#                                                 'move_atom_idx', 
#                                                 'id_saddle'])
#        else : #Read previous catalog DataFrame
#            self.logger.logger.info('reading {} catalog file'.format(catalog))
#            self.catalog = pd.read_pickle(catalog) #for restart
#        
#        self.kmc_traj = kmc_traj 
        self.logger.new_line()

    def minimize(self, minimization_style, minimization_params, potential, dimension=3, nprocs=1, backend='local') : 
        """
        Minimize the system and upadate the positions

        Parameters
        ----------
        minimization_style : str
            minimization style used, e.g. 'lammps'
        minimization_params : dict
            all commands needed by the program used in minimization_style to execute the minimization
        potential : dict
            commands to define the potential used by the program defined by minimization_style
        dimension : int, optional
            dimension of the system, by default 3
        nprocs : int, optional
            number of procs available, by default 1
        backend : str, optional
            parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

        Example 
        ------- 
        >>> minimization_params = {'min_style' : 'cg',
                                   'minimize'  : '1.0e-6 1.0e-8 100 1000'} 
        >>> potential = {'pair_style' : 'eam/alloy', 
                         'pair_coeff' : '* * Ni.eam Ni'}
        >>> system.minimize('lammps', minimization_params, potential)
        """        
        from .minimization import Minimization 
        minimizer = Minimization(self, minimization_style, minimization_params, potential, dimension, nprocs, backend)
        minimizer.run()

    def find_environment(self, environment_style, environement_params, dimension=3, nprocs=1, backend='local') : 
        """
        Find atomic environment ID for each atoms in System and update System.environment

        Parameters
        ----------
        environment_style : str
            style used to define the atomic environment, could be 'cna', 'graph', 'cna/graph'
        environement_params : dict of str: float
            dictionaty of radius parameters defining the environment, 'rnei' : radius cutoff to define
            neirest neighbors, 'rcut' : radius cutoff to define the environment (for 'graph')
        dimension : int, optional
            dimension of the system, by default 3
        nprocs : int, optional
            number of procs available, by default 1
        backend : str, optional
            parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

        Example 
        ------- 
        >>> atomicenv_params = {'rnei': 3.01,
                                'rcut'  : 6.0}    
        >>> system.find_environment('cna/graph', atomicenv_params)
        """        
        from .atomic_environment import AtomicEnvironment 
        atomic_environment = AtomicEnvironment(self, environment_style, environement_params, dimension, nprocs, backend)
        atomic_environment.run()

    def event_search(self, search_style, search_params, potential, dimension=3, nprocs=1, backend='local') : 
        """
        For each atomic environment ID that are not in catalog (except 'crist') launch 'nsearch' event search 
        and add new events to the catalog

        Parameters
        ----------
        search_style : str
            event search style used, can be 'pARTn'
        search_params : dict of str: str
            all commands needed by the program defined by search_style to run an event search
        potential : dict of str: str
            all commands needed by the program that is used by search_style to compute forces
        dimension : int, optional
            dimension of the system, by default 3
        nprocs : int, optional
            number of procs available, by default 1
        backend : str, optional
            parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

        Example 
        -------
        >>> potential = {'pair_style' : 'eam/alloy', 
                         'pair_coeff' : '* * Ni.eam Ni'}
        >>> search_params = {'nsearch' : 10, 
                            'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so'}
        >>> system.event_search('pARTn', search_params, potential)
        """        
        from .event import EventSearch 
        event_search = EventSearch(self, search_style, search_params, potential, dimension, nprocs, backend)
        event_search.run()
    
    def point_set_registration(self, psr_style, psr_parameters, idx_cat,central_atom_index, rcutevent, dimension=3, nprocs=1, backend='local') : 
        """
        run a point set registration between the central_atom_index in the system and the idx_cat event of the catalog

        Parameters
        ----------
        psr_style : str
            style used for the point set registration, can be 'ira'
        idx_cat : int
            index of the event in the catalog
        central_atom_index : int
            index of atom system on which we perform the point set registration
        rcutevent : float
            radius cutoff defining the subsystem that we save in the catalog
        dimension : int, optional
            dimension of the system, by default 3
        nprocs : int, optional
            number of procs available, by default 1
        backend : str, optional
            parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

        Example
        ------- 
        >>> system.point_set_registration('ira', 0, 250, 10.0)
        """
        from .point_set_registration import PointSetRegistration 
        psr = PointSetRegistration(self, psr_style, psr_parameters, idx_cat,central_atom_index, rcutevent, dimension, nprocs, backend)
        rmat, tr, perm, dh = psr.run()
        return rmat, tr, perm, dh

    #def kmc(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension=3, backend='local') :
    def kmc(self)  : 
        """
        Run kinetic monte carlo simulation

        Parameters
        ----------
        kmc_parameters : dict
            dictionary of parameters related to the kinetic monte carlo simulation
        minimization_params : dict
            see the minimize method
            for the moment kmc use the minimization_style 'lammps'(hardcoded)
        atomenv_params : dict
            see the find_environment method 
            for the moment kmc use the style 'cna/graph' (hardcoded)
        eventsearch_params : _type_
            see the event_search method 
            for the moment kmc use the style 'pARTn' (hardcoded)
        potential : dict of str: str
            all commands needed by the program that is used by search_style to compute forces
        dimension : int, optional
            dimension of the system, by default 3
        backend : str, optional
            parameter used by Executorlib, can be 'local', 'slurm_allocation', 'slurm_submission', by default 'local'

        Example
        ------- 
        >>> potential = {'pair_style' : 'eam/alloy', 
                         'pair_coeff' : '* * ./Ni_v6_2.0_LKBeland2016.eam Ni'} 
        >>> minimization = {'min_style'     : 'cg', 
                            'minimize '     : '1.0e-6 1.0e-8 100 1000'}
        >>> atomenv = {'rnei' : 3.01, 
                       'rcut' : 5.0}
        >>> search_params = {'nsearch' : 10, 
                             'path_artnso' : '/root/programs/artn-plugin/lib/libartn-lmp.so'}
        >>> kmc_parameters = {'nkmc_steps' : 50}
        >>> system.kmc(kmc_parameters, minimization,atomenv, search_params, potential)
        """
        from .kmc import KMC 
        #kmc = KMC(self, kmc_parameters, minimization_params, atomenv_params, eventsearch_params, potential,dimension, backend)
        kmc = KMC(self)
        kmc.run()

    def _initialize_log(self) : 
        """ 
        Initialize the log file and print title 
        """ 
        logger = Logger('pykmc.log')
        logger.title()
        return logger 

    