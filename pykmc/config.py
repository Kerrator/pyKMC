from dataclasses import dataclass
import configparser

@dataclass 
class Parameters : 
   """
   Physical parameters in lammps metal units
   """ 
   #Boltzmann constant
   kb = 8.6173303e-05 #eV.K^-1
   #Planck constant 
   h = 6.582119e-03 #eV.ps 



DEFAULT = {
    'Control' : {
        'catalog' : None,
        'dimension' : 3,
        'nprocs' : 1, 
        'backend' : 'local'

    }, 
    'Minimization' : {
        'style' : 'lammps'
    }, 
    'AtomicEnvironment' : {
        'radd_cna' : 0
    },
    'EventSearch' : {
        'emin_event' : 0.2,
        'emax_event' : 6, 
        'partn_dmax' : 6.0, 
        'partn_verbose' : 2, 
        'partn_ninit' : 2, 
        'partn_forc_thr' : 0.01,
        'partn_push_mode' : 'rad', 
        'partn_push_dist_thr' : 3.0, 
        'partn_push_step_size' : 0.4, 
        'partn_eigen_step_size' : 0.2, 
        'partn_lanczos_disp' : 0.0005,
        'partn_nsmooth' : 3, 
        'partn_nperp' : 5,
        'k0' : 1
    }, 
    'PSR' : {
        'kmax_factor' : 1.8
    }
}




@dataclass 
class SystemConfig : 
    """ 
    """ 
    #Default values : 

    @staticmethod 
    def from_file(config_file: str): 

        
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation()) 
        config.optionxform = str #Enable uppercase variable
        try : 
            with open(config_file) as f : #So we can raise an exception
                config.read(config_file)
        except :
            raise Exception('No configuration file found')
        
        #Check 
        if not config.has_section('Control')  : 
            raise Exception('Control section in configuration file is mandatory')
        if not config.has_section('Potential') : 
            raise Exception('Potential section in configuration file is mandatory')
        if not config.has_section('Minimization') : 
            raise Exception('Minimization section in configuration file is mandatory')
        if not config.has_section('EventSearch') : 
            raise Exception('EventSearch section in configuration file is mandatory')
        if not config.has_section('PSR') : 
            raise Exception('PSR section in configuration file is mandatory')


        
        #Convert ConfigParser in a dictionaries of dictionary and convert values 
            #Initialize with default values 
        config_dict = {section: default for section, default in DEFAULT.items()}
            #Add config values
        for section in config.sections() : 
            if section not in config_dict: 
                config_dict[section] = {}
            config_dict[section].update({
                    key : SystemConfig._convert_value(config.get(section,key))
                    for key, _ in config.items(section)
            })

        return config_dict
    
    def _convert_value(value) : 
        try : 
            return int(value)
        except ValueError : 
            pass 
        try : 
            return float(value) 
        except ValueError : 
            pass 
        return value

