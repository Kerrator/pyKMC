from dataclasses import dataclass
import configparser

@dataclass 
class SystemConfig : 
    """ 
    """ 
    #Default values : 

    @staticmethod 
    def from_file(config_file: str): 

        
        config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation()) 
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
        
        #Convert ConfigParser in a dictionaries of dictionary and convert values 
        config_dict = {}        
        for section in config.sections() : 
            config_dict[section] = {
                    key : SystemConfig._convert_value(config.get(section,key))
                    for key, _ in config.items(section)
            }

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

