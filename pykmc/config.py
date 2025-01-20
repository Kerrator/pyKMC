from dataclasses import dataclass
import configparser

@dataclass 
class DefaultConfig :
    """ 
    """ 


@dataclass 
class SystemConfig : 
    """ 
    """ 
    
    @staticmethod 
    def from_file(config_file: str): 

        
        config = configparser.ConfigParser() 
        try : 
            with open(config_file) as f : #So we can raise an exception
                config.read(config_file)
        except :
            raise Exception('No configuration file found')
        
        #Check if 
        if not config.has_section('Control')  : 
            raise Exception('Control section in configuration file is mandatory')

