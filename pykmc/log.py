""" 
Logger class to handle log file and log parameters
"""
import logging 
from logging import handlers 

class Logger() : 
    #TODO : See how to add verbosity
    def __init__ (self, logfile_name) : 
        #Could use different logger for different level 
        #See https://dev.to/luca1iu/using-the-logger-class-in-python-for-effective-logging-4ghc#:~:text=The%20Logger%20class%20provides%20several,warning%2C%20error%2C%20crit).
        self.logger = logging.getLogger('pykmc')
        logging.basicConfig(filename=logfile_name, filemode='a', level=logging.DEBUG, format='%(message)s')

    def title(self, log) : 
        """ 
        Head of the log file 
        """
        log.logger.info("          |              ")
        log.logger.info(",---.,   .|__/ ,-.-.,---.")
        log.logger.info("|   ||   ||  \ | | ||    ")
        log.logger.info("|---'`---|`   `` ' '`---'")
        log.logger.info("|    `---              ")
        log.logger.info("\n")

