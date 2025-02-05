""" 
Logger class to handle log file and log parameters
"""
import logging 
from logging import handlers 
from itertools import zip_longest

class Logger() : 
    def __init__ (self, logfile_name) : 
        #Could use different logger for different level 
        #See https://dev.to/luca1iu/using-the-logger-class-in-python-for-effective-logging-4ghc#:~:text=The%20Logger%20class%20provides%20several,warning%2C%20error%2C%20crit).
        self.logger = logging.getLogger('pykmc')
        logging.basicConfig(filename=logfile_name, filemode='a', level=logging.DEBUG, format='%(message)s')

    def title(self) : 
        """ 
        Head of the log file 
        """
        self.logger.info("          |              ")
        self.logger.info(",---.,   .|__/ ,-.-.,---.")
        self.logger.info("|   ||   ||  \ | | ||    ")
        self.logger.info("|---'`---|`   `` ' '`---'")
        self.logger.info("|    `---              ")
        self.logger.info("\n")

    def first_line_table(self, reconstruction) : 
        """
        First line of the log table

        Parameters
        ----------
        reconstruction : boolean
            if we use the reconstruction of event during the KMC simulation or not
        """        
        if reconstruction : 
            self.logger.info('{:<10s} {:<12s} {:<10s} {:<10s} {:<14s} {:<11s} {:<12s} {:<18s} {:<18s}'.format('Step', 'Time', 'Ndiff_env', 'N_event', 'n_select_event', 'dE_event', 'dh', 'Recontruction dE', 'Reconstruction Topo'))
        else : 
            self.logger.info('{:<10s} {:<12s} {:<10s} {:<10s} {:<13s} {:<10s}'.format('Step', 'Time', 'Ndiff_env', 'N_event', 'n_select_event', 'dE_event'))

    def table_line_info_kmc(self, *args) : 
        """
        Print info of a kmc step in the log 
        """ 
        formats = ['{:<10n}', '{:<10e}', '{:<10n}', '{:<10n}', '{:<13n}', '{:<10e}', '{:<10e}', '{:<18}', '{:<18}']
        formatted_values = [fmt.format(e) if e is not None else " "*10 for fmt, e in zip_longest(formats, args, fillvalue=None)]
        self.logger.info(" ".join(formatted_values))

    def new_line(self) : 
        """ 
        New line in the log file
        """ 
        self.logger.info("")

