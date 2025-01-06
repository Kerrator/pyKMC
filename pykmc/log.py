""" 
Logger class to handle log file and log parameters
"""
import logging 
from logging import handlers 

class Logger() : 
    def __init__ (self, logfile_name) : 
        self.logger = logging.getLogger('pykmc')
        logging.basicConfig(filename=logfile_name, filemode='a', level=logging.DEBUG, format='%(message)s')

    def title(self, log) : 
        log.logger.info("          |              ")
        log.logger.info(",---.,   .|__/ ,-.-.,---.")
        log.logger.info("|   ||   ||  \ | | ||    ")
        log.logger.info("|---'`---|`   `` ' '`---'")
        log.logger.info("|    `---              ")
        log.logger.info("\n")
