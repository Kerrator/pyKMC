""" 
Logger class to handle log file and log parameters
"""
import logging 
from logging import handlers 
from itertools import zip_longest

class Logger() : 
    """A logger to manage log informations
    """
    def __init__ (self, config) : 
        self.config = config
        self.logger = logging.getLogger('pykmc')
        #logging.basicConfig(filename=self.config['Control']['log_file_name'], filemode='a', level=logging.DEBUG, format='%(message)s')

        #Configuration : 
        self.logger.setLevel(logging.DEBUG) 
        formatter = CustomFormatter()
        level_map = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }
        log_level = level_map.get(self.config['Control']['verbosity'], logging.INFO)
        file_handler = logging.FileHandler(self.config['Control']['log_file_name'], mode='a')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

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

    def write_parameter(self, width = 60, indent = 4) : 
        centered_title = f"SIMULATION PARAMETERS".center(width)
        separator = "=" * width
        self.logger.info("\n{}\n{}\n{}".format(separator, centered_title, separator))

        max_key_len = max(len(key) for section in self.config.values() for key in section)

        for section in self.config : 
            self.logger.info(section)
            for key, value in self.config[section].items() : 
                self.logger.info("{}{:<{}} : {}".format(" " * indent, key, max_key_len, value))
            


        self.new_line()

    def first_line_table(self) : 
        """
        First line of the log table

        Parameters
        ----------
        reconstruction : boolean
            if we use the reconstruction of event during the KMC simulation or not
        """       
        reconstruction = self.config['Control']['reconstruction'] 
        if reconstruction : 
            self.logger.info('{:<10s} {:<12s} {:<10s} {:<10s} {:<14s} {:<11s}'.format('Step', 'Time(s)', 'Ndiff_env', 'N_event', 'dE_event(eV)', 'k(ps-1)'))
        else : 
            self.logger.info('{:<10s} {:<12s} {:<10s} {:<10s} {:<13s} {:<10s}'.format('Step', 'Time(s)', 'Ndiff_env', 'N_event', 'n_select_event', 'dE_event'))

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



class CustomFormatter(logging.Formatter):
    def format(self, record):
        """ 
        Rewrite format function of Formatter, change _fmt when writing
        """
        if record.levelno == logging.INFO:
            self._style._fmt = "%(message)s"
        else:
            self._style._fmt = "%(levelname)-12s | %(message)s"
        return super().format(record)
