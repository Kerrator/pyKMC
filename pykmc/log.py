""" 
Logger class to handle log file and log parameters
"""
import sys
import logging 
import logging.config
from logging import handlers 
from itertools import zip_longest
from typing import Any
from enum import Enum
import re


class LogManager:
    """
    Manages the configuration and usage of multiple standard Python loggers.
    It is setup via dictionary configuration and provides convenience
    methods for sending messagers to specific loggers.

    Attributes
    ----------
    config_dict (dict[str, Any], optional): 
        Configuration dictionary for loggers in the format expected by logging.config.dictConfig.
        No configuration is applied if None.
    """

    def __init__(self, config_dict: dict[str, Any] | None =None) -> None:
        self._logger = {} # Stores logger instances by name
        #Configure loggers
        if config_dict is not None:
            self.configure_from_dict(config_dict)

    def configure_from_dict(self, config_dict: dict[str, Any]) -> None :
        """Configures logger instances from a dictionary.

        Parameters
        ----------
        config_dict : dict[str, Any]
            Dictionary with the loggers settings

        Raises
        ------
        Exception
            If loggers configuration fails.
        """
        try:
            logging.config.dictConfig(config_dict)
            #Retrieve and store configured loggers
            for logger_name in config_dict.get("loggers", {}):
                self._logger[logger_name] = logging.getLogger(logger_name)

        except Exception as e:
            raise Exception("Unable to setup loggers from dictionary")

    def _get_active_logger(self, logger_name: str) -> logging.Logger:
        """Retrieves a logger instance by its name.

        Parameters
        ----------
        logger_name : str
            Logger name.

        Returns
        -------
        logging.Logger
            The requested logger instance.

        Raises
        ------
        Exception
            ValueError: If the logger name is not configured within this LogManager.
        """
        if logger_name not in self._logger:
            raise ValueError(f"Logger '{logger_name}' is not configured.")
        else:
            logger = self._logger.get(logger_name)

        return logger

    def debug(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """Logs a message with level DEBUG using the specified logger.
        This is a convenience method that wraps the underlying `logging.Logger.debug()` call.
        Refer to `logging.Logger.debug()` documentation for `*args` and `**kwargs` usage.

        Parameters
        ----------
        logger_name : str
            Logger name.
        msg : str
            Log message.
        """
        self._get_active_logger(logger_name).debug(msg, *args, **kwargs)

    def info(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Similar to `debug()`, but for the INFO level.
        """
        self._get_active_logger(logger_name).info(msg, *args, **kwargs)
        
    def warning(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Similar to `debug()`, but for the WARNING level.
        """
        self._get_active_logger(logger_name).warning(msg, *args, **kwargs)

    def error(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Similar to `debug()`, but for the ERROR level.
        """
        self._get_active_logger(logger_name).error(msg, *args, **kwargs)

    def critical(self, logger_name: str, msg: str, *args: Any, **kwargs: Any) -> None:
        """
        Similar to `debug()`, but for the CRITICAL level.
        """
        self._get_active_logger(logger_name).critical(msg, *args, **kwargs)

class Colors(Enum) : 
    """
    An enumeration of ANSI escape codes for common text colors and styles.
    Access the color string using the .value attribute (e.g., AnsiColors.RED.value).
    """
    RESET = "\x1b[0m"
    BOLD = "\x1b[1m"
    WHITE = "\x1b[37m"
    RED = "\x1b[31m"
    GREEN = "\x1b[32m"
    YELLOW = "\x1b[33m"
    BLUE = "\x1b[34m"

class CustomFormatter(logging.Formatter):
    """Custom Formatter to display different style messages depending on their level.
    """
    def format(self, record: logging.LogRecord) -> str:
        """Dynamically formats log records based on their severity level.
        For INFO and DEBUG level records, only the log message is displayed.
        For other levels (WARNING, ERROR, CRITICAL), the level name is
        included, formatted in red for easy visibility, followed by the message.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be formatted

        Returns
        -------
        str
            The formatted message.
        """
        if record.levelno == logging.INFO or record.levelno == logging.DEBUG :
            self._style._fmt = "%(message)s"
        else : 
            self._style._fmt = f"{Colors.RED.value}%(levelname)-12s{Colors.RESET.value} | %(message)s"
        return super().format(record)

class ProgressHandler(logging.StreamHandler) : 
    def __init__(self, stream=sys.stdout):
        super().__init__(stream)
    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.stream.write('\r' + msg)
            self.stream.flush()
            #msg = self.format(record)
            #stream = self.stream
            #stream.write('\u001b[1000D' + msg + '\u001b[0K')
            #stream.write
            #self.flush()
        except Exception:
            self.handleError(record)

ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
class AnsiStrippingFormatter(CustomFormatter):
    """
    Custom Formatter to remove Ansi caracter. 
    Used when logging a colored message in the stdout and a file. 
    """
    def format(self, record: logging.LogRecord) -> str :
        """Formats log records to remove Ansi caracters.

        Parameters
        ----------
        record : logging.LogRecord
            The log record to be fromatted.
        Returns
        -------
        str
            The formatted message.
        """
        formatted_message = super().format(record)
        clean_message = ANSI_ESCAPE_PATTERN.sub('', formatted_message)
        clean_message = clean_message.replace('\r', '')
        return clean_message


LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {  
        "default_formatter": { 
            "()" : CustomFormatter,
        },
        "file_formatter": { 
            "()": AnsiStrippingFormatter,  
        }
    },
    "handlers": {  
        "log_file": {
            "class": "logging.FileHandler",
            "formatter": "file_formatter",
            "level": "DEBUG",
            "filename": "pykmc.log",
        },
        "console_output_handler": {
            "class": "logging.StreamHandler",
            "formatter": "default_formatter",
            "level": "DEBUG",
            "stream": "ext://sys.stdout",
        },
        "general_output_file" : {
            "class" : "logging.FileHandler", 
            "formatter" : "file_formatter", 
            "level" : "DEBUG", 
            "filename" : "pykmc.out"
        }, 
        "step_informations" : {
            "class" : "logging.FileHandler", 
            "formatter" : "file_formatter", 
            "level" : "DEBUG", 
            "filename" : "pykmc.info"
        }, 
        "progress_bar_handler": { 
            "class": "pykmc.log.ProgressHandler",
            "formatter": "default_formatter",
            "level": "INFO", 
            "stream": "ext://sys.stdout",
    } } ,
    "loggers": {  
        "log": {  
            "handlers": ["log_file", "console_output_handler"],
            "propagate": False,  
        },
        "output": {  
            "handlers": ["general_output_file"],  
        },
        "info": { 
            "handlers" : ["step_informations"]
        },
        "progress" : {
            "handlers" : ["log_file", "progress_bar_handler"],
        }
    },
}


class LogKMC(LogManager) : 
    """Manages logging for the KMC, offering dynamic verbosity control.

    Extends `LogManager` to adjust logging levels for specified loggers
    (e.g., 'log', 'output') based on a simple verbosity setting (0-2).
    Provides convenience methods for KMC-specific log messages.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary for loggers in the format expected by logging.config.dictConfig.
    verbosity : int, optional
        Defines the loggers level (0=WARNING, 1=INFO, 2=DEBUG). Defaults to 1.
    """
    def __init__(self, config_dict: dict[str, Any], verbosity: int = 1):
        super().__init__(config_dict)
        self._verbosity = verbosity 
        #apply verbosity option modifying logger and handlers level
        self._apply_verbosity_level()

    def _apply_verbosity_level(self):
        """Modify loggers and their handlers levels.

        Raises
        ------
        ValueError
           if verbosity value is not 0, 1 or 2. 
        """
        if self._verbosity == 0:
            level = logging.WARNING
        elif self._verbosity == 1:
            level = logging.INFO
        elif self._verbosity == 2:
            level = logging.DEBUG
        else : 
            raise ValueError("verbosity should be 0, 1 or 2")

        for logger_name in self._logger : 
            logger = self._get_active_logger(logger_name)
            logger.setLevel(level)
            if logger_name == "progress" : #To pass debug level for progress bar 
                logger.setLevel(logging.DEBUG)
            for handler in logger.handlers : 
                if logger_name == "progress" and isinstance(handler, ProgressHandler):
                    handler.setLevel(logging.DEBUG) #always display bar in stdout bug only debug level for log_file
                else : 
                    handler.setLevel(level) 

    def title(self, logger_name: str) -> None : 
        """Display pyKMC title to the logger

        Parameters
        ----------
        logger_name : str
            the logger name.
        """
        self.info(logger_name, "          |              ")
        self.info(logger_name, ",---.,   .|__/ ,-.-.,---.")
        self.info(logger_name, "|   ||   ||  \ | | ||    ")
        self.info(logger_name, "|---'`---|`   `` ' '`---'")
        self.info(logger_name, "|    `---              ")
        self.info(logger_name, "\n")

    def write_parameter(self, logger_name: str, config: dict[str, Any], width: int = 60, indent: int = 4) -> None: 
        """Write simulation parameter to the logger.

        Parameters
        ----------
        logger_name : str
            The logger name.
        config : dict[str, Any]
            The configuration dictionnary.
        width : int, optional
            Width of the lines, by default 60
        indent : int, optional
            How many space define the indentation, by default 4
        """
        centered_title = f"SIMULATION PARAMETERS".center(width)
        separator = "=" * width
        self.info(logger_name, "\n{}\n{}\n{}".format(separator, centered_title, separator))

        max_key_len = max(len(key) for section in self.config.values() for key in section)

        for section in config : 
            self.info(logger_name, section)
            for key, value in config[section].items() : 
                self.info(logger_name, "{}{:<{}} : {}".format(" " * indent, key, max_key_len, value))
        self.new_line()

    def output_file_header(self, logger_name: str) -> None : 
        """
        Write the header of the output file.

        Parameters
        ----------
        logger_name: str 
            The logger name.
        """     
        #Information header : 
        self.info(logger_name, '# Simulation Progress Tracking File') 
        self.new_line(logger_name) 
        self.info(logger_name, '# Column Details:')
        self.info(logger_name, '\t# Step          : Simulation step number.')
        self.info(logger_name, '\t# dT(s)         : Time elapsed for this specific step.')
        self.info(logger_name, '\t# T(s)          : Total time since simulation start.')
        self.info(logger_name, '\t# Ea(eV)        : Event activation energy barrier.')
        self.info(logger_name, '\t# k_evt(ps-1)   : Rate constant of the selected event.')
        self.info(logger_name, '\t# k_tot(ps-1)   : Total rate constant of all possible events at this step')
        self.info(logger_name, '\t# E(eV)         : Energy of the system.')
        self.new_line(logger_name)
        #First line of the table
        self.info(logger_name, '{:<9s} {:<8s} {:<8s} {:<10s} {:<14s} {:<14s} {:<10s}'.format('Step', 'dT(s)', 'T(s)',  'Ea(eV)',  'k_evt(ps-1)',  'k_tot(ps-1)', 'E(eV)'))
        self.info(logger_name,'{:s}'.format(80*'-') )

    def table_line_info_kmc(self, logger_name: str, *args: int | float) : 
        """Writes a formatted line of simulation output values into the output table.

        Parameters
        ----------
        logger_name : str 
            The logger name. 
        *args : int | float 
            Values representing the columns of the simulation output progress table.
            These should correspond to:
                - Step Number
                - Time elapsed for this step (dT in s)
                - Total cumulative time (T in s)
                - Event energy barrier (Ea in eV)
                - Rate constant of current event (k_evt in ps-1)
                - Total rate constant of all events (k_tot in ps-1)
                - Total energy (E in eV)

        """
        formats = ['{:<9n}', '{:<8e}', '{:<8e}', '{:<10e}', '{:<14e}', '{:<14e}', '{:<10e}']
        formatted_values = [fmt.format(e) if e is not None else " "*10 for fmt, e in zip_longest(formats, args, fillvalue=None)]
        self.info(logger_name, " ".join(formatted_values))

    def new_line(self, logger_name: str) -> None: 
        """Write a new line in the logger

        Parameters
        ----------
        logger_name : str
           The logger name. 
        """
        self.info(logger_name, "")

    def progress_bar(self, logger_name: str, current_step: int, total_steps: int, bar_length: int = 40) -> None : 
        """Displays a progression bar

        Parameters
        ----------
        logger_name : str
            _description_
        current_step : int
            _description_
        total_steps : int
            _description_
        bar_length : int, optional
            _description_, by default 40
        """
        #Compute percentage
        percent = (current_step / total_steps) * 100
    
        static_text_color = Colors.WHITE.value 
        # Dynamical bar colors
        bar_fill_color = Colors.WHITE.value 

        if percent < 30:
            bar_fill_color = Colors.RED.value 
        elif percent < 70:
            bar_fill_color = Colors.YELLOW.value 
        else:
            bar_fill_color = Colors.GREEN.value

        # bar fill lenght 
        filled_length = int(bar_length * current_step / total_steps)
        # all bar 
        bar_segment = '#' * filled_length + '-' * (bar_length - filled_length)
        progress_message = (
            f"{static_text_color}\rProgression: " 
            f"{bar_fill_color}[{bar_segment}]{Colors.RESET.value}" 
            f"{static_text_color} {percent:.1f}% "  
        ) 
    
        # Envoi du message via le logger
        self.debug(logger_name, progress_message)