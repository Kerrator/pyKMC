from typing import TypeAlias, TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np

"""
Construction of the Result Type is based on Rust/rustedpy and https://www.youtube.com/watch?v=1P7J2wI46sg
"""

# Construction of the Result Type : 

TOK = TypeVar("TOK")
TERR = TypeVar("TERR") 

class Ok(Generic[TOK]) : 

    _value: TOK 

    def __init__(self, value: TOK) : 
        self._value = value

    def is_ok(self) -> bool : 
        return True 
    
    def ok_value(self) -> TOK : 
        return self._value 
    
class Err(Generic[TERR]) : 
    
    _err = TERR 

    def __init__(self, err: TERR) : 
        self._err = err 

    def is_ok(self) -> bool : 
        return False 
    
    def err_value(self) -> TERR : 
        return self._err 
    
Result : TypeAlias = Ok[TOK] | Err[TERR]

@dataclass 
class ErrorInfo : 
    type : "ErrorType"
    message : str 
    details: Optional[str] = None
    variables: Optional[Dict[str, Any]] = None

class ErrorType(Enum) : 
    EVENT_NOT_FOUND = 1 
    EVENT_MINIMA_NOT_MATCH_POSITIONS = 2
    EVENT_NOT_WITHIN_ENERGY_LIMITS = 11 
    EVENT_NOT_NEW = 12



# Dataclass to store operation outputs 

@dataclass 
class EventSearchOutput : 
    central_atom_index : int 
    min1_positions : np.ndarray 
    saddle_positions : np.ndarray
    min2_positions : np.ndarray 
    dE_forward : float 
    dE_backward : float 
    move_atom_index : int 

@dataclass 
class KMCLoopInfo : 
    step : int = 0
    time : float = 0
    nb_visited_environments : int = 0
    nb_current_atomic_environments : int = 0
    nb_of_reference_event_searches : int = 0 
    size_reference_event_table : int = 0
    nb_applicable_reference_events : int = 0
    nb_of_refinements_attempts : int = 0 

    def print_informations(self) : 
        print("KMC Step n°{}".format(self.step))
        print("\t number of visited environment = {}".format(self.nb_visited_environments))
        print("\t size of the reference event table = {}".format(self.size_reference_event_table))
        print("\t number of current atomic environements = {}".format(self.nb_current_atomic_environments))
        print("\t number of reference events that can be applied = {}".format(self.nb_applicable_reference_events))
        print("\t time after event = {} ps".format(self.time))