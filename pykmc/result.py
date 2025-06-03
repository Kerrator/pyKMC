from typing import TypeAlias, TypeVar, Generic, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import yaml

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
    EVENT_ENERGY_HIGHER_THAN_THRESHOLD = 11 
    EVENT_ENERGY_LOWER_THAN_THRESHOLD = 12
    EVENT_BACKWARD_ENERGY_LOWER_THAN_THRESHOLD = 13
    EVENT_ASYMMETRIC = 14
    EVENT_NOT_NEW = 15
    PSR_NO_MATCH_FOUND = 21
    PSR_MATCHING_SCORE_ABOVE_ACCEPTANCE_THRESHOLD = 22
    REFINEMENT_INVALID_ENERGY_BARRIER = 31



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

#@dataclass 
#class EventRefinementOutput: 
#    central_atom_index : int 
#    min1_positions : np.ndarray 
#    saddle_positions : np.ndarray 
#    min2_positions : np.ndarray 
#    dE_foward : float 
#    dE_backward: float 

@dataclass
class PSROutput : 
    rotation_matrix: np.ndarray 
    translation_matrix: np.ndarray
    permutation_matrix: np.ndarray 
    matching_score : float

@dataclass
class AtomicEnvironmentInfo:
    """
    Store informations on atomic environments for one KMC step.

    Attributes
    ----------
    total_atomic_environments_encounter : int 
        Total unique atomic environments seen so far.
    n_current_atomic_environments : int  
        Number of environments in the current configuration.
    n_new_atomic_environments : int 
        Number of new environments discovered in the last step.
    atoms_grouped_by_environment : list[list[int]] 
        List of atom index groups sharing identical environments.
    """
    total_atomic_environments_encounter: int = 0
    n_current_atomic_environments: int = 0
    n_new_atomic_environments: int = 0
    atoms_grouped_by_environment: list[list[int]] = field(default_factory=list)

@dataclass
class ReferenceEventSearchInfo : 
    total_event_searches: int 
    n_success : int 
    n_fails : dict[str, int] 


@dataclass 
class KMCLoopInfo : 
    step : int = 0
    atomic_environment_info: AtomicEnvironmentInfo = None
    reference_event_searches_info: ReferenceEventSearchInfo = None

    def output_msg(self) : 
        cleaned = clean_dict(asdict(self))
        return yaml.dump(
            cleaned,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            explicit_start=True,
            Dumper=CustomDumper
        )


class CustomDumper(yaml.Dumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


# Custom representer to force inner lists to be in flow style
def represent_list_preserve_flow(dumper, data):
    if all(isinstance(i, int) for i in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data)


#custom representer for lists
CustomDumper.add_representer(list, represent_list_preserve_flow) 

def clean_dict(d):
    """Clean empty attribute/non initiate attribute"""
    if isinstance(d, dict):
        return {
            k: clean_dict(v)
            for k, v in d.items()
            if v not in (None, [], {}, '')
        }
    elif isinstance(d, list):
        return [clean_dict(v) for v in d if v not in (None, [], {}, '')]
    return d