from typing import TypeAlias, TypeVar, Generic, Optional
from dataclasses import dataclass
from enum import Enum

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
    details : str 

class ErrorType(Enum) : 
    EVENT_NOT_FOUND = 1 
    EVENT_MINIMA_NOT_MATCH_POSITIONS = 2