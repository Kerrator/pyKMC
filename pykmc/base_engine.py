from abc import ABC, abstractmethod 
class BaseEngine(ABC) : 

    @abstractmethod 
    def minimize(self, system) : 
        pass 

    def compute_distances(self, system) : 
        pass

    def neighbors(self, system) : 
        pass


