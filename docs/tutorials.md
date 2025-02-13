# Basic KMC simulation 

This tutorial provides a first overview of pyKMC's features and explains how to run a simple simulation.

We will use an FCC Ni system with a vacancy and basic input parameters to see how the algorithm works.





## Input file 

To run a KMC simulation, an input file, with a format readable by [configparser](https://docs.python.org/3/library/configparser.html) is used. 

It is separated in different sections, each one controlling a part of the KMC simulation. 

- The [Control] section : 
	This section is used to define general KMC parameters and resources. 
- The [Potential] section : 
	To define the potential used by the E/F engine (e.g. Lammps) 
- The [Minimization] section : 
	defines parameters related to the minimization of the system done a each KMC step. 
- The [AtomicEnvironment] section : 
	pyKMC 
