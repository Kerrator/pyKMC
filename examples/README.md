# Examples : 

This directory contains example atomistic systems used to test and illustrate the code.  

Each example is provided as a separate subdirectory.

## How to run an example

Each example directory contains an `input.in.template` file.

To run a simulation:
1. Copy the template input file:
```bash
cp input.in.template input.in
``` 
`input.in` files are user-specific and are not tracked. Do not modify input.in.template.

2. Edit ```input.in``` if needed. 
3. Run the simulation. Default template values are intended to be run in parallel with mpi using at least 8 cores. For example : 
```bash
mpirun -n 8 python -m pykmc -in input.in 
``` 

# Nickel systems : 

Nickel systems are the simplest examples provided. 
Nickel has relatively weak elastic effects and is well suited for basic testing and validation 

- `./Ni_fcc_2047at_monovacancy` : 
Simplest example, system with 1 vacancy. 

- `./Ni_fcc_4001at_sia`: 
Isolated SIA, test of intersitial handling and symmetries.

- `./Ni_fcc_4000at_monovacancy+sia` : 
Vacancy-SIA recombination test.


# Iron system : 

- `Fe_bcc_6746at_bivacancies` : 
4 vacancies in a 2x1NN initial configuration. Well suited to study basins and flickering.

# Copper system : 

- `Cu_fc `: 
Vacancy-SIA recombination test. Stronger elastic effects than in Nickel systems.
