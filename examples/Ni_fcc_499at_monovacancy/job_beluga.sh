#!/bin/bash
#SBATCH --account=def-mousseau
#SBATCH --ntasks=1               # nombre de processus MPI
#SBATCH --cpus-per-task=10
#SBATCH --time=0-00:2

module load python/3.12.4  
module load mpi4py/4.0.0   
	
source /home/hmoison/pythonenvs/kart/bin/activate
export PYTHONPATH=$PYTHONPATH:/home/hmoison/programs/IterativeRotationsAssignments/interface:/home/hmoison/programs/artn-plugin/interface:/home/hmoison/myprojects/pyKMC

python run.py
