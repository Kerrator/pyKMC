#for pARTn
export PYTHONPATH=$PYTHONPATH:/root/programs/artn-plugin/interface
#for IRA
export PYTHONPATH=$PYTHONPATH:/root/programs/IterativeRotationsAssignments-master/interface

#to run on root (docker) mpi
export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1


export PYTHONFAULTHANDLER=1
time python3 run.py
