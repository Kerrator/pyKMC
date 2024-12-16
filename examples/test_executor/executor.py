from executorlib import Executor 

def calc(i):
    from mpi4py import MPI

    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    return i, size, rank


with Executor(backend="local") as exe:
    fs = exe.submit(calc, 3, resource_dict={"cores": 2})
    print(fs.result())

