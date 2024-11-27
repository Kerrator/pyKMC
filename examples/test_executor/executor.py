from executorlib import Executor 

with Executor(backend="slurm_allocation") as exe:
    future = exe.submit(sum, [1, 1])
    print(future.result())

