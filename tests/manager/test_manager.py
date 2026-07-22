from mpi4py import MPI
from pykmc.manager import ManagerFactory
import pytest


def _require_8_ranks():
    if MPI.COMM_WORLD.Get_size() != 8:
        pytest.skip("Manager tests must be ran with mpirun -n 8.")


class Operations:

    def __init__(self, comm) : 
        self.comm = comm

    @property
    def _is_rank0(self) -> bool:
        return self.comm is None or self.comm.Get_rank() == 0

    def parallel_add(self, a, b):
        local = a + b
        result = self.comm.gather(local, root=0)
        if self._is_rank0:
            return result

    def parallel_multiplication(self, a, b):
        local = a*b
        result = self.comm.gather(local, root=0)
        if self._is_rank0:
            return result
    
def division(comm, a, b):
    local = a/b
    result = comm.gather(local, root=0)
    if comm.Get_rank() == 0:
        return result


class TestsManager:

    @pytest.fixture(autouse=True)
    def setup(self):
        _require_8_ranks()
        MPI.COMM_WORLD.Barrier()
        comm = MPI.COMM_WORLD
        self.manager = ManagerFactory(
            obj_factory=lambda comm, _: Operations(comm),
            n_workers=4,
            comm=comm,
            has_global=True,
            group_size=6,
            extra_ops={"division": division}
        ).launch()
        yield
        if self.manager is not None:
            self.manager.shutdown()
        MPI.COMM_WORLD.Barrier()

    def test_check_available_operations(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        ops = self.manager.list_ops()
        assert set(ops) == {"parallel_add", "parallel_multiplication", "division"}

    def test_parallel_add(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        expected = 2 + 3

        result = self.manager.parallel_add(a=2, b=3).result()
        assert result is not None and all(v == expected for v in result)

        result = self.manager.group_parallel_add(a=2, b=3)
        assert result is not None and all(v == expected for v in result)

        result = self.manager.global_parallel_add(a=2, b=3)
        assert result is not None and all(v == expected for v in result)

    def test_parallel_multiplication(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        expected = 3 * 4

        result = self.manager.parallel_multiplication(a=3, b=4).result()
        assert result is not None and all(v == expected for v in result)

        result = self.manager.group_parallel_multiplication(a=3, b=4)
        assert result is not None and all(v == expected for v in result)

        result = self.manager.global_parallel_multiplication(a=3, b=4)
        assert result is not None and all(v == expected for v in result)

    def test_division(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        expected = 10 / 2

        result = self.manager.division(a=10, b=2).result()
        assert result is not None and all(v == expected for v in result)

        result = self.manager.group_division(a=10, b=2)
        assert result is not None and all(v == expected for v in result)

        result = self.manager.global_division(a=10, b=2)
        assert result is not None and all(v == expected for v in result)


class TestsManagerValidation:

    @pytest.fixture(autouse=True)
    def setup(self):
        _require_8_ranks()

    def _base_factory(self, **overrides):
        kwargs = dict(
            obj_factory=lambda *_: None,
            n_workers=4,
            comm=MPI.COMM_WORLD,
            has_global=False,
        )
        kwargs.update(overrides)
        return kwargs

    def test_not_enough_ranks(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        with pytest.raises(ValueError, match="Not enough MPI ranks"):
            ManagerFactory(**self._base_factory(n_workers=10))

    def test_group_size_not_multiple_of_local(self):
        if MPI.COMM_WORLD.Get_rank() != 0:
            return
        with pytest.raises(ValueError, match="must be a multiple of the per-worker rank count"):
            ManagerFactory(**self._base_factory(group_size=3))
