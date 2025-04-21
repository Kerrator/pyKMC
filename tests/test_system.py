import pytest 
from pytest_lazy_fixtures import lf
from pykmc import System 
import numpy as np
print(pytest)
class TestSystem() : 

    @pytest.mark.parametrize("system", [lf("system_single_type")])
    def test_update_all_positions_nowrap(self, system: System) : 
        new_positions = np.array([[1.0, 1.0, 0.0], [1.5, 3.4, 2.4], [2.3, 0.3, 0.8 ], [2.6, 1.4, 1.4]])
        system.update_positions(new_positions)
        np.testing.assert_allclose(system.positions, new_positions, rtol=1e-6)
