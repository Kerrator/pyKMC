import pytest
from pytest_lazy_fixtures import lf

from pykmc import AtomicEnvironment, NeighborsList


@pytest.mark.parametrize(
    "system, config", [(lf("system_single_type_fcc_vacancy"), lf("config_system_single_type"))]
)
def test_threshold_flows_from_config_object(system, config):
    """Constructing the environment the way the run sites do must honour coordination_threshold."""
    config.atomicenvironment.style = "coordination"
    config.atomicenvironment.coordination_threshold = 12
    nl = NeighborsList(system, config.atomicenvironment.rnei, config.atomicenvironment.rcut)
    ae = AtomicEnvironment(
        config.atomicenvironment.style,
        nl.neighbors_list["rnei"],
        nl.neighbors_list["rcut"],
        config.atomicenvironment.neighbors_add,
        coordination_threshold=config.atomicenvironment.coordination_threshold,
    )
    assert ae.coordination_threshold == 12
    assert sum(e == "noncrystal" for e in ae.atomic_environment_list) == 12
