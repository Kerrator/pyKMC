"""The service's species map is authoritative or derived per request, never frozen.

With an authoritative ``species_masses`` (the preflight map) every request
must be described by that map and an unknown symbol is a named error listing
the symbols and the map. Without one (the offline/test path) the descriptor
is derived from each request's own full-system types through the one species
rule, so a later request introducing a new species widens the map instead of
being rejected or mis-mapped by the first batch's species.
"""

from pathlib import Path

import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.engine.lammps import species_map
from pykmc.htst.request import HTSTRequestError
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService

_INPUT = Path(pykmc.__file__).resolve().parent.parent / "tests" / "data" / "input.in"


def htst_config() -> Config:
    base = Config.from_ini_file(str(_INPUT))
    return base.model_copy(
        update={"rateconstant": RateConstantConfig(style="htst", k0=1.0)}
    )


def service(**kwargs) -> PrefactorService:
    cfg = htst_config()
    return PrefactorService(
        cfg, object(), create_rate_constant(cfg.rateconstant), method="fd", **kwargs
    )


def test_derived_descriptor_widens_for_a_species_absent_from_the_first_request():
    svc = service()
    assert svc.current_descriptor is None
    first = svc.descriptor_for(("Si", "Si", "Si"))
    assert first.engine.species == ("Si",)
    assert (first.engine.species, first.engine.masses) == species_map(["Si"])
    # A later request whose full system holds a new species is described by
    # the one species rule applied to its own types, not rejected.
    wider = svc.descriptor_for(("Si", "Ge", "Si"))
    assert (wider.engine.species, wider.engine.masses) == species_map(["Si", "Ge"])
    assert wider.descriptor_id != first.descriptor_id
    # The narrower system keeps its own stable descriptor.
    again = svc.descriptor_for(("Si", "Si"))
    assert again.descriptor_id == first.descriptor_id
    assert svc.current_descriptor.descriptor_id == again.descriptor_id


def test_authoritative_map_rejects_unknown_species_by_name():
    svc = service(species_masses=(("Si",), (28.0855,)))
    assert svc.current_descriptor.engine.species == ("Si",)
    with pytest.raises(HTSTRequestError) as captured:
        svc.descriptor_for(("Si", "Ge"))
    message = str(captured.value)
    assert "Ge" in message and "Si" in message
    assert "species map" in message
    # The authoritative map is never widened by a request.
    assert svc.current_descriptor.engine.species == ("Si",)
    assert svc.descriptor_for(("Si",)).engine.species == ("Si",)
