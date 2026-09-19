"""Scratch snapshots must agree with the request before Hessian dispatch.

The recording scratch models two documented boundary behaviors: pair_coeff can
replace requested masses, and a force file can change after the adapter's early
check but before scratch initialization. These are descriptor/protocol oracles,
not physical frequency or force calculations.
"""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.mark.parametrize("change", ["potential_mass_override", "late_force_file_edit"])
def test_initialized_scratch_must_match_the_producing_descriptor(
    tmp_path, monkeypatch, change
):
    import pykmc
    import pykmc.engine.htst_lammps as adapter
    from pykmc.config import Config, RateConstantConfig
    from pykmc.engine.lammps import FullSystem
    from pykmc.physics import EnginePhysics
    from pykmc.rate_constant import create_rate_constant
    from pykmc.rate_constant.prefactors import PrefactorService

    potential = tmp_path / "recorded-force.eam.alloy"
    potential.write_bytes(b"Scratch-boundary coefficient fixture A\n")
    config = Config.from_ini_file(
        str(Path(pykmc.__file__).resolve().parent.parent / "tests/data/input.in")
    )
    config = config.model_copy(
        update={
            "rateconstant": RateConstantConfig(
                style="htst", k0=1.0, premin=False, free_radius=2.0
            ),
            "frozen_atoms": None,
            "lammps": config.lammps.model_copy(
                update={
                    "pair_style": "eam/alloy",
                    "pair_coeff": f"* * {potential} Ni Cu",
                }
            ),
        }
    )
    engine_physics = EnginePhysics.capture(
        config.lammps, species=("Ni", "Cu"), masses=(61.0, 65.0)
    )
    service = PrefactorService(
        config,
        object(),
        create_rate_constant(config.rateconstant),
        engine_physics=engine_physics,
    )
    initial = np.array([[1.0, 1.0, 1.0], [2.0, 1.0, 1.0]])
    saddle = initial.copy()
    saddle[0, 0] += 0.1
    final = initial.copy()
    final[0, 0] += 0.2
    request = service.build_request(
        event_key=("review", change),
        min1_positions=initial,
        saddle_positions=saddle,
        min2_positions=final,
        types=("Ni", "Ni"),
        cell=20.0 * np.eye(3),
        pbc=(True, True, True),
        center_index=0,
    )
    original_identity = request.descriptor.descriptor_id
    calls = []

    class Scratch:
        full_system = None

        def start(self):
            calls.append("start")

        def initialize_parameters(self):
            calls.append("parameters")

        def initialize_system(self, **kwargs):
            calls.append("system")
            self.full_system = FullSystem(
                types=kwargs["types"],
                species=kwargs["species"],
                masses=kwargs["masses"],
                cell=np.array(kwargs["cell"], copy=True),
                pbc=kwargs["pbc"],
            )

        def initialize_potential(self):
            calls.append("potential")
            # The native LammpsEngine.initialize_potential contract refreshes
            # masses and captures EnginePhysics after applying pair_coeff.
            actual_masses = (
                (62.0, 65.0)
                if change == "potential_mass_override"
                else self.full_system.masses
            )
            actual = EnginePhysics.capture(
                config.lammps, self.full_system.species, actual_masses
            )
            self.full_system = replace(
                self.full_system, masses=actual_masses, physics=actual
            )

        def close(self):
            calls.append("close")

    scratch = Scratch()
    extension = object.__new__(adapter.LammpsHTSTExtension)
    extension.engine = SimpleNamespace(config=config.lammps, comm=None, engine_id=0)

    def create_scratch():
        calls.append("create")
        if change == "late_force_file_edit":
            # Happens after compute_event_prefactors' initial ForceModel check.
            # The scratch sees stable B before/after its own initialization.
            potential.write_bytes(b"Scratch-boundary coefficient fixture B\n")
        return scratch

    sentinel = object()

    def recording_kernel(local_request, hessian, **kwargs):
        calls.append("kernel")
        assert local_request.descriptor.engine == scratch.full_system.physics, (
            "The kernel received a producing descriptor that does not describe "
            f"the initialized scratch after {change}"
        )
        assert local_request.masses == scratch.full_system.masses
        return sentinel

    monkeypatch.setattr(extension, "_new_scratch", create_scratch)
    monkeypatch.setattr(extension, "_hessian_fn", lambda *args: None)
    monkeypatch.setattr(adapter, "_kernel_compute_event_prefactors", recording_kernel)
    try:
        result = extension.compute_event_prefactors(request)
    except (ValueError, RuntimeError) as exc:
        # A deliberate mismatch rejection is valid and must precede the kernel.
        assert any(
            word in str(exc).lower()
            for word in ("descriptor", "mass", "physics", "force", "potential")
        ), f"Unrelated failure is not the expected physics rejection: {exc}"
        assert "kernel" not in calls
    else:
        # Alternatively the implementation must transport actual producing
        # physics, which the recording kernel above independently checks.
        assert result is sentinel
        assert "kernel" in calls
    assert calls[-1] == "close", "mismatch must release the initialized scratch"
    assert request.descriptor.descriptor_id == original_identity, (
        "repair must not mutate the captured request's original provenance"
    )
