"""Native closed-crop retry and actual installed wrapper-policy boundary."""

import json
import numpy as np
import pytest
import lammps
from pykmc.engine import lammps as engine_module
from tests.engine.test_lammps_restore_contract import (
    initialized_engine,
    descriptor_fields,
    POSITIONS,
    EXPECTED_PAIR_ENERGY,
    EXPLICIT_MAP,
)


def assert_restored(engine, descriptor):
    assert engine.lmp is not None
    assert engine.system_is_cropped is False
    assert descriptor_fields(engine.full_system) == descriptor_fields(descriptor)
    assert engine.full_system.physics == descriptor.physics
    assert int(engine.lmp.get_natoms()) == len(POSITIONS)
    assert engine.get_total_energy() == pytest.approx(
        EXPECTED_PAIR_ENERGY, rel=0, abs=1e-12
    )
    np.testing.assert_allclose(engine.get_positions(), POSITIONS, rtol=0, atol=1e-12)
    assert tuple(np.ctypeslib.as_array(engine.lmp.gather_atoms("type", 0, 1))) == (2, 2)
    live = engine.lmp.extract_atom("mass")
    assert tuple(float(live[i]) for i in (1, 2)) == descriptor.masses


def test_native_closed_pending_crop_restarts_with_original_physics():
    # An absent species slot and isotope masses must survive recreation
    # with the original physics snapshot; no potential file is needed.
    with initialized_engine(**EXPLICIT_MAP) as engine:
        descriptor = engine.full_system
        old = engine.lmp
        engine.command("clear")
        engine.close()
        assert engine.lmp is None and engine.system_is_cropped
        assert engine.ensure_full_system(POSITIONS) is True
        assert engine.lmp is not old
        assert_restored(engine, descriptor)
        assert engine.ensure_full_system(POSITIONS + 0.1) is False
        assert_restored(engine, descriptor)


def test_native_intact_explicit_close_keeps_existing_no_restart_policy():
    with initialized_engine(**EXPLICIT_MAP) as engine:
        descriptor = engine.full_system
        engine.close()
        assert engine.lmp is None and not engine.system_is_cropped
        assert engine.ensure_full_system(POSITIONS) is False
        assert engine.lmp is None and engine.full_system is descriptor


def test_native_installed_wrapper_failure_then_explicit_closed_retry():
    with initialized_engine(**EXPLICIT_MAP) as engine:
        descriptor = engine.full_system
        old = engine.lmp
        engine.command("clear")
        # Real decorated method and real native error. pair_coeff without a
        # box fails after native pair_style; no simulated exception export.
        with pytest.raises(
            Exception, match="Pair_coeff command before simulation box is defined"
        ) as raised:
            engine.initialize_potential()
        wrapper_closed = engine.lmp is None
        exported = getattr(lammps, "LAMMPSException", None)
        print(
            json.dumps(
                {
                    "installed_exports_LAMMPSException": exported is not None,
                    "handled_types": [
                        t.__name__ for t in engine_module._LAMMPS_EXCEPTIONS
                    ],
                    "observed_error_type": type(raised.value).__name__,
                    "observed_wrapper_closed_handle": wrapper_closed,
                    "message": str(raised.value),
                }
            ),
            flush=True,
        )
        assert engine.full_system is descriptor
        assert engine.system_is_cropped
        if not engine_module._LAMMPS_EXCEPTIONS:
            assert engine.lmp is old  # documented no-op decorator policy
        # Explicitly exercise native recreation even on wrappers that keep a
        # failed handle open; do not claim the wrapper itself closed this one.
        engine.close()
        assert engine.system_is_cropped and engine.lmp is None
        assert engine.ensure_full_system(POSITIONS) is True
        assert_restored(engine, descriptor)
