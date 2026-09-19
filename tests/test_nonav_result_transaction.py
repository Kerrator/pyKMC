"""Public postcondition rejection must remain inside the non-AV transaction.

Real public wrappers, payload resolution/validation, resource scope and native
gather/scatter conversion run. Only the pARTn implementation and native storage
are recording seams; no LAMMPS constructor, MPI rank or minimizer is launched.
"""

from types import SimpleNamespace
import ctypes

import numpy as np
import pytest

from pykmc.config import RegionConfig
from pykmc.engine.lammps import FullSystem, LammpsEngine
from pykmc.physics import EnginePhysics, ResolvedConstraints
from pykmc.result import EventRefinementOutput, EventSearchOutput, Ok


class NativeStorage:
    def __init__(self, positions):
        self.positions = positions.copy()
        self.fixes = {"user_fix"}
        self.groups = {"all", "user_group"}
        self.user_callback = {"function": object(), "caller": object()}
        self.callback = {"user_fix": self.user_callback}
        self.commands = []
        self.scatters = []

    def get_natoms(self):
        return len(self.positions)

    def has_id(self, kind, name):
        return name in (self.fixes if kind == "fix" else self.groups)

    def gather_atoms(self, name, type_code, count):
        assert (name, type_code, count) == ("x", 1, 3)
        self.gathered = (ctypes.c_double * self.positions.size)(*self.positions.flat)
        return self.gathered

    def scatter_atoms(self, name, type_code, count, values):
        assert (name, type_code, count) == ("x", 1, 3)
        self.positions = np.ctypeslib.as_array(values).reshape((-1, 3)).copy()
        self.scatters.append(self.positions.copy())

    def command(self, command):
        self.commands.append(command)
        words = command.split()
        if words[0] == "unfix":
            assert words[1] != "user_fix"
            self.fixes.remove(words[1])
        elif words[0] == "group" and words[2] == "delete":
            assert words[1] != "user_group"
            self.groups.remove(words[1])
        else:
            raise AssertionError(f"unexpected postcondition-cleanup command: {command}")


@pytest.mark.parametrize("operation", ["search", "refine"])
def test_invalid_ok_restores_nonav_entry_and_preserves_original_invalid_output(
    operation, monkeypatch
):
    entry = np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    source = entry.copy()
    source[0, 0] += 0.05  # Requested input differs from existing native entry.
    source_saved = source.copy()
    types = np.array(["Ni", "Ni"])
    cell = np.diag([10.0, 12.0, 14.0])
    pbc = (True, False, True)
    cfg = SimpleNamespace(
        control=SimpleNamespace(active_volume=False),
        frozen_atoms=RegionConfig(indices=[1]),
        lammps=SimpleNamespace(
            pair_style="zero 2.5",
            pair_coeff="* *",
            min_style="cg",
            frz_min="0 1e-8 100 1000",
            verbosity=0,
        ),
    )
    payload = ResolvedConstraints.resolve(
        source, types, cfg.frozen_atoms, (91, 17), cell=cell, pbc=pbc
    )
    engine = LammpsEngine(cfg.lammps, comm=None)
    native = NativeStorage(entry)
    engine.lmp = native
    engine._is_orthorhombic = True
    original = FullSystem(
        types=tuple(types),
        species=("Ni",),
        masses=(60.0,),
        cell=cell.copy(),
        pbc=pbc,
        physics=EnginePhysics.capture(cfg.lammps, ("Ni",), (60.0,)),
    )
    engine.full_system = original
    invalid = source.copy()
    invalid[0, 0] += 0.75
    invalid[1, 1] += 0.2  # Physical motion on a nonperiodic fixed direction.
    observed = []

    def producing_impl(*args, **kwargs):
        # The actual public wrapper validated the valid input before reaching
        # this seam. Simulate a successful native return with a bad postcondition.
        native.fixes.update({"10", "f_frozen_pre", "f_frozen_post"})
        native.groups.add("g_frozen")
        engine.set_positions(invalid)
        if operation == "search":
            output = EventSearchOutput(
                central_atom_index=0,
                move_atom_index=0,
                min1_positions=source.copy(),
                saddle_positions=invalid.copy(),
                min2_positions=source.copy(),
                dE_forward=0.2,
                dE_backward=0.2,
                cell=cell.copy(),
                types=types.copy(),
            )
        else:
            output = EventRefinementOutput(
                central_atom_index=0,
                saddle_positions=invalid.copy(),
                E_saddle=0.2,
                refined="T",
            )
        observed.append(output)
        return Ok(output)

    monkeypatch.setattr(engine, f"_partn_{operation}_impl", producing_impl)
    with pytest.raises(ValueError, match="incompatible constrained event"):
        getattr(engine, f"partn_{operation}")(
            config=cfg,
            central_atom_idx=0,
            positions=source,
            types=types,
            cell=cell,
            constraints=payload,
            user_constraints=payload,
        )
    assert len(observed) == 1, "must reach real public result validation"
    np.testing.assert_array_equal(native.positions, entry)
    assert len(native.scatters) == 2, "native operation then entry restoration"
    np.testing.assert_array_equal(native.scatters[0], invalid)
    np.testing.assert_array_equal(native.scatters[1], entry)
    assert native.fixes == {"user_fix"}
    assert native.groups == {"all", "user_group"}
    assert native.callback == {"user_fix": native.user_callback}
    assert engine.full_system is original and not engine._cleared_since_init
    assert engine.lmp is native, "non-AV recovery must preserve unrelated resources"
    np.testing.assert_array_equal(source, source_saved)
    np.testing.assert_array_equal(observed[0].saddle_positions, invalid)
    assert payload.fixed_ids == (17,)
