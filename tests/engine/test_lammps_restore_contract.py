"""Restore contract of ``LammpsEngine``: explicit species map and retryable rebuild.

Promoted from the independent review's red-to-green pack (findings F5 and F6
of the HTST/backend integration review), assertions unchanged.
``ensure_full_system`` must replay the *remembered* species/mass map, so a
system initialised with a species map larger than its own symbols keeps its
type numbering and explicit masses, and a rebuild counts as complete only once
``initialize_potential`` has succeeded: a replay that raises leaves the engine
reported as cropped with the remembered descriptor intact, the next call
retries, and the restored energy is the fresh one.

Serial only (``MPI.COMM_SELF``): no MPI worker pool, no potential file. The
engines use LAMMPS's built-in ``lj/cut`` with an analytic two-atom energy.
Native dependencies are required; nothing here skips.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from mpi4py import MPI

from pykmc.engine.lammps import FullSystem, LammpsEngine

POSITIONS = np.array([[2.0, 2.0, 2.0], [3.3, 2.0, 2.0]])
CELL = np.diag([10.0, 10.0, 10.0])
# Two atoms, epsilon = sigma = 1, r = 1.3, and no periodic image within cutoff.
EXPECTED_PAIR_ENERGY = 4.0 * ((1.0 / 1.3) ** 12 - (1.0 / 1.3) ** 6)
EXPLICIT_MAP: dict[str, tuple[Any, ...]] = {
    "species": ("Fe", "Ni"),
    "masses": (56.0, 60.0),
}
EXPLICIT_MAP_STATE: dict[str, Any] = {
    "remembered_species": ("Fe", "Ni"),
    "remembered_masses": (56.0, 60.0),
    "native_ntypes": 2,
    "native_atom_types": (2, 2),
    "native_masses": (56.0, 60.0),
}


@contextmanager
def initialized_engine(
    **species_overrides: tuple[str, ...] | tuple[float, ...],
) -> Iterator[LammpsEngine]:
    """Yield a started serial ``lj/cut`` engine with the two-Ni-atom system loaded.

    Parameters
    ----------
    **species_overrides : tuple[str, ...] | tuple[float, ...]
        Forwarded to ``initialize_system`` (``species=`` / ``masses=``) to
        give the system an explicit map larger than its own symbols.

    Yields
    ------
    LammpsEngine
        Engine with parameters, system and potential initialised; closed on
        exit.

    """
    config = SimpleNamespace(
        pair_style="lj/cut 2.5",
        pair_coeff="* * 1.0 1.0",
        min_style="cg",
        minimize="1e-8 1e-8 100 1000",
        frz_min="1e-8 1e-8 100 1000",
        verbosity=0,
    )
    engine = LammpsEngine(config, comm=MPI.COMM_SELF)
    try:
        engine.start()
        engine.initialize_parameters()
        engine.initialize_system(
            types=("Ni", "Ni"),
            positions=POSITIONS.copy(),
            cell=CELL.copy(),
            pbc=(True, True, True),
            **species_overrides,
        )
        engine.initialize_potential()
        yield engine
    finally:
        engine.close()


def live_species_state(engine: LammpsEngine) -> dict[str, Any]:
    """Read the actual native state as well as the remembered Python descriptor.

    Parameters
    ----------
    engine : LammpsEngine
        Started engine holding a system.

    Returns
    -------
    dict[str, Any]
        Remembered species/masses, live ``ntypes``, live per-atom integer
        types (LAMMPS id order) and live per-type masses.

    """
    ntypes = int(engine.lmp.extract_global("ntypes"))
    type_ids = tuple(
        int(value)
        for value in np.ctypeslib.as_array(engine.lmp.gather_atoms("type", 0, 1))
    )
    native_masses = engine.lmp.extract_atom("mass")
    return {
        "remembered_species": engine.full_system.species,
        "remembered_masses": engine.full_system.masses,
        "native_ntypes": ntypes,
        "native_atom_types": type_ids,
        "native_masses": tuple(float(native_masses[i]) for i in range(1, ntypes + 1)),
    }


def descriptor_fields(fs: FullSystem) -> tuple[Any, ...]:
    """Return the value content of a ``FullSystem`` (it compares by identity).

    Parameters
    ----------
    fs : FullSystem
        Remembered full-system descriptor.

    Returns
    -------
    tuple[Any, ...]
        ``(types, species, masses, cell as nested lists, pbc)``.

    """
    return (fs.types, fs.species, fs.masses, fs.cell.tolist(), fs.pbc)


def test_intact_engine_restore_is_a_noop_and_has_real_pair_energy() -> None:
    """Positive control: dependencies and the unchanged-system path work."""
    with initialized_engine() as engine:
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)
        assert engine.system_is_cropped is False
        assert engine.ensure_full_system(POSITIONS + 0.1) is False
        np.testing.assert_allclose(engine.get_positions(), POSITIONS)
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)


def test_restore_preserves_absent_species_type_ids_and_nondefault_masses() -> None:
    """F5: a subset atom array must retain the full potential's type map."""
    expected = dict(EXPLICIT_MAP_STATE)
    with initialized_engine(**EXPLICIT_MAP) as engine:
        assert live_species_state(engine) == expected  # Initial setup is valid.
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)

        engine.command("clear")
        assert engine.system_is_cropped is True
        assert engine.ensure_full_system(POSITIONS) is True

        assert live_species_state(engine) == expected
        assert engine.system_is_cropped is False
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)


def test_failed_potential_restore_stays_dirty_and_retry_rebuilds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """F6: a one-shot replay failure cannot make an incomplete engine clean."""
    with initialized_engine() as engine:
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)
        original_potential = engine.initialize_potential
        replay_calls = 0

        def fail_once_then_initialize() -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == 1:
                raise RuntimeError("injected one-shot potential replay failure")
            return original_potential()

        monkeypatch.setattr(engine, "initialize_potential", fail_once_then_initialize)
        engine.command("clear")
        with pytest.raises(RuntimeError, match="one-shot potential replay failure"):
            engine.ensure_full_system(POSITIONS)

        # Exercise recovery before asserting, so the red result reports both
        # the incorrect state marker and the native zero-energy consequence.
        dirty_after_failure = engine.system_is_cropped
        rebuilt_on_retry = engine.ensure_full_system(POSITIONS)
        energy_after_retry = engine.get_total_energy()
        assert {
            "dirty_after_failure": dirty_after_failure,
            "rebuilt_on_retry": rebuilt_on_retry,
            "potential_replay_calls": replay_calls,
            "energy_after_retry": energy_after_retry,
        } == {
            "dirty_after_failure": True,
            "rebuilt_on_retry": True,
            "potential_replay_calls": 2,
            "energy_after_retry": pytest.approx(EXPECTED_PAIR_ENERGY),
        }
        assert engine.system_is_cropped is False
        np.testing.assert_allclose(engine.get_positions(), POSITIONS)


def test_failed_system_replay_keeps_the_explicit_map_for_the_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure in the first replay step keeps the authoritative descriptor.

    F5 and F6 together: with ``initialize_system`` failing once after the
    ``clear``, ``full_system`` must still be the pre-failure descriptor (same
    object, explicit ``('Fe', 'Ni')`` / ``(56, 60)`` map) and the retry must
    rebuild the same live types and masses, not an inferred one-species
    system.
    """
    with initialized_engine(**EXPLICIT_MAP) as engine:
        descriptor = engine.full_system
        original_initialize_system = engine.initialize_system
        replay_calls = 0

        def fail_once_then_initialize(*args: Any, **kwargs: Any) -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == 1:
                raise RuntimeError("injected one-shot system replay failure")
            return original_initialize_system(*args, **kwargs)

        monkeypatch.setattr(engine, "initialize_system", fail_once_then_initialize)
        engine.command("clear")
        with pytest.raises(RuntimeError, match="one-shot system replay failure"):
            engine.ensure_full_system(POSITIONS)

        assert engine.system_is_cropped is True
        assert engine.full_system is descriptor
        assert engine.ensure_full_system(POSITIONS) is True
        assert replay_calls == 2
        assert engine.system_is_cropped is False
        assert descriptor_fields(engine.full_system) == descriptor_fields(descriptor)
        assert live_species_state(engine) == EXPLICIT_MAP_STATE
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)


def test_count_mismatch_rebuild_is_retryable_after_a_potential_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rebuild flagged only by the atom count is retryable as well.

    A ``clear`` sent straight to the LAMMPS handle bypasses ``command``, so
    the engine is cropped through the atom-count mismatch alone. Once the
    replay has re-created the atoms the count matches again, so
    ``ensure_full_system`` must raise its own marker before replaying;
    otherwise a potential failure would leave a potential-less instance
    reported as intact.
    """
    with initialized_engine() as engine:
        original_potential = engine.initialize_potential
        replay_calls = 0

        def fail_once_then_initialize() -> None:
            nonlocal replay_calls
            replay_calls += 1
            if replay_calls == 1:
                raise RuntimeError("injected one-shot potential replay failure")
            return original_potential()

        monkeypatch.setattr(engine, "initialize_potential", fail_once_then_initialize)
        engine.lmp.command("clear")  # not LammpsEngine.command: no marker
        assert engine._cleared_since_init is False
        assert engine.system_is_cropped is True  # by the atom count only
        with pytest.raises(RuntimeError, match="one-shot potential replay failure"):
            engine.ensure_full_system(POSITIONS)

        assert int(engine.lmp.get_natoms()) == len(POSITIONS)  # count matches
        assert engine.system_is_cropped is True  # ... the marker still reports
        assert engine.ensure_full_system(POSITIONS) is True
        assert replay_calls == 2
        assert engine.system_is_cropped is False
        assert engine.get_total_energy() == pytest.approx(EXPECTED_PAIR_ENERGY)
        np.testing.assert_allclose(engine.get_positions(), POSITIONS)
