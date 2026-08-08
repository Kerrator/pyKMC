"""Unit tests for multi-species support in the active-volume ``reset()``.

``reset()`` used to hardcode ``create_box 1 box``, so any alloy (NiCr, NiFe, ...)
crashed at the first AV event search: ``initialize_potential`` maps a
multi-element ``pair_coeff`` onto a one-type box and LAMMPS rejects it. These
tests drive ``reset()`` with a command-recording fake engine and pin the emitted
``create_box`` command, plus the ``n_types`` the HTST prefactor path derives
from the full-system types array.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("lammps")
pytest.importorskip("pypARTn")

from pykmc.activevolume import active_volume as av  # noqa: E402
from pykmc.enginemanager.lmpi import lammps_operations as ops  # noqa: E402

_CELL = np.diag([20.0, 20.0, 50.0])


class _FakeEngine:
    """Record every LAMMPS command string instead of executing it."""

    def __init__(self) -> None:
        self.commands: list[str] = []

    def command(self, cmd: str) -> None:
        """Store ``cmd`` verbatim.

        Parameters
        ----------
        cmd
            The LAMMPS command line the caller would run.

        """
        self.commands.append(cmd)


class _Lammps:
    """LAMMPS config shim: a two-element eam/alloy potential."""

    pair_style = "eam/alloy"
    pair_coeff = "* * Bonny_2013_NiFeCr.eam Cr Ni"


class _Cfg:
    """Config shim carrying only what ``reset()`` reads."""

    lammps = _Lammps()


def test_reset_default_single_species_box() -> None:
    """Default ``n_types`` keeps the original single-species ``create_box``."""
    engine = _FakeEngine()
    av.reset(engine, _Cfg(), _CELL)
    assert "create_box 1 box" in engine.commands


def test_reset_two_species_box() -> None:
    """``n_types=2`` emits ``create_box 2`` so alloy crops hold both types."""
    engine = _FakeEngine()
    av.reset(engine, _Cfg(), _CELL, n_types=2)
    assert "create_box 2 box" in engine.commands
    assert not any(c.startswith("create_box 1 ") for c in engine.commands)


def test_reset_orders_box_before_potential() -> None:
    """``create_box`` precedes ``pair_coeff`` (eam/alloy sets masses there)."""
    engine = _FakeEngine()
    av.reset(engine, _Cfg(), _CELL, n_types=2)
    i_box = engine.commands.index("create_box 2 box")
    i_coeff = next(
        i for i, c in enumerate(engine.commands) if c.startswith("pair_coeff")
    )
    assert i_box < i_coeff


class _RCShim:
    """Rate-constant shim: just the radii the zone-crop branch reads."""

    nu0_zone_radius = 9.0
    free_radius = 3.5


class _ControlShim:
    """Control shim: AV off so the op takes the nu0-zone branch."""

    active_volume = False


class _OpsCfg:
    """Config shim for ``compute_event_prefactors`` up to the ``reset`` call."""

    rateconstant = _RCShim()
    control = _ControlShim()


def test_prefactor_reset_receives_alloy_type_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The HTST prefactor path passes ``n_types`` from the FULL types array.

    The crop may hold fewer species, but integer type refs come from the
    full-system ``sorted(set(types))`` map, so ``n_types`` must count the
    full-system species. The sentinel raise after capture exercises the op's
    never-raise fallback at the same time.
    """
    captured: dict[str, int] = {}

    def _fake_reset(
        engine: object, config: object, cell: np.ndarray, n_types: int = 1
    ) -> None:
        """Capture ``n_types`` then abort the op via its graceful-fallback path.

        Parameters
        ----------
        engine, config, cell
            Ignored stand-ins for the real arguments.
        n_types
            The value under test, recorded for the assertion.

        """
        captured["n_types"] = n_types
        raise RuntimeError("sentinel: stop after reset")

    monkeypatch.setattr(ops, "reset", _fake_reset)
    monkeypatch.setattr(ops, "_get_serial_hessian_engine", lambda engine: object())

    geom = np.zeros((4, 3))
    res = ops.compute_event_prefactors(
        object(),
        _OpsCfg(),
        central_atom_idx=0,
        min1_positions=geom,
        saddle_positions=geom,
        min2_positions=geom,
        types=["Ni", "Cr", "Ni", "Ni"],
        cell=_CELL,
    )

    assert captured["n_types"] == 2
    assert res.nu0_forward is None  # graceful fallback, not a raise
    assert "sentinel" in res.reason
