"""``[frozen_atoms]`` stay put through the refinement pre-relaxation.

Serial: pytest tests/engine/test_frozen_prerelax.py

Before refining a saddle, the refiner moves the rcut atoms to the reference
saddle and minimizes the rest. That minimization held the rcut atoms (and,
with the active volume, the atoms beyond ``rmov``) but not ``[frozen_atoms]``:
on the dimer tests' Ni vacancy a frozen second-shell atom moved 0.05 Å in
full-system ARTn refinement and 0.08 Å on the active volume. On the active
volume the refiner also moved frozen atoms in the buffer, by about 1e-5 Å,
since it moves the whole buffer as pARTn does.

The ARTn tests refine the vacancy hop on real LAMMPS, once on the full system
and once on pARTn's active volume, with one second-shell atom and one buffer
atom frozen, and check the frozen atoms in the refiner's start, its movable
mask and the refined saddle. The pARTn test checks that ``partn_refine``
hands ``[frozen_atoms]`` to its pre-relaxation on either path.
"""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

lammps = pytest.importorskip("lammps")

from mpi4py import MPI  # noqa: E402

import pykmc.artn  # noqa: E402
from pykmc.atomic_environment import frozen_atom_indices  # noqa: E402
from pykmc.config import PartnConfig, RegionConfig  # noqa: E402
from pykmc.engine import lammps as lammps_engine  # noqa: E402
from pykmc.engine.artn_search import ArtnSearchExtension  # noqa: E402
from pykmc.engine.lammps import LammpsEngine  # noqa: E402
from pykmc.eventsearch.strategies.artn import artn_refine_options  # noqa: E402
from pykmc.eventsearch.strategies.dimer import movable_atom_indices  # noqa: E402

from .test_engine_artn_search import AV_RADII, _av_sets, _close, _search  # noqa: E402
from .test_engine_dimer_search import (  # noqa: E402, F401 (fixtures)
    _make_engine,
    _minimum_image,
    eam_config,
    vacancy,
)

# The rcut atoms: the central atom and its first shell.
RCUT = 2.7
REFINE_OPTIONS = artn_refine_options(PartnConfig())
SEEDS = [1, 2, 3]
DELR_SAD_THR = 0.5


@pytest.fixture(scope="module")
def hop(eam_config, vacancy):  # noqa: ANN001, ANN201, F811
    """The hop's saddle from a full-system ARTn search."""
    engine = _make_engine(
        eam_config,
        None,
        vacancy.positions,
        vacancy.types,
        vacancy.cell,
        vacancy.pbc,
        ArtnSearchExtension,
    )
    try:
        result = _search(engine, vacancy)
    finally:
        _close(engine)
    assert result.success
    return result.saddle_positions


def _frozen(system, saddle) -> np.ndarray:  # noqa: ANN001
    """One second-shell atom (within rmov) and one buffer atom (rmov to ract).

    The buffer atom is the one the hop's saddle displaces most. The
    second-shell atom is the one it displaces least: freezing one it pushes
    harder moves the full-system saddle past ``DELR_SAD_THR``.
    """
    distance = np.linalg.norm(
        _minimum_image(
            system.positions - system.positions[system.central], system.cell
        ),
        axis=1,
    )
    push = np.linalg.norm(
        _minimum_image(saddle - system.positions, system.cell), axis=1
    )
    rmov, ract = AV_RADII
    second = np.flatnonzero((distance > RCUT) & (distance <= rmov))
    buffer = np.flatnonzero((distance > rmov) & (distance <= ract))
    return np.sort([second[np.argmin(push[second])], buffer[np.argmax(push[buffer])]])


@pytest.mark.parametrize("path", ["full", "av"])
def test_artn_refinement_keeps_frozen_atoms(  # noqa: D103
    eam_config,  # noqa: ANN001, F811
    vacancy,  # noqa: ANN001, F811
    hop,  # noqa: ANN001
    monkeypatch,  # noqa: ANN001
    path,  # noqa: ANN001
) -> None:
    if MPI.COMM_WORLD.Get_size() > 1:
        pytest.skip("serial tests must run without mpirun")
    minimum = vacancy.positions
    frozen = _frozen(vacancy, hop)
    distance = np.linalg.norm(
        _minimum_image(minimum - minimum[vacancy.central], vacancy.cell), axis=1
    )
    saddle_idx = np.flatnonzero(distance <= RCUT)
    assert not np.isin(frozen, saddle_idx).any()

    # The movable set and options as submit_artn_refine builds them.
    rmov, ract = AV_RADII
    region = RegionConfig(indices=frozen.tolist())
    config = SimpleNamespace(
        control=SimpleNamespace(active_volume=path == "av"),
        activevolume=SimpleNamespace(rmov=rmov, ract=ract),
        frozen_atoms=region,
    )
    movable = movable_atom_indices(config, vacancy.central, vacancy)
    options = dict(REFINE_OPTIONS)
    if path == "av":
        options["av_radii"] = AV_RADII
        options["av_frozen"] = frozen_atom_indices(region, minimum, vacancy.types)
        av_idx, _ = _av_sets(vacancy)
        local = np.searchsorted(av_idx, frozen)
        np.testing.assert_array_equal(av_idx[local], frozen)
    else:
        local = frozen

    real = pykmc.artn.artn_refine
    seen = {}

    def recording(positions, cell, pbc, mask, central, seeds, query, **kwargs):  # noqa: ANN001, ANN003, ANN202
        seen.update(start=positions.copy(), mask=mask.copy())
        return real(positions, cell, pbc, mask, central, seeds, query, **kwargs)

    monkeypatch.setattr("pykmc.engine.artn_search.run_refine", recording)
    engine = _make_engine(
        eam_config,
        None,
        minimum,
        vacancy.types,
        vacancy.cell,
        vacancy.pbc,
        ArtnSearchExtension,
    )
    try:
        result = engine.artn_refine(
            positions=minimum,
            types=vacancy.types,
            cell=vacancy.cell,
            pbc=vacancy.pbc,
            movable=movable,
            central_atom=vacancy.central,
            saddle_idx=saddle_idx,
            saddle_positions=hop[saddle_idx],
            seeds=SEEDS,
            delr_sad_thr=DELR_SAD_THR,
            options=options,
        )
    finally:
        _close(engine)

    assert result.success, result.reason
    assert result.prerelax_iterations > 0
    # The pre-relaxation held them, the refiner may not move them, and the
    # refined saddle has them where the minimum has them.
    np.testing.assert_allclose(
        _minimum_image(seen["start"][local] - minimum[frozen], vacancy.cell),
        0.0,
        atol=1e-10,
    )
    assert not seen["mask"][local].any()
    np.testing.assert_allclose(
        _minimum_image(result.saddle_positions[frozen] - minimum[frozen], vacancy.cell),
        0.0,
        atol=1e-10,
    )


class _Stop(Exception):
    """Raised by the stub pARTn: the pre-relaxation has been recorded."""


@pytest.mark.parametrize("active_volume", [False, True])
@pytest.mark.parametrize("frozen", [None, [0, 9]])
def test_partn_refine_holds_frozen_atoms_in_the_pre_relaxation(  # noqa: D103
    monkeypatch,  # noqa: ANN001
    active_volume,  # noqa: ANN001
    frozen,  # noqa: ANN001
) -> None:
    positions = np.array([[float(i), 0.0, 0.0] for i in range(12)])
    types = ["Ni"] * len(positions)
    saddle_idx = np.array([2, 3])
    held = {}

    def refine_av(engine, config, central, pos, cell, typ, idx, saddle, held_idx=None):  # noqa: ANN001, ANN202
        held["av"] = held_idx
        return 0.0, np.arange(len(pos)), np.array([central + 1])

    def stop(engine):  # noqa: ANN001, ANN202
        raise _Stop

    monkeypatch.setattr(lammps_engine, "partn_refine_AV", refine_av)
    monkeypatch.setattr(LammpsEngine, "set_positions", lambda self, positions: None)
    monkeypatch.setattr(
        LammpsEngine, "minimize_freeze_core", lambda self, idx: held.update(full=idx)
    )
    monkeypatch.setitem(sys.modules, "pypARTn", SimpleNamespace(artn=stop))
    config = SimpleNamespace(
        frozen_atoms=None if frozen is None else RegionConfig(indices=frozen),
        control=SimpleNamespace(active_volume=active_volume),
    )

    with pytest.raises(_Stop):
        LammpsEngine(config=None).partn_refine(
            config,
            2,
            positions,
            np.diag([12.0, 10.0, 10.0]),
            types,
            saddle_idx=saddle_idx,
            saddle_positions=positions[saddle_idx],
        )

    expected = [] if frozen is None else frozen
    if active_volume:
        assert held.keys() == {"av"}
        np.testing.assert_array_equal(held["av"], expected)
    else:
        assert held.keys() == {"full"}
        np.testing.assert_array_equal(held["full"], np.union1d(saddle_idx, expected))
