"""Unit tests for full-colour typing in the reference event table."""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pykmc import Config, NeighborsList, System, constants
from pykmc.dimer import _minimum_image
from pykmc.environments.graph_nauty import graph
from pykmc.event_table import ActiveEventTable, ReferenceEventTable
from pykmc.psr import PointSetRegistration
from pykmc.rate_constant import RateConstantError, RateConstantOutput, RateRequest
from pykmc.result import (
    Err,
    ErrorInfo,
    ErrorType,
    EventRefinementOutput,
    EventSearchOutput,
    Ok,
)
from pykmc.search_rules import move_atom


def _build_trivial_series(
    config: Config, system: System
) -> tuple[pd.Series, pd.Series]:
    """Run ``_build_event_series`` for a trivial (min1==saddle==min2) event.

    This exercises the graph/symmetry/type-storage path without going through
    barrier validation, which is unrelated to colouring.
    """
    table = ReferenceEventTable(config)
    pos = system.positions
    return table._build_event_series(
        min1_positions=pos,
        saddle_positions=pos,
        min2_positions=pos,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.5,
        cell=system.cell,
        types=list(system.types),
    )


class TestReferenceTableTypes:
    """The ``types`` column is always populated, so the schema is mode-independent."""

    def test_types_stored_in_both_modes(
        self, system_binary_fcc: System, config_system_single_type: Config
    ) -> None:
        """Both grey and full store the real forward/backward neighbour element types."""
        config = config_system_single_type

        # Expected forward types = element of each atom in atom 0's rcut shell.
        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]
        expected = list(np.array(system_binary_fcc.types)[forward_neighbors])

        for mode in ("grey", "full"):
            config.atomicenvironment.atom_coloring_mode = mode
            fwd, bwd = _build_trivial_series(config, system_binary_fcc)

            assert fwd["types"] is not None, mode
            assert list(fwd["types"]) == expected, mode
            # A genuine multi-element environment carries both species.
            assert set(fwd["types"]) == {"Ni", "Fe"}, mode
            assert bwd["types"] is not None, mode

    def test_event_id_differs_by_mode_same_types(
        self, system_binary_fcc: System, config_system_single_type: Config
    ) -> None:
        """Full colouring changes the reference hash (event_id), but not the stored types."""
        config = config_system_single_type

        config.atomicenvironment.atom_coloring_mode = "grey"
        grey_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        config.atomicenvironment.atom_coloring_mode = "full"
        full_fwd, _ = _build_trivial_series(config, system_binary_fcc)

        # Colouring changes the hash...
        assert grey_fwd["event_id"] != full_fwd["event_id"]
        # ...but the stored types are identical (storage is mode-independent).
        assert list(grey_fwd["types"]) == list(full_fwd["types"])

    def test_types_aligned_with_positions(
        self, system_binary_fcc: System, config_system_single_type: Config
    ) -> None:
        """``types`` is element-wise aligned with the position slices (the ordering invariant).

        Equal length alone is tautological (everything is sliced by the same index
        array), so we tie both the type label *and* the position row of each entry
        back to the same global atom index. For this trivial event
        (min1==saddle==min2==system.positions) every position slice reduces to
        ``system.positions[neighbour_list]``.
        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        nl = NeighborsList(
            system_binary_fcc,
            config.atomicenvironment.rnei,
            config.atomicenvironment.rcut,
        )
        forward_neighbors = nl.neighbors_list["rcut"][0]

        fwd, _ = _build_trivial_series(config, system_binary_fcc)

        n = len(forward_neighbors)
        assert len(fwd["types"]) == n
        # Each entry's type and its three position rows all map to the same atom.
        for k, atom in enumerate(forward_neighbors):
            assert fwd["types"][k] == system_binary_fcc.types[atom]
            for key in ("initial_positions", "saddle_positions", "final_positions"):
                assert np.array_equal(fwd[key][k], system_binary_fcc.positions[atom]), (
                    key,
                    k,
                )


class TestGreyDedupSpeciesGating:
    """Specify the coloring-mode gating of de-dup in ``is_new_event``.

    ``is_new_event`` mirrors the PSR / classification paths: in "full" mode it
    feeds the REAL element types to IRA, so geometrically-identical but
    species-swapped saddles stay DISTINCT; in "grey" mode every atom is greyed to a
    single dummy label (``'X'``), so the same swapped saddles de-duplicate as one
    event (grey-alloy approximation). These tests pin both sides of that gate.
    """

    def _insert_forward_row(self, table: ReferenceEventTable, fwd: pd.Series) -> None:
        """Append a single forward event ``Series`` as a row of ``table.table``.

        Parameters
        ----------
        table : ReferenceEventTable
            The reference table whose ``table`` DataFrame is appended to.
        fwd : pd.Series
            The forward event series to insert as a single row.

        """
        fwd = fwd.copy()
        fwd["idx_ref"] = 0
        fwd["idx_backward"] = 0
        table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)

    def test_grey_swapped_species_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Grey mode merges swapped-species, geometrically-identical duplicates.

        A trivial (min1==saddle==min2) event is stored, then an identical-geometry
        event with every Ni<->Fe swapped is queried. The grey ``event_id`` hash is
        species-blind so both share an id and reach the IRA de-dup path; that path
        greys every atom to ``'X'``, so the geometries match and ``is_new_event``
        reports the swapped event as a duplicate.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``grey`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "grey"

        pos = system_binary_fcc.positions
        cell = system_binary_fcc.cell
        types = list(system_binary_fcc.types)
        swapped = ["Fe" if t == "Ni" else "Ni" for t in types]

        table = ReferenceEventTable(config)
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=types,
        )
        self._insert_forward_row(table, fwd)

        fwd_swapped, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=swapped,
        )

        # Grey hash is species-blind: identical geometry -> identical event_id,
        # so the swapped event reaches the (grey-gated) IRA de-dup path.
        assert fwd_swapped["event_id"] == fwd["event_id"]
        # ...the stored/queried element types still genuinely differ (storage is
        # mode-independent; greying to 'X' happens inside is_new_event).
        assert list(fwd_swapped["types"]) != list(fwd["types"])

        # Grey-gated de-dup: types are greyed to 'X', so the swapped event MERGES.
        assert table.is_new_event(fwd_swapped) is False

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False

    def test_full_swapped_species_not_merged(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """Full mode keeps swapped-species saddles distinct (species-aware de-dup).

        Same geometry as the grey case, but in "full" mode ``is_new_event`` feeds
        the real (swapped) element types to IRA, so the match is rejected and the
        swapped event is reported as new.

        Parameters
        ----------
        system_binary_fcc : object
            Binary Ni/Fe FCC ``System`` fixture (alternating species).
        config_system_single_type : object
            Loaded ``Config`` fixture; its coloring mode is set to ``full`` here.

        """
        config = config_system_single_type
        config.atomicenvironment.atom_coloring_mode = "full"

        pos = system_binary_fcc.positions
        cell = system_binary_fcc.cell
        types = list(system_binary_fcc.types)
        swapped = ["Fe" if t == "Ni" else "Ni" for t in types]

        table = ReferenceEventTable(config)
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=types,
        )
        self._insert_forward_row(table, fwd)

        fwd_swapped, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.5,
            dE_backward=0.5,
            cell=cell,
            types=swapped,
        )

        # Full mode compares real element types (species-aware event_id and/or
        # type-aware IRA), so the swapped saddle is not merged -> reported as new.
        assert table.is_new_event(fwd_swapped) is True

        # Sanity control: the byte-identical event IS recognised as a duplicate.
        assert table.is_new_event(fwd) is False


@pytest.fixture
def recorded_saddle_row() -> pd.Series:
    """Extract the moving atom's 6.5 Å neighbourhood from the Fe replay."""
    path = (
        Path(__file__).parent / "data/partn_replay/avoff_swapped_boundary_bystander.npz"
    )
    with np.load(path, allow_pickle=False) as replay:
        initial, saddle = replay["res_min1"], replay["res_saddle"]
        cell = np.diag(replay["box_hi"] - replay["box_lo"])
        pbc = np.ones(3, dtype=bool)
        mover = move_atom(initial, saddle, cell, pbc)
        offsets = _minimum_image(initial - initial[mover], cell, pbc)
        neighbours = np.flatnonzero(np.linalg.norm(offsets, axis=1) <= 6.5)
        local_mover = int(np.searchsorted(neighbours, mover))
        assert (mover, len(neighbours), local_mover) == (922, 112, 85)
        return pd.Series(
            dict(
                event_id="Fe replay",
                energy_barrier=0.6,
                saddle_positions=_minimum_image(
                    saddle[neighbours] - saddle[mover], cell, pbc
                ),
                types=list(replay["res_types"][neighbours]),
                move_atom_idx=local_mover,
            )
        )


def test_translated_saddle_is_a_duplicate(
    config_system_single_type: Config, recorded_saddle_row: pd.Series
) -> None:
    """Real IRA registers a translated saddle with the same topology and nearby barrier."""
    pytest.importorskip("ira_mod")
    config_system_single_type.psr.matching_score_thr = 0.3
    table = ReferenceEventTable(config_system_single_type)
    table.table = pd.DataFrame([recorded_saddle_row])
    candidate = recorded_saddle_row.copy()
    candidate["saddle_positions"] = candidate.saddle_positions + [3.0, -1.0, 2.0]
    candidate["energy_barrier"] = 0.601
    assert table.is_new_event(candidate) is False
    candidate["event_id"] = "other"
    assert table.is_new_event(candidate) is True


def test_rejected_registration_with_same_topology_is_a_new_event(
    config_system_single_type: Config,
    recorded_saddle_row: pd.Series,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed match between the supplied saddles leaves a distinct event."""
    config_system_single_type.psr.matching_score_thr = 0.3
    table = ReferenceEventTable(config_system_single_type)
    table.table = pd.DataFrame([recorded_saddle_row])
    candidate = recorded_saddle_row.copy()
    candidate["saddle_positions"] = np.roll(candidate.saddle_positions, 1, axis=0)
    candidate.saddle_positions[1, 2] += 0.6
    candidate["move_atom_idx"] += 1
    match = MagicMock()
    match.return_value.is_ok.return_value = False
    monkeypatch.setattr(PointSetRegistration, "match_and_check", match)

    assert table.is_new_event(candidate) is True
    match.assert_called_once()
    args, centres = match.call_args
    np.testing.assert_array_equal(args[2], candidate.saddle_positions)
    np.testing.assert_array_equal(args[5], recorded_saddle_row.saddle_positions)
    assert args[6] == 0.3
    assert centres == dict(
        candidate1=candidate.move_atom_idx, candidate2=recorded_saddle_row.move_atom_idx
    )


class _SpyRateConstant:
    """Record each ``compute`` call; fail on the barriers listed in ``fail``."""

    def __init__(self, fail: tuple[float, ...] = ()) -> None:
        self.calls = []
        self.fail = fail

    def compute(self, rate_requests: list[RateRequest]) -> list:
        self.calls.append(list(rate_requests))
        results = []
        for request in rate_requests:
            if request.dE in self.fail:
                backup = RateConstantOutput(k=-request.dE, prefactor=1.0)
                results.append(
                    Err(
                        ErrorInfo(
                            type=RateConstantError.COMPUTATION_FAILED,
                            message="spy failure",
                            variables={"backup": backup},
                        )
                    )
                )
            else:
                results.append(
                    Ok(RateConstantOutput(k=10.0 * request.dE, prefactor=10.0))
                )
        return results


class TestAddEventsRateConstant:
    """``add_events`` computes the rate constants of all new events in one call.

    ``is_valid_new_event`` is replaced by prepared results: event A keeps forward
    and backward (2 rows), event B is rejected, event C keeps only forward (1 row).
    """

    @staticmethod
    def _event(dE_forward: float, dE_backward: float) -> EventSearchOutput:
        positions = np.zeros((2, 3))
        return EventSearchOutput(
            central_atom_index=0,
            min1_positions=positions,
            saddle_positions=positions + 1.0,
            min2_positions=positions + 2.0,
            dE_forward=dE_forward,
            dE_backward=dE_backward,
            move_atom_index=1,
        )

    @staticmethod
    def _rows(n: int) -> Ok:
        return Ok(pd.DataFrame({"energy_barrier": [0.0] * n, "k": [np.nan] * n}))

    def _add(
        self,
        config: object,
        monkeypatch: pytest.MonkeyPatch,
        rate_constant: _SpyRateConstant,
        loggers: object = None,
    ) -> tuple[ReferenceEventTable, list[EventSearchOutput]]:
        events = [self._event(0.1, 0.2), self._event(0.3, 0.4), self._event(0.5, 0.6)]
        validations = iter(
            [
                self._rows(2),
                Err(ErrorInfo(type=ErrorType.EVENT_NOT_NEW, message="")),
                self._rows(1),
            ]
        )
        table = ReferenceEventTable(
            config, rate_constant=rate_constant, loggers=loggers
        )
        monkeypatch.setattr(
            table, "is_valid_new_event", lambda **kwargs: next(validations)
        )
        table.add_events(events)
        return table, events

    def test_one_call_with_a_request_per_added_row(
        self, config_system_single_type: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One ``compute`` call; forward from min1, backward from min2, same saddle."""
        spy = _SpyRateConstant()
        table, (a, _, c) = self._add(config_system_single_type, monkeypatch, spy)

        assert len(spy.calls) == 1
        requests = spy.calls[0]
        assert [r.dE for r in requests] == [0.1, 0.2, 0.5]
        assert requests[0].initial_positions is a.min1_positions
        assert requests[1].initial_positions is a.min2_positions
        assert requests[2].initial_positions is c.min1_positions
        assert requests[0].saddle_positions is requests[1].saddle_positions
        assert requests[2].saddle_positions is c.saddle_positions
        assert all(r.central_atom == 1 for r in requests)

        assert list(table.table["k"]) == pytest.approx([1.0, 2.0, 5.0])
        assert list(table.table["k_pref"]) == pytest.approx([10.0, 10.0, 10.0])

    def test_failed_rate_uses_the_backup(
        self, config_system_single_type: object, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A row whose rate fails gets the backup of the error, with a warning."""
        spy = _SpyRateConstant(fail=(0.2,))
        loggers = MagicMock()
        table, _ = self._add(config_system_single_type, monkeypatch, spy, loggers)

        assert list(table.table["k"]) == pytest.approx([1.0, -0.2, 5.0])
        assert list(table.table["k_pref"]) == pytest.approx([10.0, 1.0, 10.0])
        # the backward row of event A (idx_ref 1) is the one that failed
        loggers.warning.assert_called_once_with(
            "log",
            "Rate constant compute failed on reference event 1 (spy failure), "
            "use backup value k0 = 1 ps^-1",
        )


class TestActiveAddEventsRateConstant:
    """``ActiveEventTable.add_events`` and the rate constant of its events."""

    def test_reference_prefactor_with_active_barrier(
        self, config_system_single_type: object
    ) -> None:
        """The rate uses the reference prefactor and the active barrier's Boltzmann factor."""
        config = config_system_single_type
        reference_table = ReferenceEventTable(config)
        reference_table.table = pd.DataFrame({"idx_ref": [0, 1], "k_pref": [2.0, 4.0]})
        events = [
            EventRefinementOutput(
                central_atom_index=3,
                saddle_positions=np.zeros((2, 3)),
                E_saddle=0.0,
                dE_forward=dE,
                num_reference_event=num_reference_event,
                refined="T",
            )
            for dE, num_reference_event in ((0.5, 1), (0.3, 0))
        ]

        active_table = ActiveEventTable(config)
        active_table.add_events(events, reference_table)

        kT = constants.KB * config.rateconstant.T
        assert list(active_table.table["k_pref"]) == pytest.approx([4.0, 2.0])
        assert list(active_table.table["k"]) == pytest.approx(
            [4.0 * np.exp(-0.5 / kT), 2.0 * np.exp(-0.3 / kT)]
        )


class TestSearchDetailsDoNotReachTheTable:
    """The optional search details go to the log and the info file only."""

    def test_same_rows_with_and_without_details(
        self, system_binary_fcc: object, config_system_single_type: object
    ) -> None:
        """A result with every detail set adds the same rows as one without."""
        config = config_system_single_type
        pos = np.array(system_binary_fcc.positions, dtype=float)
        saddle, final = pos.copy(), pos.copy()
        saddle[0, 0] += 0.9
        final[0, 0] += 1.8

        def event() -> EventSearchOutput:
            return EventSearchOutput(
                central_atom_index=0,
                min1_positions=pos.copy(),
                saddle_positions=saddle.copy(),
                min2_positions=final.copy(),
                dE_forward=0.6,
                dE_backward=0.4,
                move_atom_index=0,
                cell=np.array(system_binary_fcc.cell),
                types=list(system_binary_fcc.types),
            )

        detailed = event()
        detailed.backend = "dimer"
        detailed.seed = 7
        detailed.e_initial, detailed.e_saddle, detailed.e_final = -10.0, -9.4, -9.6
        detailed.force_calls, detailed.force_calls_endpoints = 100, 50
        detailed.curvature = -2.0
        detailed.mode = np.ones_like(pos) / np.sqrt(pos.size)

        tables = []
        for ev in (event(), detailed):
            table = ReferenceEventTable(config)
            results = table.add_events([ev])
            assert [r.is_ok() for r in results] == [True]
            tables.append(table.table)

        plain, with_details = tables
        assert len(plain) > 0
        assert list(plain.columns) == list(with_details.columns)
        assert list(plain.index) == list(with_details.index)
        for column in plain.columns:
            for a, b in zip(plain[column], with_details[column], strict=True):
                if isinstance(a, (np.ndarray, list)):
                    assert np.array_equal(np.asarray(a), np.asarray(b)), column
                elif isinstance(a, float) and np.isnan(a):
                    assert np.isnan(b), column
                else:
                    assert a == b, column


def test_event_series_matches_full_system_neighbors_list(
    system_binary_fcc: object, config_system_single_type: object
) -> None:
    """Ids and crops, atom order included, are those of full-system neighbor lists.

    Atom 0, at the box corner, moves in the saddle and in min2 and every other atom
    is jittered, so the three environments differ and wrap across the box. The
    reference builds a ``NeighborsList`` of the whole system per configuration.

    Parameters
    ----------
    system_binary_fcc : object
        Binary Ni/Fe FCC ``System`` fixture (alternating species).
    config_system_single_type : object
        Loaded ``Config`` fixture; its coloring mode is set to ``full`` here.

    """
    config = config_system_single_type
    config.atomicenvironment.atom_coloring_mode = "full"
    rnei = config.atomicenvironment.rnei
    rcut = config.atomicenvironment.rcut
    cell = system_binary_fcc.cell
    types = list(system_binary_fcc.types)
    rng = np.random.default_rng(0)
    min1 = system_binary_fcc.positions
    saddle, min2 = (
        np.mod(min1 + rng.normal(0.0, 0.05, min1.shape), np.diag(cell))
        for _ in range(2)
    )
    saddle[0] = min1[0] + 0.4
    min2[0] = min1[0] + 0.8

    lists = [
        NeighborsList(System(positions=p, cell=cell), rnei, rcut).neighbors_list
        for p in (min1, saddle, min2)
    ]
    ids = [graph(nl["rnei"], nl["rcut"], atom_idx=[0], types=types)[0] for nl in lists]
    forward, backward = lists[0]["rcut"][0], lists[2]["rcut"][0]
    assert len(set(ids)) == 3
    assert sorted(forward) != sorted(backward)

    fwd, bwd = ReferenceEventTable(config)._build_event_series(
        min1_positions=min1,
        saddle_positions=saddle,
        min2_positions=min2,
        index_move=0,
        dE_forward=0.5,
        dE_backward=0.3,
        cell=cell,
        types=types,
    )

    assert [fwd["event_id"], fwd["id_saddle"], fwd["id_final"]] == ids
    assert [bwd["event_id"], bwd["id_saddle"], bwd["id_final"]] == ids[::-1]
    assert np.array_equal(fwd["initial_positions"], min1[forward])
    assert np.array_equal(bwd["initial_positions"], min2[backward])
