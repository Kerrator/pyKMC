"""Falsification cases for the independent eight-search rank comparator."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import cKDTree

from pykmc.config import Config
from pykmc.dimer import _minimum_image
from pykmc.environments.graph_nauty import graph
from pykmc.search_rules import move_atom
from scripts import artn_rank_gate as gate


def replay_saddle_cluster() -> tuple[np.ndarray, int, str]:
    """Return the recorded Fe saddle within the initial minimum's 6.5 Å cutoff."""
    path = (
        Path(__file__).parent
        / "data"
        / "partn_replay"
        / "avoff_swapped_boundary_bystander.npz"
    )
    with np.load(path, allow_pickle=False) as replay:
        initial = replay["res_min1"]  # already oriented by the recorded search
        saddle = replay["res_saddle"]
        cell = np.diag(replay["box_hi"] - replay["box_lo"])
        types = replay["res_types"]
    pbc = np.ones(3, dtype=bool)
    mover = move_atom(initial, saddle, cell, pbc)
    initial_offsets = _minimum_image(initial - initial[mover], cell, pbc)
    neighbours = np.flatnonzero(np.linalg.norm(initial_offsets, axis=1) <= 6.5)
    local_mover = int(np.searchsorted(neighbours, mover))
    assert (mover, len(neighbours), local_mover) == (922, 112, 85)
    assert np.all(types[neighbours] == "Fe")
    # The replay's 8 conventional bcc cells per side give the lattice constant.
    # Keep two neighbour shells, cutting halfway between a and sqrt(2) * a.
    lattice_constant = cell[0, 0] / 8
    bond_cutoff = lattice_constant * (1 + np.sqrt(2)) / 2
    local_initial = initial_offsets[neighbours]
    first_neighbours = cKDTree(local_initial).query_ball_point(
        local_initial, bond_cutoff
    )
    for i, adjacent in enumerate(first_neighbours):
        adjacent.remove(i)
    (topology,) = graph(
        first_neighbours, {local_mover: list(range(len(neighbours)))}, [local_mover]
    )
    # Keep source atom order and join the saddle across the periodic face.
    local_saddle = _minimum_image(saddle[neighbours] - saddle[mover], cell, pbc)
    return local_saddle, local_mover, topology


def arms() -> list[dict]:
    """Use the replay saddle and its initial topology for each rank."""
    geom, move_atom, topology = replay_saddle_cluster()
    rows = [
        dict(
            center=c,
            seed=seed,
            force_calls=2,
            callback_calls=2,
            seconds=0.2,
            force_seconds=0.1,
            outcome="event",
            energy_barrier=0.6,
            event_id=topology,
            types=["Fe"] * len(geom),
            move_atom_idx=move_atom,
            saddle_positions=geom.tolist(),
        )
        for c, seed in zip(gate.CENTRES, gate.SEEDS, strict=True)
    ]
    return [
        dict(
            kind="run",
            head="fixture",
            cell=gate.CELL,
            potential=gate.POTENTIAL,
            config=gate.CONFIG,
            lammps=20250722,
            ranks=p,
            rows=deepcopy(rows),
        )
        for p in (1, 2, 4)
    ]


def config() -> Config:
    """Use the normal IRA implementation and the Fig. 2 matching threshold."""
    cfg = Config.from_ini_file("tests/data/input_Cu.in")
    cfg.control.reference_table = None
    cfg.psr.matching_score_thr = 0.3
    return cfg


@pytest.mark.parametrize(
    "field,value",
    [
        ("center", 1333.0),
        ("move_atom_idx", 1.5),
        ("energy_barrier", float("nan")),
        ("seed", 1),
        ("force_calls", 0),
        ("seconds", float("inf")),
        ("saddle_positions", [[0.0, 0.0, float("nan")]]),
    ],
)
def test_rejects_invalid_evidence(field: str, value: object) -> None:
    """Coerced indices, corrupt metrics and incomplete geometry cannot pass."""
    run = arms()
    run[1]["rows"][0][field] = value
    with pytest.raises(ValueError):
        gate.compare(run, config(), "fixture")


def test_requires_exact_arms_and_all_eight_requests() -> None:
    """Reject mislabelled ranks, settings and incomplete or duplicated requests."""
    for key, value in (("rows", []), ("ranks", True), ("config", "other")):
        run = arms()
        run[0][key] = value
        with pytest.raises(ValueError):
            gate.compare(run, config(), "fixture")
    run = arms()
    run[1]["rows"][1] = run[1]["rows"][0]
    with pytest.raises(ValueError):
        gate.compare(run, config(), "fixture")


def test_real_duplicate_and_barrier_with_independent_metrics() -> None:
    """Use registered IRA for a translated event and the inclusive one-meV bound."""
    pytest.importorskip("ira_mod")
    cfg = config()
    run = arms()
    r = run[1]["rows"][0]
    r["saddle_positions"] = (
        np.asarray(r["saddle_positions"]) + [3.0, -1.0, 2.0]
    ).tolist()
    r.update(energy_barrier=0.601, force_calls=30, seconds=60.0)
    assert gate.compare(run, cfg, "fixture") == 0
    r["energy_barrier"] = 0.6010001
    assert gate.compare(run, cfg, "fixture") == 1
    r.update(energy_barrier=0.6, event_id="other")
    assert gate.compare(run, cfg, "fixture") == 1


def test_non_events_are_inconclusive_only_when_matched() -> None:
    """Matching failures never satisfy the eight-event gate."""
    pytest.importorskip("ira_mod")
    cfg = config()
    run = arms()
    for a in run:
        r = a["rows"][0]
        for key in (
            "event_id",
            "saddle_positions",
            "move_atom_idx",
            "types",
            "energy_barrier",
        ):
            r.pop(key)
        r.update(
            outcome="max_force_calls",
            endpoints={},
            raw=dict.fromkeys(
                (
                    "e_start",
                    "e_saddle",
                    "e_min1",
                    "e_min2",
                    "delr1",
                    "delr2",
                    "curvature",
                )
            ),
        )
        r["raw"]["e_start"] = 0.0
    assert gate.compare(run, cfg, "fixture") == 2
    run[1]["rows"][0]["outcome"] = "eigenvalue_lost"
    assert gate.compare(run, cfg, "fixture") == 1


def test_registration_rejection_on_same_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A rejected registration fails the gate even at identical barrier and topology."""
    from unittest.mock import Mock
    from pykmc.psr import PointSetRegistration

    # Keep the third-party native solver out of this fault-injection test.
    # Geometry-dependent return values verify which saddle rows reach it.
    def match(
        self: PointSetRegistration,
        n1: int,
        t1: list,
        x1: np.ndarray,
        n2: int,
        t2: list,
        x2: np.ndarray,
        threshold: float,
        **centres: int,
    ) -> Mock:
        assert threshold == 0.3
        assert centres["candidate1"] == centres["candidate2"]
        return Mock(is_ok=lambda: np.array_equal(x1, x2))

    monkeypatch.setattr(PointSetRegistration, "match_and_check", match)
    run = arms()
    run[2]["rows"][3]["saddle_positions"][1][2] += 0.6
    assert gate.compare(run, config(), "fixture") == 1


def test_assembled_faces_reach_the_centred_catalogue_without_changing_evidence() -> (
    None
):
    """Assemble a two-shard result and compare its catalogue row with the facade's."""
    from copy import deepcopy
    from types import SimpleNamespace

    from pykmc.dimer import DimerResult
    from pykmc.engine.artn_ranks import DistributedSpace
    from pykmc.event_table import ReferenceEventTable
    from pykmc.eventsearch.eventsearch import EventSearch, SearchRequest
    from pykmc.eventsearch.strategies.artn import ArtnStrategy
    from pykmc.eventsearch.strategies.dimer import dimer_search_result

    cluster, mover, _ = replay_saddle_cluster()
    # Place the source-centred cluster inside the box, away from its centre
    # and the frozen boundary atom, so the catalogue must translate it.
    cluster += 10.0
    cell = np.eye(3) * 28.55312
    pbc = np.ones(3, bool)
    # The frozen bystander is reconstructed from the start by collect_result.
    start = np.vstack([cluster, [-1e-17, 0, 0]])
    movable = np.arange(len(cluster))
    req = SearchRequest(mover, start, cell, np.full(len(start), "Fe"), pbc, 9, movable)
    saddle = cluster.copy()
    saddle[mover, 0] += 0.1
    mode = np.zeros_like(saddle)
    mode[mover, 0] = 1.0
    raw = DimerResult(
        True,
        None,
        saddle,
        cluster,
        cluster,
        -10.0,
        -9.4,
        -10.0,
        -10.0,
        0.0,
        0.0,
        -1.0,
        mode,
        2,
        3,
        9,
    )
    names = ("saddle_positions", "min1_positions", "min2_positions", "mode")
    leader, peer = np.array_split(movable, 2)
    peer_rows = [getattr(raw, k)[peer] for k in names]
    comm = SimpleNamespace(gather=lambda local, root: [local, (peer, peer_rows)])
    shard = raw._replace(**{k: getattr(raw, k)[leader] for k in names})
    result = DistributedSpace.collect_result(
        SimpleNamespace(home=leader, start=start, comm=comm), shard, cell, pbc
    )
    for k in names[:3]:
        np.testing.assert_array_equal(getattr(result, k)[-1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(getattr(result, k)[movable], getattr(raw, k))
    # A shared endpoint array must survive both translations untouched.
    result = result._replace(min2_positions=result.min1_positions)
    evidence = deepcopy(result)
    cfg = config()
    row = gate.event_record(result, req, cfg)
    for k in names:
        np.testing.assert_array_equal(getattr(result, k), getattr(evidence, k))
    # Construct the oracle via the production event path with distinct storage.
    unaliased = result._replace(
        min1_positions=result.min1_positions.copy(),
        min2_positions=result.min2_positions.copy(),
        saddle_positions=result.saddle_positions.copy(),
    )
    facade = EventSearch(ArtnStrategy(cfg))
    facade._cell = cell
    facade.results = [
        dimer_search_result(unaliased, req, cfg.eventsearch.delr_thr, backend="artn")
    ]
    (ev,) = facade.get_successes_results()
    expected, _ = ReferenceEventTable(cfg)._build_event_series(
        ev.min1_positions,
        ev.saddle_positions,
        ev.min2_positions,
        ev.move_atom_index,
        ev.dE_forward,
        ev.dE_backward,
        ev.cell,
        ev.types,
    )
    assert row["outcome"] == "event" and row["event_id"] == expected.event_id
    assert row["energy_barrier"] == expected.energy_barrier
    assert row["move_atom_idx"] == expected.move_atom_idx and row["types"] == list(
        expected.types
    )
    np.testing.assert_array_equal(row["saddle_positions"], expected.saddle_positions)
    assert (
        abs(row["saddle_positions"][row["move_atom_idx"]][0] - saddle[mover, 0]) > 0.1
    )
    failed = result._replace(
        success=False,
        reason="max_force_calls",
        e_min2=None,
        min2_positions=None,
        delr2=None,
    )
    partial = gate.event_record(failed, req, cfg)
    np.testing.assert_array_equal(
        partial["endpoints"]["saddle_positions"], evidence.saddle_positions
    )
    assert (
        "energy_barrier" not in partial and "min2_positions" not in partial["endpoints"]
    )


def test_serial_gate_allows_only_enumeration_and_exact_face_aliases() -> None:
    """Global raw fields remain exact in atom order, including non-event endpoints."""
    from pykmc.dimer import DimerResult

    lengths = np.full(3, 28.55312)
    x = np.broadcast_to(np.arange(1999)[:, None] * 0.01, (1999, 3)).copy()
    x[0, 0] = lengths[0]
    base = DimerResult(
        True,
        None,
        x.copy(),
        x.copy(),
        x.copy(),
        0.0,
        0.6,
        0.0,
        0.0,
        0.0,
        1.0,
        -1.0,
        x.copy(),
        1,
        2,
        0,
    )
    before, after = arms()[0], arms()[0]
    for i, (a, b) in enumerate(zip(before["rows"], after["rows"], strict=True)):
        raw = base._replace(seed=a["seed"])
        a["raw_fingerprint"] = gate.raw_fingerprint(raw, lengths)
        canonical = {
            k: np.where(getattr(raw, k) == lengths, 0.0, getattr(raw, k))
            for k in gate._POSITIONS
        }
        corrected = raw._replace(**canonical)
        b["raw_fingerprint"] = gate.raw_fingerprint(corrected, lengths)
        if i == 0:
            # Reverse order, preserving the local point set, types and mover.
            b["saddle_positions"].reverse()
            b["types"].reverse()
            b["move_atom_idx"] = len(b["types"]) - 1 - a["move_atom_idx"]
        if i == 1:
            for r, value in ((a, raw), (b, corrected)):
                partial = value._replace(
                    success=False,
                    reason="max_force_calls",
                    min2_positions=None,
                    e_min2=None,
                    delr2=None,
                )
                for k in (
                    "types",
                    "move_atom_idx",
                    "energy_barrier",
                    "event_id",
                    "saddle_positions",
                ):
                    r.pop(k)
                r.update(
                    outcome="max_force_calls",
                    endpoints={
                        k: getattr(partial, k).tolist()
                        for k in ("saddle_positions", "min1_positions")
                    },
                    raw={
                        k: getattr(partial, k)
                        for k in (
                            "e_start",
                            "e_saddle",
                            "e_min1",
                            "e_min2",
                            "delr1",
                            "delr2",
                            "curvature",
                        )
                    },
                    raw_fingerprint=gate.raw_fingerprint(partial, lengths),
                )

    def verdict(candidate: dict) -> int:
        return gate.compare_serial(before, candidate, "fixture", "fixture", lengths)

    assert verdict(after) == 0
    # An uncorrected candidate fails even when its raw bytes match the baseline.
    assert verdict(before) == 1
    assert gate.compare_serial(after, before, "fixture", "fixture", lengths) == 1
    assert before["rows"][1]["endpoints"]["saddle_positions"][0][0] == lengths[0]
    # Preserve the actual global raw values even outside the recorded local saddle.
    aligned = base._replace(
        seed=after["rows"][3]["seed"],
        **{
            k: np.where(getattr(base, k) == lengths, 0.0, getattr(base, k))
            for k in gate._POSITIONS
        },
    )
    for outside in (-1e-17, np.nextafter(lengths[0], np.inf)):
        changed = deepcopy(after)
        invalid = aligned.saddle_positions.copy()
        invalid[0, 0] = outside
        changed["rows"][3]["raw_fingerprint"] = gate.raw_fingerprint(
            aligned._replace(saddle_positions=invalid), lengths
        )
        assert verdict(changed) == 1
        with pytest.raises(ValueError, match="baseline.*outside"):
            gate.compare_serial(changed, after, "fixture", "fixture", lengths)
    for k in (*gate._RAW_ARRAYS, "delr2"):
        changed = deepcopy(after)
        if k == "delr2":
            observed = aligned._replace(delr2=np.nextafter(aligned.delr2, np.inf))
        else:
            coords = getattr(aligned, k).copy()
            coords[-1, 2] = np.nextafter(coords[-1, 2], np.inf)
            observed = aligned._replace(**{k: coords})
        changed["rows"][3]["raw_fingerprint"] = gate.raw_fingerprint(observed, lengths)
        assert verdict(changed) == 1
    changed = deepcopy(after)
    swapped = aligned.saddle_positions.copy()
    swapped[[-2, -1]] = swapped[[-1, -2]]
    changed["rows"][3]["raw_fingerprint"] = gate.raw_fingerprint(
        aligned._replace(saddle_positions=swapped), lengths
    )
    assert verdict(changed) == 1
    changed = deepcopy(after)
    changed["rows"][0]["move_atom_idx"] = (
        changed["rows"][0]["move_atom_idx"] + 1
    ) % len(changed["rows"][0]["types"])
    assert verdict(changed) == 1
    changed = deepcopy(after)
    pos = changed["rows"][0]["saddle_positions"][1]
    pos[0] = np.nextafter(pos[0], np.inf)
    assert verdict(changed) == 1
    changed = deepcopy(after)
    changed["rows"][0]["types"][0] = "Ni"
    with pytest.raises(ValueError):
        verdict(changed)
    changed = deepcopy(after)
    changed["rows"][1]["endpoints"]["saddle_positions"][0][0] = np.nextafter(
        lengths[0], 0
    )
    with pytest.raises(ValueError, match="partial coordinates"):
        verdict(changed)
    changed = deepcopy(after)
    del changed["rows"][2]["raw_fingerprint"]
    with pytest.raises(KeyError):
        verdict(changed)
