"""Run or compare the eight pinned Fig. 2 requests for the ARTn Mac rank check.

Run the `run` and `crop-smoke` subcommands only within an approved MPI allocation.
Each invocation runs one worker on its communicator and creates one JSON file.
The comparator reads three such files. A matching non-event has no duplicate
or barrier to test and leaves the eight-event criterion unproved.
"""

from __future__ import annotations

import argparse
from decimal import Decimal
from contextlib import contextmanager
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CENTRES = (1333, 1333, 1531, 1513, 1513, 1311, 1333, 1311)
SEEDS = (
    494828123,
    712798592,
    2540808125,
    348537163,
    875863341,
    1934173456,
    900186024,
    2134761458,
)
CELL = "6afbaee3ca440c4bbb42c788a29c3cf15c7a18413484e289c5c0d04e4853d5f1"
POTENTIAL = "40a3fb9afa7da378dbb94a705d4c8143055221c8b1fbbd351a6ca0aa168020dc"
CONFIG = "9a819a57a11d746572cf3399c0333d727452a18890f88b246a36249ae617478d"


def checked(path: Path, digest: str) -> Path:
    """Verify exact Fig. 2 evidence bytes before parsing them."""
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"unexpected SHA-256 for {path}")
    return path.resolve()


def checkout() -> str:
    """Pin local package imports to the same clean checkout on every arm."""
    for hooks in (sys.meta_path, sys.path_hooks):
        hooks[:] = [h for h in hooks if "__editable___pykmc" not in str(h).lower()]
    sys.path.insert(0, str(ROOT))
    import pykmc

    if Path(pykmc.__file__).resolve().parent != ROOT / "pykmc":
        raise ValueError("pyKMC imported outside the gate checkout")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT).strip():
        raise ValueError("rank gate requires a clean implementation checkout")
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def root_action(comm: Any, action: Any) -> None:
    """Propagate leader-only output or event-construction errors to every peer."""
    message = None
    if comm.Get_rank() == 0:
        try:
            action()
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
    message = comm.bcast(message, root=0)
    if message:
        raise RuntimeError(message)


@contextmanager
def count_force_calls() -> Iterator[dict]:
    """Time underlying LAMMPS force runs, including full/crop setups and retries."""
    from lammps import lammps

    data = dict(calls=0, seconds=0.0)
    original = lammps.command

    def measured(instance: Any, command: str, *args: Any, **kwargs: Any) -> Any:
        if not command.startswith("run "):
            return original(instance, command, *args, **kwargs)
        if command not in ("run 0 post no", "run 1 pre no post no"):
            raise ValueError(f"unknown number of evaluations for {command!r}")
        before = time.perf_counter()
        result = original(instance, command, *args, **kwargs)
        data["seconds"] += time.perf_counter() - before
        data["calls"] += 1
        return result

    with patch.object(lammps, "command", measured):
        yield data


def event_record(raw: Any, request: Any, config: Any) -> dict:
    """Build the same forward reference row and apply pyKMC's orientation rule."""
    from pykmc.event_table import ReferenceEventTable
    from pykmc.eventsearch.eventsearch import EventSearch
    from pykmc.eventsearch.strategies.artn import ArtnStrategy
    from pykmc.eventsearch.strategies.dimer import dimer_search_result

    result = dimer_search_result(
        raw, request, config.eventsearch.delr_thr, backend="artn"
    )
    out = dict(
        center=request.central_atom,
        seed=request.seed,
        callback_calls=raw.force_calls_total,
    )
    if not result.is_ok():
        # Preserve partial returns without inventing an oriented event barrier.
        out["raw"] = {
            key: (None if getattr(raw, key) is None else float(getattr(raw, key)))
            for key in (
                "e_start",
                "e_saddle",
                "e_min1",
                "e_min2",
                "delr1",
                "delr2",
                "curvature",
            )
        }
        out["endpoints"] = {
            key: np.asarray(getattr(raw, key)).tolist()
            for key in ("saddle_positions", "min1_positions", "min2_positions")
            if getattr(raw, key) is not None
        }
        out["outcome"] = raw.reason or result.err_value().type.name
        return out
    # The production catalogue consumes the facade's centred successes. Its
    # translation modifies each array in place, so detach the three arrays
    # individually, including when the raw result aliases two endpoints.
    oriented = result.ok_value()
    oriented.min1_positions = oriented.min1_positions.copy()
    oriented.saddle_positions = oriented.saddle_positions.copy()
    oriented.min2_positions = oriented.min2_positions.copy()
    search = EventSearch(ArtnStrategy(config))
    search._cell = request.cell
    search.results = [result]
    (ev,) = search.get_successes_results()
    table = ReferenceEventTable(config)
    row, _ = table._build_event_series(
        ev.min1_positions,
        ev.saddle_positions,
        ev.min2_positions,
        ev.move_atom_index,
        ev.dE_forward,
        ev.dE_backward,
        request.cell,
        request.types,
    )
    out.update(
        outcome="event",
        event_id=row.event_id,
        energy_barrier=float(row.energy_barrier),
        types=list(row.types),
        saddle_positions=row.saddle_positions.tolist(),
        move_atom_idx=int(row.move_atom_idx),
    )
    return out


_POSITIONS = ("saddle_positions", "min1_positions", "min2_positions")
_RAW_ARRAYS = (*_POSITIONS, "mode")
_RAW_SCALARS = (
    "success",
    "reason",
    "e_start",
    "e_saddle",
    "e_min1",
    "e_min2",
    "curvature",
    "delr1",
    "delr2",
    "force_calls_saddle",
    "force_calls_total",
    "seed",
)


def _array_fingerprint(values: np.ndarray, lengths: np.ndarray | None) -> dict:
    """Fingerprint a field in its original atom order, preserving type and shape."""
    a = np.asarray(values)
    if (
        a.ndim != 2
        or a.shape[1] != 3
        or a.dtype.kind != "f"
        or not np.isfinite(a).all()
    ):
        raise ValueError("invalid P=1 raw array")
    original = hashlib.sha256(a.tobytes()).hexdigest()
    canonical = original
    count = 0
    outside = 0
    if lengths is not None:
        upper = a == lengths
        count = int(upper.sum())
        outside = int(np.count_nonzero((a < 0) | (a >= lengths)))
        if count:
            image = a.copy()
            image[upper] = 0
            canonical = hashlib.sha256(image.tobytes()).hexdigest()
    return dict(
        shape=list(a.shape),
        dtype=a.dtype.str,
        sha256=original,
        image_sha256=canonical,
        upper_face_components=count,
        outside_components=outside,
    )


def raw_fingerprint(raw: Any, lengths: np.ndarray) -> dict:
    """Retain all P=1 raw result fields and report exact upper-face aliases."""
    scalars = {k: getattr(raw, k) for k in _RAW_SCALARS}
    fields = {
        k: None
        if getattr(raw, k) is None
        else _array_fingerprint(getattr(raw, k), lengths if k in _POSITIONS else None)
        for k in _RAW_ARRAYS
    }
    return dict(scalars=scalars, fields=fields)


def _serial_key(row: dict, lengths: np.ndarray) -> tuple[dict, list, dict]:
    """Build an exact P=1 comparison key and retain representation diagnostics."""
    record = dict(row)
    del record["seconds"], record["force_seconds"]
    fp = record.pop("raw_fingerprint")
    scalars, fields = fp["scalars"], fp["fields"]
    if set(scalars) != set(_RAW_SCALARS) or set(fields) != set(_RAW_ARRAYS):
        raise ValueError("incomplete P=1 raw fingerprint")
    if (
        type(scalars["success"]) is not bool
        or type(scalars["seed"]) is not int
        or scalars["seed"] != row["seed"]
    ):
        raise ValueError("invalid P=1 raw result identity")
    if (
        type(scalars["force_calls_total"]) is not int
        or scalars["force_calls_total"] != row["callback_calls"]
    ):
        raise ValueError("invalid P=1 raw callback count")
    expected = {
        "saddle_positions": scalars["e_saddle"],
        "min1_positions": scalars["e_min1"],
        "min2_positions": scalars["e_min2"],
        "mode": scalars["e_saddle"],
    }
    canonical = {}
    for k, field in fields.items():
        if field is None:
            if expected[k] is not None or scalars["success"]:
                raise ValueError(f"missing P=1 raw {k}")
            canonical[k] = None
            continue
        if (
            expected[k] is None
            or field["shape"] != [1999, 3]
            or np.dtype(field["dtype"]).kind != "f"
        ):
            raise ValueError(f"invalid P=1 raw {k} shape, dtype or energy")
        for digest in ("sha256", "image_sha256"):
            if (
                not isinstance(field[digest], str)
                or len(field[digest]) != 64
                or any(c not in "0123456789abcdef" for c in field[digest])
            ):
                raise ValueError(f"invalid {k} {digest}")
        count = field["upper_face_components"]
        if (
            type(count) is not int
            or not 0 <= count <= 3 * 1999
            or (k == "mode" and count)
        ):
            raise ValueError(f"invalid {k} upper-face count")
        outside = field["outside_components"]
        if type(outside) is not int or not count <= outside <= 3 * 1999:
            raise ValueError(f"invalid {k} outside-component count")
        if k == "mode" and outside:
            raise ValueError("periodic counts on a mode")
        if not count and field["sha256"] != field["image_sha256"]:
            raise ValueError(f"inconsistent {k} fingerprints")
        canonical[k] = (field["shape"], field["dtype"], field["image_sha256"])
    record["raw_fingerprint"] = {"scalars": scalars, "fields": canonical}
    points = []
    if row["outcome"] == "event":
        # SciPy can return neighbours in a different order for the same
        # periodic point set. Move types and the central marker with the points.
        mover = record.pop("move_atom_idx")
        for i, (t, xyz) in enumerate(
            zip(record.pop("types"), record.pop("saddle_positions"), strict=True)
        ):
            points.append((t, *(float(v).hex() for v in xyz), i == mover))
        record["points"] = sorted(points)
    else:
        for k, xyz in record.pop("endpoints").items():
            if _array_fingerprint(np.asarray(xyz), lengths) != fields[k]:
                raise ValueError(f"partial coordinates differ from raw {k} fingerprint")
    return record, points, fields


def compare_serial(
    before: dict, after: dict, before_head: str, after_head: str, lengths: np.ndarray
) -> int:
    """Compare all eight P=1 outcomes, quantities and global-order raw fields exactly."""
    for data, head in ((before, before_head), (after, after_head)):
        expected = dict(
            kind="run",
            head=head,
            cell=CELL,
            potential=POTENTIAL,
            config=CONFIG,
            lammps=20250722,
            ranks=1,
        )
        if (
            any(
                type(data[k]) is not type(v) or data[k] != v
                for k, v in expected.items()
            )
            or len(data["rows"]) != 8
        ):
            raise ValueError("incompatible P=1 evidence")
    if {k: v for k, v in before.items() if k not in ("rows", "head")} != {
        k: v for k, v in after.items() if k not in ("rows", "head")
    }:
        raise ValueError("P=1 run metadata differs outside revision")
    same = True
    for i, (old, new) in enumerate(zip(before["rows"], after["rows"], strict=True)):
        for row in (old, new):
            valid_row(row, CENTRES[i], SEEDS[i])
        a, points_a, fields_a = _serial_key(old, lengths)
        b, points_b, fields_b = _serial_key(new, lengths)
        equal = json.dumps(a, sort_keys=True, allow_nan=False) == json.dumps(
            b, sort_keys=True, allow_nan=False
        )
        for k in _POSITIONS:
            x, y = fields_a[k], fields_b[k]
            if x is not None and x["outside_components"] != x["upper_face_components"]:
                raise ValueError(
                    f"baseline {k} has coordinates outside the exact face alias"
                )
            if y is not None and y["outside_components"]:
                equal = False
                print(
                    f"  candidate {k} outside [0,L): {y['outside_components']} components"
                )
        same &= equal
        changed_order = points_a != points_b and sorted(points_a) == sorted(points_b)
        print(f"P=1 search={i} equal={equal} local_enumeration_changed={changed_order}")
        for k in _RAW_ARRAYS:
            x, y = fields_a[k], fields_b[k]
            if (
                x is not None
                and y is not None
                and (
                    x["sha256"] != y["sha256"]
                    or x["upper_face_components"]
                    or y["upper_face_components"]
                )
            ):
                print(
                    f"  {k}: raw_bytes_equal={x['sha256'] == y['sha256']} "
                    f"canonical_bytes_equal={x['image_sha256'] == y['image_sha256']} "
                    f"exact_upper_face_components={x['upper_face_components']}->{y['upper_face_components']}"
                )
    status = "PASS" if same else "DIFFERENT"
    print(f"ARTN_SERIAL_COMPARISON_COMPLETE {status}", flush=True)
    return 0 if same else 1


def crop_smoke(extension: Any, request: Any, options: dict) -> dict:
    """Compare the first live move on true full and cropped instances at P=2."""
    from pykmc.engine.artn_ranks import RankForces

    original = RankForces.__call__
    samples = []

    def observe(query: RankForces, x: np.ndarray) -> tuple[float, np.ndarray]:
        initial = query.engine.lmp.get_thermo("pe")
        energy, forces = original(query, x)
        samples.append(
            (
                query.engine.lmp.get_natoms(),
                query.space.home,
                x.copy(),
                energy - initial,
                forces.copy(),
            )
        )
        return energy, forces

    comm = extension.engine.comm
    if comm.Get_size() != 2:
        raise ValueError("crop smoke requires two worker ranks")
    with patch.object(RankForces, "__call__", observe):
        for crop in (False, True):
            extension.artn_search(
                **vars(request),
                options=dict(options, eam_force_crop=crop, nevalf_max=2),
            )
    if not all(comm.allgather(len(samples) == 2)):
        raise ValueError("expected one full and one cropped force call")
    full, cropped = samples
    ok = (
        full[0] == len(request.positions)
        and 0 < cropped[0] < 0.9 * full[0]
        and np.array_equal(full[1], cropped[1])
        and np.array_equal(full[2], cropped[2])
        and np.isfinite(full[3])
        and np.isfinite(cropped[3])
        and abs(full[3] - cropped[3]) <= 1e-8
        and np.isfinite(full[4]).all()
        and np.isfinite(cropped[4]).all()
        and np.allclose(full[4], cropped[4], atol=1e-9, rtol=0)
    )
    if not all(comm.allgather(bool(ok))):
        raise ValueError("the actual cropped force evaluation did not agree with full")
    return dict(
        full_atoms=int(full[0]),
        crop_atoms=int(cropped[0]),
        max_force_difference=max(
            comm.allgather(float(np.max(np.abs(full[4] - cropped[4]), initial=0)))
        ),
        energy_increment_difference=abs(full[3] - cropped[3]),
    )


def run(args: argparse.Namespace, head: str) -> None:
    """Execute one bounded arm on the worker communicator, serially by search."""
    from mpi4py import MPI
    from pykmc.config import Config
    from pykmc.dimer import _minimum_image
    from pykmc.engine.artn_search import ArtnSearchExtension
    from pykmc.engine.lammps import LammpsEngine
    from pykmc.eventsearch.eventsearch import SearchRequest
    from pykmc.eventsearch.strategies.artn import artn_options

    comm = MPI.COMM_WORLD
    ranks = comm.Get_size()
    if ranks not in (1, 2, 4):
        raise ValueError("rank gate requires exactly 1, 2 or 4 worker ranks")
    if args.output.exists():
        raise FileExistsError(args.output)
    config = Config.from_ini_file(str(checked(args.input, CONFIG)))
    config.lammps.pair_coeff = [f'* * "{checked(args.potential, POTENTIAL)}" Fe']
    config.eventsearch.style = "artn"
    with np.load(checked(args.cell, CELL), allow_pickle=False) as fixture:
        start = fixture["positions"]
        cell = fixture["cell"]
        if list(fixture["centres"]) != list(CENTRES):
            raise ValueError("Fig. 2 centres differ")
    pbc = np.ones(3, dtype=bool)
    types = np.full(len(start), "Fe")
    worker = LammpsEngine(config.lammps, comm)
    extension = ArtnSearchExtension(worker)
    worker.start()
    try:
        version = int(worker.lmp.version())
        if version != 20250722:
            raise ValueError(f"LAMMPS {version} differs from the Fig. 2 build")
        output = dict(
            kind=args.action,
            head=head,
            cell=CELL,
            potential=POTENTIAL,
            config=CONFIG,
            lammps=version,
            ranks=ranks,
            rows=[],
        )
        options = artn_options(config.partn)
        for c, seed in zip(CENTRES, SEEDS, strict=True):
            offsets = _minimum_image(start - start[c], cell, pbc)
            radius = 3.0 if args.action == "crop-smoke" else config.activevolume.rmov
            mobile = np.flatnonzero(np.linalg.norm(offsets, axis=1) <= radius)
            request = SearchRequest(c, start, cell, types, pbc, seed, mobile)
            if args.action == "crop-smoke":
                output["crop"] = crop_smoke(extension, request, options)
                break
            if len(mobile) != 440:
                raise ValueError(f"unexpected Fig. 2 mobile count {len(mobile)}")
            comm.Barrier()
            before = time.perf_counter()
            with count_force_calls() as measured:
                raw = extension.artn_search(**vars(request), options=options)
            elapsed = comm.allreduce(time.perf_counter() - before, op=MPI.MAX)
            counts = comm.allgather(measured["calls"])
            if any(count != counts[0] or count <= 0 for count in counts):
                raise ValueError(f"inconsistent underlying force-call counts {counts}")
            force_seconds = comm.allreduce(measured["seconds"], op=MPI.MAX)

            def record(
                raw: Any = raw,
                request: Any = request,
                counts: list = counts,
                elapsed: float = elapsed,
                force_seconds: float = force_seconds,
            ) -> None:
                row = event_record(raw, request, config)
                if ranks == 1:
                    row["raw_fingerprint"] = raw_fingerprint(
                        raw, request.cell.diagonal()
                    )
                row.update(
                    force_calls=counts[0], seconds=elapsed, force_seconds=force_seconds
                )
                output["rows"].append(row)

            root_action(comm, record)

        def save() -> None:
            encoded = json.dumps(output, allow_nan=False)
            with args.output.open("x") as f:
                f.write(encoded + "\n")

        root_action(comm, save)
    finally:
        extension._drop_dedicated_engine()
        worker.close()


def valid_row(row: dict, center: int, seed: int) -> None:
    """Reject missing, malformed or non-finite evidence for one paired search."""
    for key, expected in (("center", center), ("seed", seed)):
        if type(row[key]) is not int or row[key] != expected:
            raise ValueError(f"wrong Fig. 2 {key}")
    for key in ("force_calls", "callback_calls"):
        if type(row[key]) is not int or row[key] <= 0:
            raise ValueError(f"invalid {key}")
    if row["callback_calls"] > row["force_calls"]:
        raise ValueError("fewer engine force calls than callback calls")
    for key in ("seconds", "force_seconds"):
        if (
            type(row[key]) not in (int, float)
            or not math.isfinite(row[key])
            or row[key] <= 0
        ):
            raise ValueError(f"invalid {key}")
    if row["outcome"] != "event":
        if row["outcome"] not in {
            "empty_push",
            "eigenvalue_lost",
            "energy_limit",
            "non_finite_force",
            "max_force_calls",
            "EVENT_MINIMA_NOT_MATCH_POSITIONS",
        }:
            raise ValueError(f"unknown outcome {row['outcome']!r}")
        if any(
            k in row
            for k in (
                "event_id",
                "energy_barrier",
                "types",
                "saddle_positions",
                "move_atom_idx",
            )
        ):
            raise ValueError("non-event contains event fields")
        raw = row["raw"]
        scalars = {
            "e_start",
            "e_saddle",
            "e_min1",
            "e_min2",
            "delr1",
            "delr2",
            "curvature",
        }
        if (
            not isinstance(raw, dict)
            or raw.keys() != scalars
            or any(
                v is not None and (type(v) not in (int, float) or not math.isfinite(v))
                for v in raw.values()
            )
        ):
            raise ValueError("invalid partial search scalars")
        pairs = {
            "saddle_positions": "e_saddle",
            "min1_positions": "e_min1",
            "min2_positions": "e_min2",
        }
        if row["endpoints"].keys() != {
            k for k, v in pairs.items() if raw[v] is not None
        }:
            raise ValueError("partial endpoint energies and geometries differ")
        for values in row["endpoints"].values():
            xyz = np.asarray(values)
            if (
                xyz.shape != (1999, 3)
                or xyz.dtype.kind not in "fiu"
                or not np.isfinite(xyz).all()
            ):
                raise ValueError("invalid partial endpoint geometry")
        if row["outcome"] == "EVENT_MINIMA_NOT_MATCH_POSITIONS" and (
            len(row["endpoints"]) != 3 or any(v is None for v in raw.values())
        ):
            raise ValueError("disconnected outcome requires both endpoints")
        return
    xyz = np.asarray(row["saddle_positions"])
    if (
        type(row["energy_barrier"]) not in (int, float)
        or not math.isfinite(row["energy_barrier"])
        or not isinstance(row["event_id"], str)
        or not row["event_id"]
        or xyz.ndim != 2
        or xyz.shape[1] != 3
        or not len(xyz)
        or xyz.dtype.kind not in "fiu"
        or not np.isfinite(xyz).all()
        or row["types"] != ["Fe"] * len(xyz)
        or type(row["move_atom_idx"]) is not int
        or not 0 <= row["move_atom_idx"] < len(xyz)
    ):
        raise ValueError("invalid production event row")


def compare(arms: list[dict], config: Any, head: str) -> int:
    """Compare event identity and barrier; report timing and calls for every rank."""
    import pandas as pd
    from pykmc.event_table import ReferenceEventTable

    if len(arms) != 3:
        raise ValueError("require arms P=1,2,4")
    for p, arm in zip((1, 2, 4), arms, strict=True):
        expected = dict(
            kind="run",
            head=head,
            cell=CELL,
            potential=POTENTIAL,
            config=CONFIG,
            lammps=20250722,
            ranks=p,
        )
        if (
            any(type(arm[k]) is not type(v) or arm[k] != v for k, v in expected.items())
            or len(arm["rows"]) != 8
        ):
            raise ValueError(f"incomplete or incomparable P={p} evidence")
        for i, row in enumerate(arm["rows"]):
            valid_row(row, CENTRES[i], SEEDS[i])
            print(
                f"P={p} search={i} outcome={row['outcome']} force_calls={row['force_calls']} "
                f"callback_calls={row['callback_calls']} force_ms/call={1000 * row['force_seconds'] / row['force_calls']:.6g} "
                f"search_ms/call={1000 * row['seconds'] / row['force_calls']:.6g}"
            )
    different = False
    incomplete = False
    for i, base in enumerate(arms[0]["rows"]):
        table = ReferenceEventTable(config)
        if base["outcome"] == "event":
            table.table = pd.DataFrame(
                [dict(base, saddle_positions=np.asarray(base["saddle_positions"]))]
            )
        for arm in arms[1:]:
            row = arm["rows"][i]
            if base["outcome"] != row["outcome"]:
                different = True
                verdict = f"different outcomes {base['outcome']} / {row['outcome']}"
            elif row["outcome"] != "event":
                incomplete = True
                verdict = (
                    f"same non-event {row['outcome']}; event comparison unavailable"
                )
            else:
                candidate = pd.Series(
                    dict(row, saddle_positions=np.asarray(row["saddle_positions"]))
                )
                duplicate = not table.is_new_event(candidate)
                # Subtraction of two float eV values can round past a displayed meV.
                delta = abs(
                    Decimal(str(row["energy_barrier"]))
                    - Decimal(str(base["energy_barrier"]))
                )
                different |= not duplicate or delta > Decimal("0.001")
                verdict = f"duplicate={duplicate} abs_forward_barrier_delta_eV={delta}"
            print(f"P={arm['ranks']} search={i} {verdict}")
    status = "DIFFERENT" if different else "INCONCLUSIVE" if incomplete else "PASS"
    print(f"ARTN_RANK_COMPARISON_COMPLETE {status}", flush=True)
    return 1 if different else 2 if incomplete else 0


def main() -> int:
    """Run a live arm, or compare complete evidence with the pinned duplicate test."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("run", "crop-smoke"):
        arm = sub.add_parser(name)
        for key in ("cell", "potential", "input", "output"):
            arm.add_argument(f"--{key}", type=Path, required=True)
    check = sub.add_parser("compare")
    check.add_argument("--input", type=Path, required=True)
    check.add_argument("arms", nargs=3, type=Path, metavar="ARM")
    serial = sub.add_parser("compare-serial")
    serial.add_argument("--cell", type=Path, required=True)
    serial.add_argument("--before-head", required=True)
    serial.add_argument("--after-head", required=True)
    serial.add_argument("before", type=Path)
    serial.add_argument("after", type=Path)
    args = parser.parse_args()
    if args.action == "compare-serial":
        with np.load(checked(args.cell, CELL), allow_pickle=False) as cell:
            lengths = cell["cell"].diagonal()
        return compare_serial(
            json.loads(args.before.read_text()),
            json.loads(args.after.read_text()),
            args.before_head,
            args.after_head,
            lengths,
        )
    if args.action == "compare":
        import mpi4py

        mpi4py.rc.initialize = False
        mpi4py.rc.finalize = False
        sys.modules["lammps"] = sys.modules["pypARTn"] = None
    head = checkout()
    if args.action == "compare":
        from pykmc.config import Config

        return compare(
            [json.loads(p.read_text()) for p in args.arms],
            Config.from_ini_file(str(checked(args.input, CONFIG))),
            head,
        )
    run(args, head)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (KeyError, TypeError, ValueError, OSError) as error:
        print(f"Incomplete rank gate evidence: {error}", file=sys.stderr)
        sys.exit(2)
