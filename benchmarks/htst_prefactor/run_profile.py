"""Profile the two HTST Hessian modes (eskm vs Python-FD) and write a CSV.

Runs ``pykmc.htst.profiling.time_event`` for both modes across a free-radius
sweep on a chosen system (``ni100`` = the committed 130-atom surface-hop fixture,
1 event; ``ni4000`` = the 2 stored neighbour-cluster events of the 4000-atom Ni
reference table), ``--repeats`` times each, and writes one CSV row per
(mode, system, event, free_radius, hessian, repeat). Use ``compare.py`` to turn
the CSV into a markdown report + recommendation. See README.md for invocations,
potentials, and notes.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from pykmc.htst import profiling  # noqa: E402

_DATA = _REPO_ROOT / "tests" / "data"
_NIALH = _REPO_ROOT / "basin_testing" / "NiAlH_jea.eam"
_NI_V6 = _DATA / "Ni_v6_2.0_LKBeland2016.eam"
_NI100_NPZ = _DATA / "htst_ni100_surface_hop.npz"
_NI4000_PICKLE = _DATA / "reference_table_Ni_fcc_4000at_monovacancy+sia.pickle"

_PBC = np.array([True, True, True])
_FD_STEP = 0.01

_FIELDS = [
    "mode", "system", "event", "free_radius", "n_free", "hessian_idx",
    "t_hessian", "t_forces_total", "n_calls", "t_forces_mean",
    "t_set_run0", "t_dynmat_cmd", "t_file_read", "t_eigh",
    "nu0_hz", "nu0_thz", "round_trips", "repeat",
]


def _default_potential(system: str) -> Path:
    """Pick the default potential: NiAlH for ni100 if present, else tracked Ni_v6."""
    if system == "ni100" and _NIALH.exists():
        return _NIALH
    return _NI_V6


def _load_events(system: str) -> list[dict[str, object]]:
    """Return a list of event dicts (min1/saddle/min2/move/n) for the system."""
    events: list[dict[str, object]] = []
    if system == "ni100":
        data = np.load(_NI100_NPZ)
        events.append(
            {
                "min1": data["initial_positions"],
                "saddle": data["saddle_positions"],
                "min2": data["final_positions"],
                "move": int(data["move_atom_idx"]),
                "n": int(data["n_atoms"]),
            }
        )
    elif system == "ni4000":
        df = pd.read_pickle(_NI4000_PICKLE)
        for _, row in df.iterrows():
            init = np.asarray(row["initial_positions"], dtype=float)
            events.append(
                {
                    "min1": init,
                    "saddle": np.asarray(row["saddle_positions"], dtype=float),
                    "min2": np.asarray(row["final_positions"], dtype=float),
                    "move": int(row["move_atom_idx"]),
                    "n": int(init.shape[0]),
                }
            )
    else:
        raise ValueError(f"unknown system {system!r} (expected ni100 or ni4000)")
    return events


def main(argv: "list[str] | None" = None) -> None:
    """Run the profiling sweep and write the CSV."""
    p = argparse.ArgumentParser(prog="run_profile", description=__doc__)
    p.add_argument("--system", choices=["ni100", "ni4000"], required=True)
    p.add_argument("--modes", default="fd,eskm", help="Comma list: fd,eskm.")
    p.add_argument("--radii", default="3.0,4.0,5.0,6.0,7.5", help="Comma list of free radii (A).")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--potential", default=None, help="LAMMPS potential file (defaults per system).")
    p.add_argument("--pair-style", default="eam/alloy")
    p.add_argument("--out", required=True, help="Output CSV path.")
    args = p.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    radii = [float(r) for r in args.radii.split(",") if r.strip()]
    potential = Path(args.potential) if args.potential else _default_potential(args.system)
    if not potential.exists():
        raise SystemExit(f"potential not found: {potential}")

    events = _load_events(args.system)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    for ev_idx, ev in enumerate(events):
        engine, cell = profiling.build_serial_engine(
            ev["saddle"], potential=str(potential), pair_style=args.pair_style, element="Ni"
        )
        run_modes = list(modes)
        if "eskm" in run_modes and not profiling.phonon_available(engine):
            print(
                "[profile] PHONON package absent: dynamical_matrix unavailable — "
                "skipping eskm, running FD only."
            )
            run_modes = [m for m in run_modes if m != "eskm"]
        types = ["Ni"] * int(ev["n"])
        for mode in run_modes:
            for radius in radii:
                for rep in range(args.repeats):
                    try:
                        rows = profiling.time_event(
                            engine, mode=mode,
                            min1=ev["min1"], saddle=ev["saddle"], min2=ev["min2"],
                            types=types, central_index=int(ev["move"]),
                            free_radius=radius, fd_step=_FD_STEP, cell=cell, pbc=_PBC,
                            system=args.system, event=ev_idx,
                        )
                    except profiling.PhononUnavailable as exc:
                        print(f"[profile] {exc} — dropping eskm.")
                        run_modes = [m for m in run_modes if m != "eskm"]
                        break
                    for row in rows:
                        d = asdict(row)
                        d["repeat"] = rep
                        all_rows.append(d)
                print(f"[profile] {args.system} ev{ev_idx} mode={mode} r={radius} done")

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[profile] wrote {len(all_rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
