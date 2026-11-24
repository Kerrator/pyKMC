"""Step replay of a recorded pARTn v1.1.0 search through pyKMC's ARTn core.

Issue 02 of the Python ARTn plan recorded the eight Fig. 2 searches at every
force evaluation (``searchJ.npz``, NumPy only; layout in that issue's
``NOTE.md``). :func:`replay` runs :func:`pykmc.artn.artn_search` from the
recorded start with the recorded initial push and first Lanczos vector. Its
force callback returns the recorded forces of pARTn's next step and compares
the positions the core asked for with the positions pARTn evaluated there.
The core carries its own state; nothing is reset from the recording.

pARTn evaluates more often than it steps: the setup evaluation, each blank
call A after a FIRE restart (the repeats, whose forces fix artn discards) and
the evaluation after the stop. The core evaluates once per step, so the
replay skips those and counts the repeats. Steps are pARTn's ``partn_step``
indices, which count calls to ``artn()``; the core's k-th callback call is
pARTn's step k.

At each step the replay compares, within ``tolerance``:

- the positions, as the largest minimum-image component of the difference, Å;
- the block: the core's move against pARTn's ``disp_code`` that led there;
- after a FIRE step (PERP, RELX), FIRE's time step and mixing parameter
  (relative 1e-12) and the velocities (``tolerance / dt0``, Å/ps).

It stops at the first step that differs, or after ``last_step``: by default
the first step a Lanczos move led to, which closes the initial-push phase.

Run ``python -m tests.artn_replay DIR`` to print the per-step differences of
every ``searchJ.npz`` in ``DIR``.
"""

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import numpy as np

import pykmc.artn as artn
from pykmc.dimer import _minimum_image
from pykmc.eventsearch.strategies.artn import basin_perp_limit

# Environment variable naming the folder of the full recordings.
RECORDINGS_ENV = "PYKMC_ARTN_RECORDINGS"
# Largest position difference accepted at a step, Å (open question 2).
TOLERANCE = 1.0e-10
# Fe mass of the recordings' potential, FeP_mm.eam.fs (element line), amu.
FE_MASS = 55.85
# pARTn's bohr in Å (artn:src/h_artn_units.f90:71); the pushes are recorded in bohr.
B2A = 0.529177210903
# pARTn's disp_code values (move_mode), as the recordings store them.
DISP_CODES = {
    1: "VOID",
    2: "INIT",
    3: "PERP",
    4: "EIGN",
    5: "LANC",
    6: "RELX",
    7: "OVER",
    8: "SMTH",
    9: "RSET",
}
# The core's move names, as pARTn's blocks.
CORE_BLOCKS = {
    "start": ("start",),
    "init": ("INIT",),
    "perp": ("PERP",),
    "lanczos": ("LANC",),
    "eigen": ("EIGN", "SMTH"),
    "over": ("OVER",),
    "relax": ("RELX",),
}
FIRE_BLOCKS = ("PERP", "RELX")


@dataclass
class StepDiff:
    """What the replay compared at one pARTn step."""

    step: int
    evaluation: int
    block: str
    move: Optional[str]
    position: float
    dt: Optional[float] = None
    alpha: Optional[float] = None
    velocity: Optional[float] = None
    problem: Optional[str] = None


@dataclass
class ReplayReport:
    """The outcome of :func:`replay` on one recording."""

    search: int
    tolerance: float
    last_step: int
    steps: list = field(default_factory=list)
    repeats_skipped: int = 0
    repeats_total: int = 0
    other_skipped: int = 0
    end: str = ""

    @property
    def first_mismatch(self) -> Optional[StepDiff]:
        """The first step that differs, if any."""
        return next((s for s in self.steps if s.problem), None)

    @property
    def passed(self) -> bool:
        """Every step up to ``last_step`` matched."""
        return self.first_mismatch is None and len(self.steps) == self.last_step + 1

    def describe(self) -> str:
        """One line: the first differing step, or how far the replay matched."""
        bad = self.first_mismatch
        if bad is not None:
            return (
                f"search {self.search}: first differing step {bad.step} "
                f"(evaluation {bad.evaluation}, block {bad.block}, core move {bad.move}): "
                f"{bad.problem}"
            )
        if not self.passed:
            return (
                f"search {self.search}: the core stopped after {len(self.steps)} of "
                f"{self.last_step + 1} steps ({self.end})"
            )
        return (
            f"search {self.search}: steps 0-{self.last_step} match, largest position "
            f"difference {max(s.position for s in self.steps):.2e} A; "
            f"{self.repeats_skipped} repeats skipped"
        )


class _Stop(Exception):
    """Ends the core's search from inside the callback."""


def load_recording(path: Path) -> dict:
    """Read the arrays :func:`replay` needs from one ``searchJ.npz``."""
    z = np.load(path, allow_pickle=False)
    meta = json.loads(str(z["meta_json"]))
    cols = meta["scalars"]
    s_in, s_out = z["scalars_in"], z["scalars_out"]
    sets = {
        name: value for kind, name, value in meta["result"]["sets"] if kind == "set"
    }
    config = meta["conditions"]["partn_config"]
    codes = s_out[:, cols.index("partn_disp_code")]
    params = dict(
        push_mode=sets["push_mode"],
        push_step_size=sets["push_step_size"],
        ninit=sets["ninit"],
        lanczos_min_size=sets["lanczos_min_size"],
        lanczos_max_size=sets["lanczos_max_size"],
        lanczos_disp=sets["lanczos_disp"],
        lanczos_eval_conv_thr=sets["lanczos_eval_conv_thr"],
        eigval_thr=sets["eigval_thr"],
        eigen_step_size=sets["eigen_step_size"],
        nperp=basin_perp_limit(
            SimpleNamespace(
                nperp=sets.get("nperp"), nperp_limitation=sets.get("nperp_limitation")
            )
        ),
        forc_thr=sets["forc_thr"],
        push_over=sets["push_over"],
        convergence_property=config["convergence_property"],
        dmax=config["dmax"],
    )
    if "push_dist_thr" in sets:
        params["push_dist_thr"] = sets["push_dist_thr"]
    n_atoms = z["movable"].shape[0]
    return dict(
        search=meta["search"],
        x=z["x"],
        f=z["f"],
        v_in=z["v_in"],
        energy=s_in[:, cols.index("energy")],
        fire_dt=s_in[:, cols.index("fire_dt")],
        fire_alpha=s_in[:, cols.index("fire_alpha")],
        block_out=[DISP_CODES.get(int(c)) if np.isfinite(c) else None for c in codes],
        partn_step=z["partn_step"],
        repeat=z["repeat"],
        kind=z["kind"],
        initial_push=z["draw_initial_push"] * B2A,
        first_lanczos_vector=z["draw_first_lanczos_vector"],
        movable=z["movable"],
        cell=z["cell"],
        pbc=z["pbc"],
        central_atom=sets["push_ids"][0] - 1,
        masses=np.full(n_atoms, FE_MASS),
        params=params,
    )


def replay(
    rec: dict, tolerance: float = TOLERANCE, last_step: Optional[int] = None
) -> ReplayReport:
    """Run the core on a recording's forces and compare it step by step."""
    steps = np.flatnonzero(rec["partn_step"] >= 0)
    if not np.array_equal(rec["partn_step"][steps], np.arange(len(steps))):
        raise ValueError("the recording's pARTn steps are not 0, 1, 2, ... in order")
    led_by = ["start"] + [rec["block_out"][e] for e in steps[:-1]]
    if last_step is None:
        last_step = led_by.index("LANC") if "LANC" in led_by else len(steps) - 1
    report = ReplayReport(
        rec["search"], tolerance, last_step, repeats_total=int(rec["repeat"].sum())
    )
    seen: dict = {}

    def observe(move: str, fire_state: Optional[tuple]) -> None:
        seen["move"], seen["fire"] = move, fire_state

    def forces(x: np.ndarray) -> tuple[float, np.ndarray]:
        k = len(report.steps)
        if k > last_step:
            raise _Stop("window")
        e = int(steps[k])
        diff = _minimum_image(x - rec["x"][e], rec["cell"], rec["pbc"])
        step = StepDiff(k, e, led_by[k], seen.get("move"), float(np.max(np.abs(diff))))
        problems = []
        if step.block not in CORE_BLOCKS.get(step.move, ()):
            problems.append(f"block {step.block} against core move {step.move}")
        if step.position > tolerance:
            problems.append(
                f"positions differ by {step.position:.3e} A (tolerance {tolerance:.1e} A)"
            )
        fire_state = seen.get("fire")
        if step.block in FIRE_BLOCKS and fire_state is not None:
            dt, alpha, v = fire_state
            step.dt = abs(dt - rec["fire_dt"][e])
            step.alpha = abs(alpha - rec["fire_alpha"][e])
            step.velocity = float(np.max(np.abs(v - rec["v_in"][e])))
            if step.dt > 1e-12 * rec["fire_dt"][e]:
                problems.append(f"FIRE dt {dt!r} against {rec['fire_dt'][e]!r}")
            if step.alpha > 1e-12 * max(abs(rec["fire_alpha"][e]), 1.0):
                problems.append(
                    f"FIRE alpha {alpha!r} against {rec['fire_alpha'][e]!r}"
                )
            if step.velocity > tolerance / artn._DT0:
                problems.append(f"velocities differ by {step.velocity:.3e} A/ps")
        step.problem = "; ".join(problems) or None
        report.steps.append(step)
        if step.problem:
            raise _Stop("mismatch")
        return float(rec["energy"][e]), rec["f"][e]

    p = rec["params"]
    try:
        result = artn.artn_search(
            rec["x"][steps[0]],
            rec["cell"],
            rec["pbc"],
            rec["movable"],
            rec["central_atom"],
            0,
            forces,
            masses=rec["masses"],
            initial_push=rec["initial_push"],
            first_lanczos_vector=rec["first_lanczos_vector"],
            observe=observe,
            **p,
        )
        report.end = result.reason or "search finished"
    except _Stop as stop:
        report.end = str(stop)
    reached = int(steps[len(report.steps) - 1]) if report.steps else -1
    before = np.arange(len(rec["repeat"])) < reached
    report.repeats_skipped = int(np.sum(rec["repeat"] & before))
    report.other_skipped = int(
        np.sum(~rec["repeat"] & (rec["partn_step"] < 0) & before)
    )
    return report


def _main(folder: str) -> None:
    for path in sorted(Path(folder).glob("search*.npz")):
        report = replay(load_recording(path))
        print(report.describe())
        for s in report.steps:
            fire = (
                f"  dt {s.dt:.1e}  alpha {s.alpha:.1e}  v {s.velocity:.2e} A/ps"
                if s.dt is not None
                else ""
            )
            print(
                f"  step {s.step:3d}  eval {s.evaluation:4d}  {s.block:5s}  dx {s.position:.2e} A{fire}"
            )
        print(
            f"  repeats skipped {report.repeats_skipped} (in the recording {report.repeats_total})"
        )


if __name__ == "__main__":
    _main(sys.argv[1])
