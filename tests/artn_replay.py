"""Step replay of a recorded pARTn v1.1.0 search through pyKMC's ARTn core.

Issue 02 of the Python ARTn plan recorded the eight Fig. 2 searches at every
force evaluation (``searchJ.npz``, NumPy only; layout in that issue's
``NOTE.md``). :func:`replay` runs :func:`pykmc.artn.artn_search` from the
recorded start with the recorded initial push, first Lanczos vector and the
push each new chance built. Its force callback returns the recorded forces of
pARTn's next step, with the positions LAMMPS held there (pARTn's alignment
test reads them, wrapped into the cell), and compares the positions the core
asked for with the positions pARTn evaluated there. The core carries its own
state; nothing is reset from the recording.

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
  (relative 1e-12) and the velocities (``tolerance / dt0``, Å/ps);
- after a Lanczos move (a probe or the move back), the lowest eigenvalue and
  the eigenvector pARTn held after the call that made the move: the
  eigenvalue's difference in eV/Å², and the eigenvector's largest component
  difference up to sign. Both are reported at every probe. They are judged at
  the end of a block, the state the move back carries, which is the only one
  later moves use, when pARTn's next step pushes along the eigenvector: the
  eigenvector within ``tolerance / eigen_step_size``, and the eigenvalue
  within that times max(|eigenvalue|, 0.48587 eV/Å²), so that the push stays
  within ``tolerance`` of pARTn's. In the basin the eigenvector only seeds
  the next block and the eigenvalue only sets basin or not, so a difference
  there shows in the steps that follow (ruling of 2026-11-24).

It stops at the first step that differs, or after ``last_step``: by default
the last step of the first Lanczos block, the move back to where the block
started (or the last step the recording holds before it). Past the window the
callback returns a non-finite energy, so the core ends its search and returns
what it found.

:func:`check_climb` replays to the step where pARTn found its saddle, or
failed, and compares the end: the core asks for the push over the saddle
next, with the same saddle, energy and eigenvalue, or ends with pyKMC's reason
for pARTn's failure.

:func:`replay_block` replays one Lanczos block from its recorded starting
state: the core starts at the block's first position with ``ninit = 0`` and
the recorded starting vector. :func:`lanczos_blocks` lists a recording's
blocks.

Run ``python -m tests.artn_replay DIR`` to print, for every ``searchJ.npz``
in ``DIR``, the first Lanczos block step by step, the climb to pARTn's saddle
or failure, and every block replayed alone.
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

import pykmc.artn as artn
from pykmc.dimer import DimerResult, _minimum_image

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
    eigval: Optional[float] = None
    eigval_scaled: Optional[float] = None
    mode: Optional[float] = None
    mode_sign: Optional[int] = None
    block_end: bool = False
    problem: Optional[str] = None


@dataclass
class ReplayReport:
    """The outcome of :func:`replay` on one recording."""

    search: int
    tolerance: float
    last_step: int
    first_step: int = 0
    steps: list = field(default_factory=list)
    repeats_skipped: int = 0
    repeats_total: int = 0
    other_skipped: int = 0
    end: str = ""
    # The core's Lanczos moves seen (callback calls asked for, the last one
    # possibly past the window), and the differences of the Lanczos state
    # after the last of them: (eigenvalue, scaled eigenvalue, mode, sign).
    lanczos_moves: int = 0
    lanczos_end: Optional[tuple] = None
    # The move the core asked for past the window (None if it asked for
    # none), and the result it returned.
    next_move: Optional[str] = None
    result: Optional[DimerResult] = None

    @property
    def first_mismatch(self) -> Optional[StepDiff]:
        """The first step that differs, if any."""
        return next((s for s in self.steps if s.problem), None)

    @property
    def passed(self) -> bool:
        """Every step up to ``last_step`` matched."""
        return (
            self.first_mismatch is None
            and len(self.steps) == self.last_step - self.first_step + 1
        )

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
                f"{self.last_step - self.first_step + 1} steps ({self.end})"
            )
        return (
            f"search {self.search}: steps {self.first_step}-{self.last_step} match, "
            f"largest position difference {max(s.position for s in self.steps):.2e} A"
            f"{_largest(self.steps, 'eigval', ', eigenvalue {:.2e} eV/A^2')}"
            f"{_largest(self.steps, 'mode', ', mode {:.2e}')}; "
            f"{self.repeats_skipped} repeats skipped"
        )


def _largest(steps: list, name: str, text: str) -> str:
    values = [getattr(s, name) for s in steps if getattr(s, name) is not None]
    return text.format(max(values)) if values else ""


class _Stop(Exception):
    """Ends the core's search from inside the callback."""


class _PastWindow(Exception):
    """The core asked for a step past the window."""


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
        nsmooth=sets["nsmooth"],
        alpha_mix_cr=sets["alpha_mix_cr"],
        nnewchance=sets["nnewchance"],
        nperp=sets.get("nperp"),
        nperp_limitation=sets.get("nperp_limitation"),
        forc_thr=sets["forc_thr"],
        push_over=sets["push_over"],
        convergence_property=config["convergence_property"],
        nevalf_max=config["nevalf_max"],
        dmax=config["dmax"],
    )
    if "push_dist_thr" in sets:
        params["push_dist_thr"] = sets["push_dist_thr"]
    n_atoms = z["movable"].shape[0]
    has_eigenvec = "eigenvec" in z.files
    chances = z["draw_new_chance_push"] if "draw_new_chance_push" in z.files else []
    return dict(
        search=meta["search"],
        x=z["x"],
        f=z["f"],
        v_in=z["v_in"],
        energy=s_in[:, cols.index("energy")],
        fire_dt=s_in[:, cols.index("fire_dt")],
        fire_alpha=s_in[:, cols.index("fire_alpha")],
        block_out=[DISP_CODES.get(int(c)) if np.isfinite(c) else None for c in codes],
        # pARTn's state after each call: eigenvalue (eV/Å², inf while unset),
        # run flags, and the eigenvector versions with their index.
        eigval=s_out[:, cols.index("partn_eigval_step")],
        in_lanczos=_flag(s_out[:, cols.index("partn_flags")], meta, "llanczos"),
        eigenvec=z["eigenvec"] if has_eigenvec else None,
        eigenvec_index=z["eigenvec_index"] if has_eigenvec else None,
        partn_step=z["partn_step"],
        repeat=z["repeat"],
        kind=z["kind"],
        initial_push=z["draw_initial_push"] * B2A,
        first_lanczos_vector=z["draw_first_lanczos_vector"],
        new_chance_pushes=[push * B2A for push in chances],
        # pARTn's own result (pyKMC's extract): saddle energy (eV), eigenvalue
        # (eV/Å²), the step it was found at, and the error code and message.
        partn=meta["result"]["partn"],
        tau_sad=z["partn_tau_sad"] if "partn_tau_sad" in z.files else None,
        movable=z["movable"],
        cell=z["cell"],
        pbc=z["pbc"],
        central_atom=sets["push_ids"][0] - 1,
        masses=np.full(n_atoms, FE_MASS),
        params=params,
    )


def _flag(flags: np.ndarray, meta: dict, name: str) -> np.ndarray:
    bit = meta["partn_flag_bits"].index(name)
    ok = np.isfinite(flags)
    return ok & ((np.where(ok, flags, 0).astype(np.int64) >> bit) & 1).astype(bool)


def _steps_and_blocks(rec: dict) -> tuple[np.ndarray, list]:
    """Return the evaluation of each pARTn step and the block that led to it."""
    steps = np.flatnonzero(rec["partn_step"] >= 0)
    if not np.array_equal(rec["partn_step"][steps], np.arange(len(steps))):
        raise ValueError("the recording's pARTn steps are not 0, 1, 2, ... in order")
    return steps, ["start"] + [rec["block_out"][e] for e in steps[:-1]]


@dataclass
class LanczosBlock:
    """One recorded Lanczos block, in pARTn steps."""

    start: int  # the call at the block's first position, which writes probe 1
    probes: int  # probes evaluated
    returns: bool  # the move back to ``start``'s position was evaluated
    evaluations: int  # every evaluation of the probes and the move back
    repeats: int  # of them, flagged repeats

    @property
    def end(self) -> int:
        """The last step a Lanczos move of this block led to."""
        return self.start + self.probes + int(self.returns)


def lanczos_blocks(rec: dict) -> list:
    """List the recording's Lanczos blocks, in order.

    A block's probes run to the call that ends it (pARTn's ``llanczos`` off
    after the call); a step after that which a Lanczos move led to is the
    move back. Blocks the recording cuts off are left out.
    """
    steps, led_by = _steps_and_blocks(rec)
    blocks, i = [], 1
    while i < len(steps):
        if led_by[i] != "LANC":
            i += 1
            continue
        start, j = i - 1, i
        while j < len(steps) and rec["in_lanczos"][steps[j]]:
            j += 1
        if j == len(steps):
            break  # cut off before the block ends
        returns = j + 1 < len(steps) and led_by[j + 1] == "LANC"
        last = steps[j + int(returns)]
        span = slice(steps[start] + 1, last + 1)
        blocks.append(
            LanczosBlock(
                start,
                j - start,
                returns,
                last - steps[start],
                int(rec["repeat"][span].sum()),
            )
        )
        i = j + 1 + int(returns)
    return blocks


def starting_vector(rec: dict, block: LanczosBlock) -> np.ndarray:
    """Return the vector a block's first basis vector is made from (v_in).

    The first block of a search starts from the random draw; later blocks from
    the eigenvector in force before the block's first call.
    """
    steps, led_by = _steps_and_blocks(rec)
    if block.start == led_by.index("LANC") - 1:
        return rec["first_lanczos_vector"]
    return rec["eigenvec"][rec["eigenvec_index"][steps[block.start] - 1]]


def first_lanczos_block_end(rec: dict) -> int:
    """Return the first Lanczos block's last step, or the last step recorded."""
    steps, led_by = _steps_and_blocks(rec)
    blocks = lanczos_blocks(rec)
    if blocks:
        return blocks[0].end
    if "LANC" not in led_by:
        return len(steps) - 1
    i = led_by.index("LANC")
    while i + 1 < len(steps) and led_by[i + 1] == "LANC":
        i += 1
    return i


def _lanczos_differences(state: tuple, rec: dict, e: int) -> tuple:
    """Compare the core's Lanczos state with pARTn's after evaluation ``e``.

    Returns the eigenvalue difference (eV/Å²; None while the eigenvalue is
    unset), that difference over max(|eigenvalue|, 0.48587 eV/Å²), the
    eigenvector's largest component difference up to sign (None without
    recorded eigenvectors) and the sign.
    """
    lowest, eigvec = state
    recorded = rec["eigval"][e]
    eigval = scaled = None
    if lowest != sys.float_info.max and math.isfinite(recorded):
        eigval = abs(lowest - recorded)
        scaled = eigval / max(abs(recorded), artn._CURVATURE_FLOOR)
    if rec["eigenvec"] is None:
        return eigval, scaled, None, None
    mode = rec["eigenvec"][rec["eigenvec_index"][e]]
    same, flipped = np.max(np.abs(eigvec - mode)), np.max(np.abs(eigvec + mode))
    return eigval, scaled, float(min(same, flipped)), 1 if same <= flipped else -1


def lanczos_problems(differences: Optional[tuple], rec: dict, tolerance: float) -> list:
    """List what fails at a block's end: a missing state, eigenvalue or mode."""
    if differences is None:
        return ["no Lanczos state at the end of the block"]
    _, scaled, mode, _ = differences
    bound = tolerance / rec["params"]["eigen_step_size"]
    problems = []
    if scaled is None or scaled > bound:
        problems.append(f"eigenvalue differs by {scaled} of max(|eigenvalue|, floor)")
    if mode is None or mode > bound:
        problems.append(f"eigenvector differs by {mode} (up to sign)")
    return problems


def replay(
    rec: dict,
    tolerance: float = TOLERANCE,
    last_step: Optional[int] = None,
    first_step: int = 0,
    first_lanczos_vector: Optional[np.ndarray] = None,
    keep_going: bool = False,
) -> ReplayReport:
    """Run the core on a recording's forces and compare it step by step.

    From ``first_step`` > 0 the core starts at that step's positions with
    ``ninit = 0``, so its first move is Lanczos from ``first_lanczos_vector``;
    use :func:`replay_block`. With ``keep_going`` the replay goes on past a
    step that differs only in its numbers, still feeding pARTn's forces, and
    stops at the first step the core reaches by another block; the report
    still fails at the first step that differs.
    """
    steps, led_by = _steps_and_blocks(rec)
    if last_step is None:
        last_step = first_lanczos_block_end(rec)
    report = ReplayReport(
        rec["search"],
        tolerance,
        last_step,
        first_step,
        repeats_total=int(rec["repeat"].sum()),
    )
    seen: dict = {}

    def observe(move: str, state: Optional[tuple]) -> None:
        seen["move"], seen["state"], seen["lanczos"] = move, state, None
        if move == "lanczos":
            report.lanczos_moves += 1
            i = first_step + len(report.steps)  # the step this move leads to
            if 0 < i <= len(steps):
                seen["lanczos"] = _lanczos_differences(state, rec, int(steps[i - 1]))
                report.lanczos_end = seen["lanczos"]

    def forces(x: np.ndarray) -> tuple:
        k = first_step + len(report.steps)
        if k > last_step:
            # Past the window: a non-finite energy ends the core's search, which
            # then returns what it found (the saddle, if it had found one).
            report.next_move = seen.get("move")
            raise _Stop("window") if k >= len(steps) else _PastWindow()
        e = int(steps[k])
        diff = _minimum_image(x - rec["x"][e], rec["cell"], rec["pbc"])
        block = "start" if k == first_step else led_by[k]
        step = StepDiff(k, e, block, seen.get("move"), float(np.max(np.abs(diff))))
        problems = []
        if step.block not in CORE_BLOCKS.get(step.move, ()):
            problems.append(f"block {step.block} against core move {step.move}")
        if step.position > tolerance:
            problems.append(
                f"positions differ by {step.position:.3e} A (tolerance {tolerance:.1e} A)"
            )
        state = seen.get("state")
        if step.block in FIRE_BLOCKS and state is not None:
            dt, alpha, v = state
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
        if step.block == "LANC" and seen.get("lanczos") is not None:
            step.eigval, step.eigval_scaled, step.mode, step.mode_sign = seen["lanczos"]
            # The move back: pARTn's call before it ended the block.
            step.block_end = not rec["in_lanczos"][steps[k - 1]]
            pushed = rec["block_out"][e] in CORE_BLOCKS["eigen"]  # pARTn's next move
            if step.block_end and pushed:
                problems += lanczos_problems(seen["lanczos"], rec, tolerance)
        step.problem = "; ".join(problems) or None
        report.steps.append(step)
        if step.problem and not (
            keep_going and step.block in CORE_BLOCKS.get(step.move, ())
        ):
            raise _Stop("mismatch")
        # The positions as LAMMPS held them, for the alignment test.
        return float(rec["energy"][e]), rec["f"][e], rec["x"][e]

    p = dict(rec["params"])
    vector = rec["first_lanczos_vector"]
    if first_step > 0:
        p["ninit"] = 0
        vector = first_lanczos_vector

    def callback(x: np.ndarray) -> tuple:
        try:
            return forces(x)
        except _PastWindow:
            return math.nan, np.zeros_like(x)

    try:
        result = artn.artn_search(
            rec["x"][steps[first_step]],
            rec["cell"],
            rec["pbc"],
            rec["movable"],
            rec["central_atom"],
            0,
            callback,
            masses=rec["masses"],
            initial_push=rec["initial_push"],
            first_lanczos_vector=vector,
            new_chance_pushes=rec.get("new_chance_pushes"),
            observe=observe,
            **p,
        )
        report.result = result
        past = report.next_move is not None and result.reason == "non_finite_force"
        report.end = "window" if past else (result.reason or "search finished")
    except _Stop as stop:
        report.end = str(stop)
    reached = int(steps[first_step + len(report.steps) - 1]) if report.steps else -1
    evaluation = np.arange(len(rec["repeat"]))
    before = (evaluation > (steps[first_step] if first_step else -1)) & (
        evaluation < reached
    )
    report.repeats_skipped = int(np.sum(rec["repeat"] & before))
    report.other_skipped = int(
        np.sum(~rec["repeat"] & (rec["partn_step"] < 0) & before)
    )
    return report


def replay_block(
    rec: dict, block: LanczosBlock, tolerance: float = TOLERANCE
) -> ReplayReport:
    """Replay one Lanczos block from its recorded starting state.

    The core starts at the block's first position, with that position's
    recorded force and the block's starting vector, and runs to the block's
    last step. The eigenvalue pARTn carries into the block's first convergence
    test is not set: it cannot stop a block at its first probe while
    ``lanczos_min_size`` > 1, and the first probe's eigenvalue is not compared.
    """
    return replay(
        rec,
        tolerance,
        last_step=block.end,
        first_step=block.start,
        first_lanczos_vector=starting_vector(rec, block),
    )


@dataclass
class BlockCheck:
    """One Lanczos block replayed alone, against its recording."""

    block: LanczosBlock
    report: ReplayReport
    rec: dict = field(repr=False)

    @property
    def core_evaluations(self) -> int:
        """Callback calls the core made in the block (probes and the move back)."""
        # The core's last Lanczos move is the move back; past a recording that
        # ends at the last probe it is asked for but not evaluated.
        return self.report.lanczos_moves - (0 if self.block.returns else 1)

    @property
    def recorded_evaluations(self) -> int:
        """The block's recorded evaluations less the flagged repeats."""
        return self.block.evaluations - self.block.repeats

    def end(self, i: int) -> Optional[float]:
        """Item ``i`` of the differences after the last probe (see ``lanczos_end``)."""
        return self.report.lanczos_end[i] if self.report.lanczos_end else None

    def problems(self) -> list:
        """Why the block does not match; empty if it does."""
        found = [self.report.describe()] if not self.report.passed else []
        if self.core_evaluations != self.recorded_evaluations:
            found.append(
                f"{self.core_evaluations} evaluations against {self.recorded_evaluations}"
            )
        return found + lanczos_problems(
            self.report.lanczos_end, self.rec, self.report.tolerance
        )


def check_blocks(rec: dict, tolerance: float = TOLERANCE) -> list:
    """Replay every Lanczos block of a recording alone."""
    return [
        BlockCheck(b, replay_block(rec, b, tolerance), rec) for b in lanczos_blocks(rec)
    ]


# pARTn's failure messages (err_set) and pyKMC's failure reasons for them.
FAILURE_REASONS = {
    "EIGENVALUE LOST": "eigenvalue_lost",
    "NUMBER OF STEPS EXCEEDS THE LIMIT": "max_force_calls",
    "ENERGY EXCEEDS THE LIMIT": "energy_limit",
}


def climb_end(rec: dict) -> int:
    """Return the step pARTn found the saddle at, or the step it failed at.

    A recording cut before that step does not hold it (see :func:`holds_climb`).
    """
    if rec["partn"]["has_sad"]:
        return int(rec["partn"]["nevalf_sad"])
    return int(rec["partn"]["nevalf"])


def holds_climb(rec: dict) -> bool:
    """Return True if the recording runs to pARTn's saddle or failure."""
    steps, _ = _steps_and_blocks(rec)
    return climb_end(rec) < len(steps)


def partn_failure_reason(rec: dict) -> Optional[str]:
    """Return pyKMC's reason for pARTn's failure, None if pARTn found a saddle."""
    if rec["partn"]["has_sad"]:
        return None
    message = rec["partn"]["error"][1]
    return next(
        (reason for text, reason in FAILURE_REASONS.items() if text in message),
        f"unmapped: {message}",
    )


@dataclass
class ClimbCheck:
    """A whole replay to pARTn's saddle or failure, and the end against pARTn's."""

    rec: dict = field(repr=False)
    report: ReplayReport

    @property
    def saddle(self) -> bool:
        """Return True if pARTn found a saddle."""
        return bool(self.rec["partn"]["has_sad"])

    @property
    def energy(self) -> Optional[float]:
        """|core saddle energy - pARTn's|, eV."""
        r = self.report.result
        if not self.saddle or r is None or r.e_saddle is None:
            return None
        return abs(r.e_saddle - self.rec["partn"]["etot_sad"])

    @property
    def eigval(self) -> Optional[float]:
        """|core curvature - pARTn's saddle eigenvalue|, eV/Å²."""
        r = self.report.result
        if not self.saddle or r is None or r.curvature is None:
            return None
        return abs(r.curvature - self.rec["partn"]["eigval_sad"])

    @property
    def saddle_position(self) -> Optional[float]:
        """Largest minimum-image component of core saddle - pARTn's tau_sad, Å."""
        r = self.report.result
        if not self.saddle or r is None or r.saddle_positions is None:
            return None
        if self.rec["tau_sad"] is None:
            return None
        d = _minimum_image(
            r.saddle_positions - self.rec["tau_sad"], self.rec["cell"], self.rec["pbc"]
        )
        return float(np.max(np.abs(d)))

    def problems(self, tolerance: float = TOLERANCE) -> list:
        """Why the climb does not match pARTn's; empty if it does."""
        found = [] if self.report.passed else [self.report.describe()]
        r = self.report.result
        if self.saddle:
            if self.report.next_move != "over":
                found.append(
                    f"after the saddle step the core asked for {self.report.next_move}"
                )
            if r is None or r.force_calls_saddle != self.report.last_step + 1:
                calls = None if r is None else r.force_calls_saddle
                found.append(
                    f"saddle after {calls} calls, pARTn's at step {self.report.last_step}"
                )
            position = self.saddle_position
            if position is not None and position > tolerance:
                found.append(f"saddle positions differ by {position:.3e} A")
        else:
            reason = None if r is None else r.reason
            if self.report.next_move is not None:
                found.append(f"the core went on to {self.report.next_move}")
            if reason != partn_failure_reason(self.rec):
                found.append(
                    f"core ends with {reason}, pARTn with {partn_failure_reason(self.rec)}"
                )
        return found


def check_climb(
    rec: dict, tolerance: float = TOLERANCE, keep_going: bool = False
) -> ClimbCheck:
    """Replay a recording to pARTn's saddle or failure, and compare the end."""
    return ClimbCheck(
        rec, replay(rec, tolerance, last_step=climb_end(rec), keep_going=keep_going)
    )


def _main(folder: str) -> None:
    for path in sorted(Path(folder).glob("search*.npz")):
        rec = load_recording(path)
        report = replay(rec)
        print(report.describe())
        for s in report.steps:
            extra = ""
            if s.dt is not None:
                extra = f"  dt {s.dt:.1e}  alpha {s.alpha:.1e}  v {s.velocity:.2e} A/ps"
            if s.mode is not None:
                ev = "unset" if s.eigval is None else f"{s.eigval:.2e}"
                end = "  end of block" if s.block_end else ""
                extra = f"  eigval {ev}  mode {s.mode:.2e} (sign {s.mode_sign:+d}){end}"
            print(
                f"  step {s.step:3d}  eval {s.evaluation:4d}  {s.block:5s}  dx {s.position:.2e} A{extra}"
            )
        print(
            f"  repeats skipped {report.repeats_skipped} (in the recording {report.repeats_total})"
        )
        if holds_climb(rec):
            check = check_climb(rec)
            print("  climb: " + check.report.describe())
            for problem in check.problems()[1 if not check.report.passed else 0 :]:
                print(f"    {problem}")
            if check.saddle and check.energy is not None:
                print(
                    f"    saddle energy {check.energy:.2e} eV, eigenvalue {check.eigval:.2e} eV/A^2, "
                    f"positions {check.saddle_position:.2e} A"
                )
        checks = check_blocks(rec)
        bad = [c for c in checks if c.problems()]
        every = [s for c in checks for s in c.report.steps if s.block == "LANC"]
        print(
            f"  Lanczos blocks alone: {len(checks) - len(bad)} of {len(checks)} match, "
            f"probes {min(c.block.probes for c in checks)}-{max(c.block.probes for c in checks)}, "
            f"{sum(not c.block.returns for c in checks)} without the move back"
        )
        print(
            f"    largest at block ends: eigenvalue {max(c.end(0) for c in checks):.2e} eV/A^2 "
            f"(scaled {max(c.end(1) for c in checks):.2e}), mode {max(c.end(2) for c in checks):.2e}"
        )
        print(
            f"    largest at any Lanczos step: position {max(s.position for s in every):.2e} A, "
            f"eigenvalue {_largest(every, 'eigval', '{:.2e}')} eV/A^2 "
            f"(scaled {_largest(every, 'eigval_scaled', '{:.2e}')}), mode {_largest(every, 'mode', '{:.2e}')}"
        )
        for c in bad:
            print(f"    block at step {c.block.start}: " + "; ".join(c.problems()))


if __name__ == "__main__":
    _main(sys.argv[1])
