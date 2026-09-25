"""Constant mode on this tree reproduces develop ``c6f2824`` on ``examples/Si_vac``.

The oracle ``tests/data/constant_endpoint_develop_c6f2824.json`` was recorded
once on develop ``c6f2824`` with :mod:`tests.regression.si_vac_constant_harness`
in a subprocess launched with ``PYTHONHASHSEED=20260919`` (develop orders the
new environments by string hash; this tree sorts them). Its ``meta`` holds the
command, the seeds, the INI and the versions. The test runs the same harness
in-process on this tree (511-atom SW-Si with one vacancy, ``n_sessions = 1``,
``random``/NumPy seed 12345 set before ``KMC`` is built, ``zseed = 12345``,
``nsearch = 2``, ``rcut = 6.3``, ``style = constant``, ``k0 = 10``,
``T = 300``, 10 steps) and compares:

- the executed ``(atom_index, dE)`` of every step: atom indices equal, barriers
  within 1e-9 eV;
- the sorted multiset of ``(dE_forward, dE_backward)`` pairs of the final
  reference table, within 1e-9 eV;
- ``k_tot`` and the total simulated time ``T`` after step 10, within 1e-6
  relative.

Plain pytest collects it; no MPI launch and no opt-in variable are needed. The
run takes a few seconds.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from tests.regression.si_vac_constant_harness import run_si_vac_constant

ROOT = Path(__file__).resolve().parents[2]
ORACLE = ROOT / "tests" / "data" / "constant_endpoint_develop_c6f2824.json"
N_STEPS = 10
SEED = 12345
ZSEED = 12345
NSEARCH = 2
RATE_RTOL = 1e-6
BARRIER_ATOL_EV = 1e-9


def _step_context(oracle: dict[str, Any], run: dict[str, Any], step: int) -> str:
    """Describe step ``step`` (1-based) on both trees for a failure message.

    Parameters
    ----------
    oracle, run : dict
        Harness results of develop and of this tree.
    step : int
        Step number.

    Returns
    -------
    str
        Environments, searched atoms and executed rows of the step on each tree.

    """
    lines = []
    for name, result in (("develop", oracle), ("this tree", run)):
        steps = result["steps"]
        if step > len(steps):
            lines.append(f"  {name}: no step {step} recorded")
            continue
        data = steps[step - 1]
        envs = [env[:16] + "..." for env in data.get("new_environments", [])]
        lines.append(
            f"  {name}: new environments {envs}, searched atoms "
            f"{data.get('central_atoms')}, draws {data.get('draws')}, "
            f"attempts {data.get('attempts')}"
        )
    return "\n".join(lines)


def _first_selection_divergence(
    expected: list[list[float]], observed: list[list[float]]
) -> int | None:
    """Return the first 1-based step whose executed event differs, or None."""
    for step, (exp, obs) in enumerate(zip(expected, observed, strict=False), start=1):
        if int(exp[0]) != int(obs[0]) or abs(exp[1] - obs[1]) > BARRIER_ATOL_EV:
            return step
    if len(expected) != len(observed):
        return min(len(expected), len(observed)) + 1
    return None


def _pairs_mismatch(expected: list[list], observed: list[list]) -> str | None:
    """Return a description of the first differing reference pair, or None."""
    if len(expected) != len(observed):
        return f"{len(observed)} pairs on this tree, {len(expected)} on develop"
    for i, (exp, obs) in enumerate(zip(expected, observed, strict=True)):
        for a, b in zip(exp, obs, strict=True):
            if (a is None) != (b is None) or (
                a is not None and abs(a - b) > BARRIER_ATOL_EV
            ):
                return f"sorted pair {i}: develop {exp}, this tree {obs}"
    return None


def test_constant_si_vac_reproduces_develop(tmp_path: Path) -> None:
    """Ten constant-mode Si_vac steps select and time exactly what develop did."""
    oracle = json.loads(ORACLE.read_text())
    meta = oracle["meta"]
    assert meta["git_head"].startswith("c6f2824"), meta["git_head"]
    assert (meta["seeds"]["python"], meta["seeds"]["zseed"], meta["nsearch"]) == (
        SEED,
        ZSEED,
        NSEARCH,
    ), "oracle was recorded with other seeds or nsearch"
    assert oracle["n_steps"] == N_STEPS, oracle["n_steps"]

    run = run_si_vac_constant(
        ROOT, tmp_path / "run", n_steps=N_STEPS, seed=SEED, zseed=ZSEED, nsearch=NSEARCH
    )

    assert run["exit_code"] in (None, 0), (
        f"run exited with status {run['exit_code']!r} after {run['n_steps']} steps "
        f"(close calls {run['meta']['close_calls']})"
    )
    step = _first_selection_divergence(oracle["selected"], run["selected"])
    assert step is None, (
        f"executed events diverge from develop c6f2824 at step {step} "
        f"(atom_index exact, dE within {BARRIER_ATOL_EV} eV)\n"
        f"  develop:   {oracle['selected']}\n  this tree: {run['selected']}\n"
        + _step_context(oracle, run, step)
    )
    assert run["n_steps"] == N_STEPS, run["n_steps"]

    mismatch = _pairs_mismatch(oracle["reference_pairs"], run["reference_pairs"])
    assert mismatch is None, (
        f"final reference table (dE_forward, dE_backward) multiset differs from "
        f"develop: {mismatch}\n  develop:   {oracle['reference_pairs']}\n"
        f"  this tree: {run['reference_pairs']}"
    )

    for key, unit in (("k_tot", "ps^-1"), ("T", "s")):
        assert math.isclose(run[key], oracle[key], rel_tol=RATE_RTOL, abs_tol=0.0), (
            f"{key} after step {N_STEPS}: this tree {run[key]!r} {unit}, develop "
            f"{oracle[key]!r} {unit} (relative difference "
            f"{abs(run[key] - oracle[key]) / abs(oracle[key]):.3e}, "
            f"tolerance {RATE_RTOL})"
        )
