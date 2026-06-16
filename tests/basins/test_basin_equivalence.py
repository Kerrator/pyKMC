"""Basin equivalence harness — the validation backbone for the parallel merge.

Reusable comparison helpers that decide whether two basin connectivity tables (and
their selected exits) describe the **same basin**, plus a no-MPI baseline check that
the helpers correctly judge the golden Cu basin (self-consistent) and catch
perturbations (dropped transition, flipped transient flag, perturbed barrier).

The same helpers gate the exact ``parallel == serial`` check once the parallel basin
code is ported (Track 2): a parallel run's ``basin.connectivity_table.df`` and
``BasinOutput`` are compared against a serial run / the golden reference.

Two comparison levels:
  * **invariant** (default): relabeling-invariant — transient/absorbing/total state
    counts, row count, and a per-transition attribute multiset (the "edge signature").
    Two basins identical up to state renumbering pass. This is the right level for
    parallel-vs-serial, where discovery order can differ.
  * **strict**: additionally requires identical state-index numbering (sorted
    row-by-row equality). Appropriate when the *same* deterministic algorithm
    produced both tables (serial-vs-serial / serial-vs-golden).
"""
from __future__ import annotations

import collections

import numpy as np
import pandas as pd
import pytest

_INT_COLS = ["state", "state_connexion", "event_connexion", "central_atom", "sym"]
_FLOAT_COLS = ["dE_forward", "k_forward", "dE_backward", "k_backward"]


def _num(x: object, decimals: int = 6) -> object:
    """NaN-safe rounding for use in hashable signature keys (NaN -> the string 'nan',
    so NaN compares equal to NaN inside the multiset)."""
    f = float(x)
    return "nan" if f != f else round(f, decimals)


def basin_state_counts(df: pd.DataFrame) -> dict[str, int]:
    """Relabeling-invariant state counts from a connectivity table.

    Transient states are the explored sources (appear in the ``state`` column);
    absorbing states appear only as ``state_connexion`` targets.
    """
    transient = set(df["state"])
    all_states = transient | set(df["state_connexion"])
    return {
        "n_transient": len(transient),
        "n_absorbing": len(all_states) - len(transient),
        "n_total": len(all_states),
        "n_rows": int(len(df)),
    }


def basin_edge_signature(df: pd.DataFrame, *, decimals: int = 6) -> collections.Counter:
    """Relabeling-invariant multiset of per-transition attributes.

    Ignores the (discovery-order-dependent) state index *labels* and captures the
    physics of each transition: event id, central atom, symmetry, transient flag,
    and rounded forward/backward barriers + rates. Two basins that are structurally
    identical up to state renumbering share this signature.
    """
    sig: collections.Counter = collections.Counter()
    for _, r in df.iterrows():
        sig[(
            int(r["event_connexion"]),
            int(r["central_atom"]),
            int(r["sym"]),
            bool(r["transient"]),
            _num(r["dE_forward"], decimals),
            _num(r["k_forward"], decimals),
            _num(r["dE_backward"], decimals),
            _num(r["k_backward"], decimals),
        )] += 1
    return sig


def assert_connectivity_equivalent(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    *,
    strict: bool = False,
    rtol: float = 1e-6,
    atol: float = 1e-9,
) -> None:
    """Assert two basin connectivity tables describe the same basin.

    Always checks relabeling-invariant structure (state counts, row count, edge
    signature). With ``strict=True`` also requires identical state-index numbering.
    """
    ca, cb = basin_state_counts(df_a), basin_state_counts(df_b)
    assert ca == cb, f"basin state counts differ:\n  A={ca}\n  B={cb}"

    sa, sb = basin_edge_signature(df_a), basin_edge_signature(df_b)
    if sa != sb:
        only_a = list((sa - sb).elements())
        only_b = list((sb - sa).elements())
        raise AssertionError(
            f"edge signatures differ: {len(only_a)} transition(s) only in A, "
            f"{len(only_b)} only in B (up to 3 each):\n"
            f"  A-only: {only_a[:3]}\n  B-only: {only_b[:3]}"
        )

    if strict:
        a = df_a.sort_values(_INT_COLS).reset_index(drop=True)
        b = df_b.sort_values(_INT_COLS).reset_index(drop=True)
        for c in _INT_COLS:
            assert np.array_equal(a[c].to_numpy(), b[c].to_numpy()), (
                f"strict mismatch in integer column {c!r}"
            )
        assert np.array_equal(
            a["transient"].astype(bool).to_numpy(), b["transient"].astype(bool).to_numpy()
        ), "strict mismatch in 'transient'"
        for c in _FLOAT_COLS:
            np.testing.assert_allclose(
                a[c].to_numpy(dtype=float), b[c].to_numpy(dtype=float),
                rtol=rtol, atol=atol, equal_nan=True,
                err_msg=f"strict float mismatch in {c!r}",
            )


def assert_basin_outputs_equal(out_a: object, out_b: object, *, rtol: float = 1e-6) -> None:
    """Assert two BasinOutput results select the same exit (parallel == serial)."""
    assert out_a.exit_state == out_b.exit_state, (
        f"exit_state differs: {out_a.exit_state} vs {out_b.exit_state}"
    )
    assert out_a.from_state == out_b.from_state, (
        f"from_state differs: {out_a.from_state} vs {out_b.from_state}"
    )
    assert out_a.central_atom == out_b.central_atom, (
        f"central_atom differs: {out_a.central_atom} vs {out_b.central_atom}"
    )
    np.testing.assert_allclose(
        float(out_a.t_exit), float(out_b.t_exit), rtol=rtol, err_msg="t_exit differs"
    )


# ---------------------------------------------------------------------------
# Baseline validation (no MPI): the harness correctly judges the golden Cu basin.
# These run in plain pytest; the live serial/parallel basin runs (which need
# mpirun) are exercised separately and reuse the helpers above.
# ---------------------------------------------------------------------------

def test_harness_golden_self_consistent(connectivity_table_Cu) -> None:
    """The golden Cu connectivity table equals itself at the strictest level."""
    df = connectivity_table_Cu.df
    counts = basin_state_counts(df)
    assert counts["n_rows"] == len(df)
    assert counts["n_transient"] >= 1, "golden basin should have >=1 transient state"
    assert counts["n_absorbing"] >= 1, "golden basin should have >=1 absorbing state"
    assert_connectivity_equivalent(df, df.copy(), strict=True)


def test_harness_detects_dropped_transition(connectivity_table_Cu) -> None:
    """Dropping a transition is caught (row count + signature)."""
    df = connectivity_table_Cu.df
    with pytest.raises(AssertionError):
        assert_connectivity_equivalent(df, df.iloc[:-1].copy())


def test_harness_detects_flipped_transient(connectivity_table_Cu) -> None:
    """Flipping a transient flag changes the edge signature."""
    df = connectivity_table_Cu.df
    perturbed = df.copy()
    i = perturbed.index[0]
    perturbed.loc[i, "transient"] = not bool(perturbed.loc[i, "transient"])
    with pytest.raises(AssertionError):
        assert_connectivity_equivalent(df, perturbed)


def test_harness_detects_perturbed_barrier(connectivity_table_Cu) -> None:
    """A changed barrier is caught by the edge signature."""
    df = connectivity_table_Cu.df
    perturbed = df.copy()
    i = perturbed.index[0]
    perturbed.loc[i, "dE_forward"] = float(perturbed.loc[i, "dE_forward"]) + 0.5
    with pytest.raises(AssertionError):
        assert_connectivity_equivalent(df, perturbed)


def test_harness_invariant_under_relabeling(connectivity_table_Cu) -> None:
    """Renumbering state indices (a +1000 shift) leaves the basin equivalent at the
    invariant level but breaks strict equality — exactly the property needed to
    compare parallel vs serial runs that discover states in a different order."""
    df = connectivity_table_Cu.df
    relabeled = df.copy()
    relabeled["state"] = relabeled["state"] + 1000
    relabeled["state_connexion"] = relabeled["state_connexion"] + 1000
    # invariant: still the same basin
    assert_connectivity_equivalent(df, relabeled, strict=False)
    # strict: numbering differs -> must fail
    with pytest.raises(AssertionError):
        assert_connectivity_equivalent(df, relabeled, strict=True)
