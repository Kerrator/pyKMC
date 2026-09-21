"""``Refinement.execute`` and the optional ``System.index`` (V7).

``System.index`` defaults to ``None`` and is populated by ``System.from_file``
and the basin copies; a programmatic ``System()`` has none. The result loop
of ``Refinement.execute`` dereferenced it unconditionally in every style, so a
constant-mode refinement on such a system died with a TypeError. Constant mode
needs no crop identities (its reconstruction falls back to the rcut
neighbours); it skips ``crop_atom_ids``. htst/rpa cannot recycle or
reconstruct without stable identities and say so with a named error.
"""

from __future__ import annotations

from concurrent.futures import Future
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pykmc.config import Config, RateConstantConfig
from pykmc.refinement import Refinement
from pykmc.result import EventRefinementOutput, Ok
from pykmc.system import System
from tests.lifecycle.conftest import DATA_INPUT

N_ATOMS = 6


def _system(index: Any) -> System:
    positions = (
        np.array([[float(i), 0.0, 0.0] for i in range(N_ATOMS)], dtype=float) + 5.0
    )
    return System(
        types=np.array(["Ni"] * N_ATOMS),
        positions=positions,
        cell=np.eye(3) * 20.0,
        pbc=(True, True, True),
        index=index,
    )


def _frame(style: str) -> pd.DataFrame:
    columns: dict[str, Any] = {
        "idx_ref": [3],
        "event_id": ["topology"],
        "energy_barrier": [0.4],
        "k": [1.0e12 * np.exp(-0.4 / 0.0259)],
        "sym_matrix": [[np.eye(3)]],
        "sym_perm": [[np.arange(3)]],
    }
    if style != "constant":
        columns.update({"nu0": [5.0e12], "nu0_status": ["ok"], "nu0_reason": [""]})
    return pd.DataFrame(columns)


def _refinement(style: str, system: System, calls: list[Any]) -> Refinement:
    config = Config.from_ini_file(DATA_INPUT)
    rate = RateConstantConfig(style=style, k0=1.0, T=config.rateconstant.T)
    config = config.model_copy(update={"rateconstant": rate})
    neighbors = np.array([0, 1, 2], dtype=int)
    logger = SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        progress_bar=lambda *a, **k: None,
    )
    refinement = Refinement(
        config,
        logger,
        system,
        SimpleNamespace(get_neighbors=lambda *_: neighbors.copy()),
        SimpleNamespace(get_atoms_with_id=lambda _: [1]),
        manager=None,
    )

    def refine_single(self, at_idx, dfevent, total_energy, future_context, e_thr):
        calls.append((at_idx, int(dfevent["idx_ref"])))
        saddle = np.array(self.system.positions, copy=True)
        saddle[at_idx, 0] += 0.3
        future: Future = Future()
        future.set_result(
            Ok(
                EventRefinementOutput(
                    central_atom_index=at_idx,
                    saddle_positions=saddle,
                    E_saddle=0.4,
                    refined="T",
                )
            )
        )
        future_context[future] = {
            "min2_positions": saddle[neighbors] + 0.3,
            "num_reference_event": int(dfevent["idx_ref"]),
            "reference_energy_barrier": float(dfevent["energy_barrier"]),
            "neighbors": neighbors.copy(),
            "estimate": self._inherited_estimate(dfevent),
        }
        return [future]

    refinement.refine_single = refine_single.__get__(refinement, Refinement)
    return refinement


def test_constant_mode_refines_a_system_without_identities() -> None:
    calls: list[Any] = []
    refinement = _refinement("constant", _system(None), calls)
    refinement.execute(_frame("constant"), total_energy=0.0)
    assert calls == [(1, 3)]
    results = refinement.get_successes_results()
    assert len(results) == 1
    out = results[0]
    assert out.num_reference_event == 3 and out.refined == "T"
    assert out.crop_atom_ids is None, "no identities to record in constant mode"
    assert out.saddle_positions.shape == (3, 3), "the saddle is still cropped"
    assert out.nu0_source is None


def test_constant_mode_records_identities_when_the_system_has_them() -> None:
    calls: list[Any] = []
    index = np.array([40, 8, 19, 2, 77, 5])
    refinement = _refinement("constant", _system(index), calls)
    refinement.execute(_frame("constant"), total_energy=0.0)
    (out,) = refinement.get_successes_results()
    assert out.crop_atom_ids == (40, 8, 19)


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_prefactor_styles_require_stable_identities(style: str) -> None:
    calls: list[Any] = []
    refinement = _refinement(style, _system(None), calls)
    with pytest.raises(RuntimeError, match="stable atom identities") as captured:
        refinement.execute(_frame(style), total_energy=0.0)
    assert "System.index" in str(captured.value)
    assert style in str(captured.value)
    assert calls == [], "the requirement is checked before any refinement is dispatched"
