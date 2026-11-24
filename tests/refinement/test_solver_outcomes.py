"""Scientific failures are results; a failed Future is not a rejected saddle."""

from unittest.mock import call

import numpy as np
import pytest

from pykmc.result import ErrorType
from .test_paths import basin_for


@pytest.mark.parametrize("crop", [False, True])
@pytest.mark.parametrize(
    "trace,success,attempts",
    [
        ([(0, 0.249)], True, 1),
        ([(0, 0.25)], False, 2),  # exactly at the cutoff is not accepted
        ([(0, 0.251), (0, 0.249)], True, 2),
        ([(2, 0.1), (0, 0.1)], True, 2),
        ([(2, 0.1)], False, 2),
    ],
)
def test_solver_retry_and_next_job(make_case, crop, trace, success, attempts):
    sim, _ = make_case(crop)
    worker = sim.manager
    # Drive one symmetry per real caller invocation, not two pARTn calls per row.
    df = sim.reference_table.table
    for col in ["sym_perm", "sym_matrix"]:
        df.at[0, col] = df.at[0, col][:1]
    sim.config.control.refine_thr = 2.0
    worker.trace = trace
    (result,) = sim.execute_refinements(df).results
    assert result.is_ok() == success
    if not success:
        assert result.err_value().type == ErrorType.EVENT_NOT_FOUND
        assert result.err_value().variables == {"n_ref_event": 37}
    (art,) = worker.artns
    commands = [c.args[0] for c in worker.engine.lmp.command.call_args_list]
    assert commands.count("unfix 10") == attempts
    assert len([c for c in commands if c.startswith("fix 10 all artn ")]) == attempts
    for k, v in [
        ("lpush_final", False),
        ("lmove_nextmin", False),
        ("engine_units", "lammps/metal"),
    ]:
        assert call(k, v) in art.set.call_args_list
    assert [c.args[1] for c in art.set.call_args_list if c.args[0] == "zseed"] == [
        193
    ] * (attempts + (not success))
    push_ids = [c.args[1] for c in art.set.call_args_list if c.args[0] == "push_ids"]
    # Full id 3 is LAMMPS id 4; the crop contains full ids [1,3,4].
    np.testing.assert_array_equal(push_ids, [[2 if crop else 4]])
    if crop:
        assert "group core id 3 2 1" in commands
        assert "group buffer id 1 3" in commands
    worker.trace = [(0, 0.1)]
    (nxt,) = sim.execute_refinements(df).results
    assert nxt.is_ok() and nxt.ok_value().dE_forward == 0.75
    assert len(worker.artns) == 2


@pytest.mark.parametrize("consumer", ["ordinary", "basin"])
def test_raised_future_is_not_a_scientific_rejection(consumer, make_case):
    sim, _ = make_case()
    sim.manager.exception = RuntimeError("scripted worker transport")
    with pytest.raises(RuntimeError, match="scripted worker transport"):
        if consumer == "ordinary":
            sim.execute_refinements(sim.reference_table.table)
        else:
            basin_for(sim).refine_absorbing(sim.system)
