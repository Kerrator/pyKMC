# Handoff: a recycle reconstruction hangs the whole engine pool

**Status:** open bug, reproduced by a test, **not yet fixed.**
**Severity:** high — a single recoverable engine error stalls the entire run for
hours (until the per-run timeout) instead of failing fast.
**Found:** 2026-06-18, during the `feature_profile` 12-run benchmark
(`benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/04_recycle`).
**Not related to the HTST deadlock fix** (`SK_basin_htst_binary` @ `5e13d91`):
`recycle` is active-volume-off + constant-rate, so it never touches
`compute_event_prefactors`, `free_radius`, or `define_AV`.

---

## Symptom (observed)

`04_recycle` (recycle, AV-off, constant rate, 10 steps) ran steps 0–2 normally,
then **hung on step 3 for 4.7 h** with no further output. All 19 engine ranks
pinned at ~100 % CPU (busy-spin), rank 0 idle; had to kill the session manually.

The engine error captured in `04_recycle/run_stdout.log`:

```
RuntimeError: Remote minimize_with_results failed on engine rank 1 (RuntimeError):
  Error executing command 'minimize 1e-10 1e-12 10000 10000':
  ERROR: Lost atoms: original 32000 current 31992
  (../thermo.cpp:526)
```

So a `minimize` lost 8 atoms, the engine reported the error to rank 0 — and the
pool then hung instead of failing.

---

## Root cause — two distinct defects

**Defect 1 (physics, the trigger).** The recycle path reconstructs min1/min2 from
a recycled saddle by pushing the saddle's neighbours toward the target minimum,
then minimizing — `pykmc/reconstruction.py:54-61`:

```python
tmp_positions = copy.deepcopy(saddle_positions)
saddle_toward_min1_pos = push_towards(saddle_positions[neighbors],
                                      supposed_min1_positions,
                                      fraction=self.config.reconstruction.push_fraction,  # 0.15
                                      cell=cell)
tmp_positions[neighbors] = saddle_toward_min1_pos
min1_pos, _ = self.manager.global_minimize_with_results(self.config,
                                                        positions=tmp_positions,
                                                        types=self.types)
```

When a recycled event is mapped onto a *different* local environment, this pushed
geometry can be unstable (atoms too close / outside the box), so the global
`minimize` loses atoms and LAMMPS raises.

**Defect 2 (robustness — the one that turns a recoverable error into a multi-hour
hang).** The engine-side LAMMPS error *does* surface to rank 0 as a `RuntimeError`
(`pykmc/enginemanager/lmpi/sessions/mpi_api_sessions.py:170` `_receive_result_or_error`),
but the **engine ranks are not torn down or re-synced**: they stay in
`run_engine_loop` (`pykmc/enginemanager/lmpi/engines/mpi_api_engine.py:124`) and
**busy-spin at ~100 % CPU in a stuck collective** while rank 0 blocks. The pool is
left desynchronised, so neither the failing op nor a subsequent `close_all()`
returns — the job hangs until the harness's 16 h per-run timeout.

The engine already has an error-reply path (`mpi_api_engine.py:153-169`, the
`__handler_error__` marker → tag-1 "error" reply) and barriers around each handler
(`_handle_message`, lines 196 and 226). The likely failure: for a **global**,
collective `minimize` across the full `global_engine_comm`, the LAMMPS error does
not leave every rank in lockstep through those barriers (one rank unwinds the
exception while another is still inside the collective), so the `engine_comm`
barrier/bcast deadlocks and the ranks spin. That collective desync — not the
`RuntimeError` itself — is what needs fixing.

---

## Reproduction

**Test:** `tests/test_lammps_engine_api_mpi.py::TestLammpsApiMpiEngine::test_engine_error_during_global_minimize_does_not_hang`

It drives the same `global_minimize_with_results` path with a deliberately
degenerate geometry (collapse atom 1 onto atom 0 ~1e-4 Å apart → the EAM repulsion
blows the minimize up into a LAMMPS error), then asserts the error surfaces as a
`RuntimeError` on rank 0 **and** the pool stays shut-downable (`close_all()`
returns). Reaching the end of the test is the regression check.

The test is **gated behind `PYKMC_REPRODUCE_RECYCLE_HANG`** so it can't stall the
normal suite (it hangs while the bug is open). To run it:

```bash
source pykmc_env/bin/activate
PYKMC_REPRODUCE_RECYCLE_HANG=1 timeout 130 mpirun \
    -x PYKMC_REPRODUCE_RECYCLE_HANG -n 3 --oversubscribe python -m pytest \
    tests/test_lammps_engine_api_mpi.py \
    -k engine_error_during_global_minimize_does_not_hang -s
```

**Observed (current, buggy):** the run hangs — the 2 engine ranks spin at ~99 %
CPU while rank 0 blocks, and the `timeout` wrapper kills it at 130 s. **Exit
code 124 == bug reproduced** (verified 2026-06-18). After a fix the test should
pass (exit 0); at that point drop the `PYKMC_REPRODUCE_RECYCLE_HANG` gate so it
becomes a live regression guard. It uses the same skip-if-`COMM_WORLD < 3` /
wall-guard pattern as the HTST multi-rank reproducer
(`test_compute_event_prefactors_multirank_session`).

---

## Suggested fix directions (in priority order)

1. **Make an engine error re-sync / tear down the pool (Defect 2 — the important
   one).** When any rank's handler errors, every rank of the relevant
   `engine_comm`/`global_engine_comm` must reach the same post-error point (the
   `__handler_error__` marker + barriers must be hit by *all* ranks, even when the
   LAMMPS exception fires on a subset), and rank 0 must be able to `close_all()`
   afterwards without blocking. A clean-shutdown-on-error path makes the failure
   fast and recoverable regardless of the trigger.
2. **Harden the reconstruction geometry (Defect 1).** Validate/clamp the pushed
   `tmp_positions` before `global_minimize_with_results` (reject overlapping atoms;
   wrap into the box), or catch the minimize error in `reconstruction.py` and
   return an `Err(...)` so recycle drops that reconstruction instead of crashing.
3. **Add a per-op wall guard** so a stuck collective fails the run in minutes
   rather than spinning for the full 16 h timeout.

## Key files / lines

| what | location |
|---|---|
| recycle reconstruction (trigger) | `pykmc/reconstruction.py:54-61`, `:77-80` |
| recycle feature | `pykmc/event_recycling.py` |
| error surfaces to rank 0 | `pykmc/enginemanager/lmpi/sessions/mpi_api_sessions.py:170-185` |
| engine service loop + barriers + error reply | `pykmc/enginemanager/lmpi/engines/mpi_api_engine.py:124-169`, `:183-226` |
| global-op dispatch | `pykmc/enginemanager/lmpi/pool/manager.py:223-236` (`global_method`), `:213-220` (`close_all`) |
| reproducing test | `tests/test_lammps_engine_api_mpi.py` (`test_engine_error_during_global_minimize_does_not_hang`) |
| killed-run evidence | `benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/04_recycle/run_stdout.log` |
