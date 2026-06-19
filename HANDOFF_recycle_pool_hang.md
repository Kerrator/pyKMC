# Handoff: a recycle reconstruction hangs the whole engine pool

**Status:** **FIXED** (2026-06-19) for the reproducible + documented triggers; one
narrow residual gap (an unforeseen mid-op `error->one` from *valid* input) is left
as a recommended follow-up — see "Residual gap" below.
**Severity:** high — a single recoverable engine error stalled the entire run for
hours (until the per-run timeout) instead of failing fast.
**Found:** 2026-06-18, during the `feature_profile` 12-run benchmark
(`benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/04_recycle`).
**Not related to the HTST deadlock fix** (`SK_basin_htst_binary` @ `5e13d91`):
`recycle` is active-volume-off + constant-rate, so it never touches
`compute_event_prefactors`, `free_radius`, or `define_AV`.

---

## Corrected mechanism (after stack-sampling the hung ranks)

Stack-sampling the live hung processes (macOS `sample`, no source changes) revised
the original two-defect model in two important ways:

1. **The survivor rank is trapped *inside* `lammps_command`, below the Python layer.**
   When a LAMMPS `error->one` (per-rank error, e.g. "Non-numeric atom coords",
   `domain.cpp:790`) fires on only the subset of engine ranks owning the bad atom,
   that rank unwinds to the Python `finally`-barrier in
   `mpi_api_engine.py:_handle_message`, **while the surviving rank stays blocked in
   liblammps's own `MPI_Sendrecv` inside the still-running `minimize`**. Rank 0 then
   blocks in `receive_status`. Confirmed frames: erroring rank in `MPI_Barrier`,
   survivor in `lammps_command`→`MPI_Sendrecv`, rank 0 in `MPI_Mprobe/recv`.
   Because the survivor never returns from `lammps_command`, **no *post-op*
   Python-level resync can recover it** — the error must be caught *before* LAMMPS
   runs, or the pool torn down *after* the error surfaces.

2. **A symmetric error (`error->all`, e.g. "Lost atoms" — the production trigger)
   surfaces cleanly** as a `RuntimeError` on rank 0 (both engine ranks raise in
   lockstep, hit the `finally` barrier together, send the error reply). The
   production hang was therefore **not** the surfacing — it was that the propagating
   `RuntimeError` left `KMC.run()` **before `_close()`**, so `manager.close_all()`
   was skipped and the (still-looping) engine ranks were stranded busy-spinning in
   `run_engine_loop`.

Note on the original reproducer: on macOS LAMMPS (22 Jul 2025) + the 256-atom FCC
fixture, the old `atom1 = atom0 + 1e-4 Å` trigger **did not error at all** (the
minimize converged); its exit-124 was a *false positive* — the assertion failed and
rank 0 exited before `close_all`, stranding the ranks. The test now uses a NaN
coordinate, a faithful `error->one` trigger.

---

## The fix (three changes + a faithful test)

1. **Pre-op collective validation (the asymmetric trigger).**
   `pykmc/enginemanager/lmpi/lammps_operations.py` — `_require_finite_positions()`,
   called at the top of `minimize_with_results`. Every engine rank receives the same
   broadcast positions, so the check raises **symmetrically** (all ranks together)
   *before* LAMMPS can raise `error->one` asymmetrically. The failure then travels
   the symmetric path, which unwinds cleanly.

2. **Always tear the pool down on failure (the production strand).**
   `pykmc/kmc.py` — `run()` now wraps the body: `try: self._run_impl() except
   Exception: self.manager.close_all(); raise`. A surfaced engine error tears the
   pool down (unstranding the engine ranks) and re-raises so the failure stays
   visible. The normal `_close()`→`sys.exit()` path raises `SystemExit`
   (not `Exception`), so the pool is closed exactly once — no double close.

3. **Graceful recycle (Defect 1, the production trigger geometry).**
   `pykmc/reconstruction.py` — both `global_minimize_with_results` calls in
   `reconstruct()` now catch `RuntimeError` and return
   `Err(RECONSTRUCTION_MINIMIZE_FAILED)` (new `ErrorType`). An unstable pushed
   geometry that errors in LAMMPS now **drops that one reconstruction** and the run
   continues, instead of crashing. (The engine ranks have already handled the error
   symmetrically and are back in their service loop, so the manager stays usable.)

4. **Faithful regression test.**
   `tests/test_lammps_engine_api_mpi.py::test_engine_error_during_global_minimize_does_not_hang`
   now uses a NaN trigger, puts `close_all()` in a `finally`, and the
   `PYKMC_REPRODUCE_RECYCLE_HANG` gate is **removed** (it is now a live guard:
   passes, exit 0, under any `mpirun -n >= 3`; skips with fewer ranks).

---

## Verification

- **MPI reproducer** `test_engine_error_during_global_minimize_does_not_hang`:
  RED before fix = hang (exit 124, all 3 ranks ~98% CPU); GREEN after = `1 passed`,
  exit 0, with **both** engine ranks raising the validation error symmetrically and
  both engines closing cleanly (`mpirun -n 3 --oversubscribe`).
- **`tests/test_kmc.py`** (non-MPI): `run()` closes the pool when the body raises and
  re-raises; a normal `SystemExit` is not caught (no double close).
- **`tests/test_reconstruction.py`** (non-MPI): a minimize `RuntimeError` becomes
  `Err(RECONSTRUCTION_MINIMIZE_FAILED)`; the Ok path still works.
- ruff/mypy: zero net-new findings in all four changed source files.

---

## Residual gap (recommended follow-up — NOT yet implemented)

A LAMMPS `error->one` arising mid-minimize from an otherwise **finite/valid** geometry
would still trap the survivor: the finite pre-check can't catch it, and no exception
surfaces on rank 0 (it blocks in `receive_status`), so the `run()` teardown guard never
fires. This is rare (the dominant asymmetric trigger, non-finite coords, is now caught)
but not impossible.

The only generic defense is a **per-op wall guard**: bound rank 0's wait for an engine
reply/status; on timeout, log and `MPI.COMM_WORLD.Abort()` so the job fails in minutes
instead of stalling for the full per-run timeout. It is deferred because (a) it needs a
timeout *policy* (a value generous enough not to abort legitimately slow ops — large
minimizes, pARTn searches, basins) and (b) it cannot be verified locally without a
cluster repro of a valid-input `error->one`. Implement + verify on the cluster.

## Key files (post-fix)

| what | location |
|---|---|
| pre-op finite validation | `pykmc/enginemanager/lmpi/lammps_operations.py` (`_require_finite_positions`, called in `minimize_with_results`) |
| run() teardown guard | `pykmc/kmc.py` (`run` → `_run_impl`) |
| graceful recycle | `pykmc/reconstruction.py` (`reconstruct`, both minimize calls) |
| new error type | `pykmc/result.py` (`ErrorType.RECONSTRUCTION_MINIMIZE_FAILED`) |
| MPI regression test | `tests/test_lammps_engine_api_mpi.py::test_engine_error_during_global_minimize_does_not_hang` |
| unit tests | `tests/test_kmc.py`, `tests/test_reconstruction.py` |
| engine service loop + barriers + error reply | `pykmc/enginemanager/lmpi/engines/mpi_api_engine.py` |
| error surfaces to rank 0 | `pykmc/enginemanager/lmpi/sessions/mpi_api_sessions.py` |
| killed-run evidence (original) | `benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/04_recycle/run_stdout.log` |
