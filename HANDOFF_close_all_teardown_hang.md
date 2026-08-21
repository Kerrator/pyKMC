# Handoff: `close_all()` hangs when tearing down a pool whose engine errored mid-run

**Status:** **FIXED** (2026-06-19). Successor to the (fixed) per-op recycle
pool-hang — read `HANDOFF_recycle_pool_hang.md` first.
**Branch:** `bug_recycling_pool_hang`.
**Severity:** high for recycle — every recycle config stalled at shutdown until an
external kill, so recycle could not be benchmarked to completion.
**Found:** 2026-06-19, re-running the recycle benchmark configs with the per-op
fix in place.

---

## RESOLUTION (2026-06-19)

**Confirmed root cause — a *mode-vs-teardown* bug, not the desync the hypotheses
below guessed at.** `close_all()` was mode-unaware. The global `engine_comm` spans
*every* engine rank, so when the pool is torn down while in **GLOBAL mode** — exactly
the recycle "All event reconstructions failed" path (`kmc.py`: `use_global()` →
reconstruction fails → `_close()` → `close_all()`, with no intervening `use_local()`)
— the old `global_session.close(wait_status=False)` broadcast a shutdown over the
global comm to *all* engines at once and exited their run loops together. In global
mode **only the global master rank emits a status** (`mpi_api_engine.py:144-147`), so
the *first* local `session.close(wait_status=True)` consumed that single stray status
and the *second* local session blocked forever in `receive_status()`, waiting on an
engine that had already gone. The `n_sessions=1` reproducer masked it (no second
session to strand); the handoff's observed "frozen at rank 4" was just the master of
the second chunk in that run's `np.array_split` layout. The error earlier in the run
was incidental — it only determined that teardown happened via the global-mode
`_close()`; the hang needs neither hypothesis (1) (survivor stuck below Python) nor
(2) (off-by-one status desync).

**Fix** (`pykmc/enginemanager/lmpi/pool/manager.py`, `close_all`): switch the pool to
local mode first (`self.use_local()`, a guarded no-op when already local), then close
each local session. With every engine listening on its own local comm, each `close`
is a self-contained 1-send/1-recv handshake acknowledged by exactly one engine.
`MpiApiEngine.close()` already shuts down **both** that rank's local and global LAMMPS
instance, so closing every local session tears the whole pool down — the separate
global-session close was redundant as well as unsafe and is removed.

**Test** (`tests/test_lammps_engine_api_mpi.py::test_close_all_after_error_with_multiple_sessions_does_not_hang`):
parametrized over `single_rank` (one rank per session, `mpirun -n 3`) and `multi_rank`
(≥2 ranks per session, the production benchmark shape, `mpirun -n 5`) chunk topologies.
It mirrors the recycle path (`use_global()` → handled global-minimize error → close in
global mode) and wall-guards `close_all()` in a daemon thread so a hang surfaces as an
assertion failure, not a stall. Both cases: **RED** (close_all did not return within
60 s) before the fix, **GREEN** after. The opt-in `engine_op_timeout_s` wall guard is
deliberately *not* set, so the test exercises the teardown fix rather than the abort
band-aid.

**Verified:** new test GREEN under `mpirun -n 3` and `-n 5` (RED on the old `close_all`
for both topologies); `test_engine_error_during_global_minimize_does_not_hang`,
`test_compute_event_prefactors_multirank_session`,
`test_compute_forces_and_dynamical_matrix_manager` (n_sessions=7, `-n 8`),
`tests/test_kmc.py`, `tests/test_reconstruction.py` all still pass. (Pre-existing,
unrelated failures left untouched: `tests/manager/lmpi/test_manager.py` hangs in
`broadcast_command` while the pool is in global mode — same hang on the unpatched code;
`tests/basins/test_basin.py` fails a `BASIN_NO_VIABLE_EXIT` *domain* assertion computed
before teardown — `close_all` itself returns cleanly there.)

The original analysis (kept below for context) follows.

---

## What is already fixed (don't re-do)

`HANDOFF_recycle_pool_hang.md` fixed the **per-operation** hang: a `minimize`
error during a recycle reconstruction now (1) is caught by a pre-op finite check
or surfaces symmetrically, (2) is dropped gracefully
(`Err(RECONSTRUCTION_MINIMIZE_FAILED)`), and (3) the `run()` teardown guard calls
`close_all()` on any exception. **Verified working:** the MPI reproducer is GREEN,
and a real 32k recycle run logged **108** `Lost atoms` errors all handled
gracefully (reconstruction dropped, run continued). That part is solid.

## The remaining bug — the teardown itself hangs

After all the per-op errors are handled, **`manager.close_all()` hangs**. Observed
in three separate recycle configs (`recycle`, `av+recycle`, `av+basin+htst+recycle`):
the run prints engine-close lines for most ranks, then freezes forever:

```
[Engine Rank 0] Closing LAMMPS engine.
[Engine Rank 1] Closing LAMMPS engine.
...
[Engine Rank 18] Closing LAMMPS engine.
[Session] Sending close message to engine at rank 4     <-- frozen here, indefinitely
```

- Rank 0 spins at **~99.9% CPU**; `pykmc.out`/`run_stdout.log` never advance again
  (killed by a 30-min watchdog in the benchmark — otherwise it blocks for the full
  16 h per-run timeout).
- It stalls on the session whose engine (here "rank 4") **had a handled error
  earlier in the run**. Sessions whose engines never errored close fine.
- It is independent of *why* the run is ending: it happens both after a clean
  `End of simulation` (all reconstructions failed) and after a normal mid-run
  teardown (`av+recycle` had reached step 4 with only 2 recon-fails when it ended).

So this is a **teardown / shutdown-path** bug, distinct from the per-op hang and
also distinct from the "Residual gap" in the other handoff (that was a *mid-op*
`error->one` from valid input).

## Why the existing reproducer doesn't catch it

`tests/test_lammps_engine_api_mpi.py::test_engine_error_during_global_minimize_does_not_hang`
uses **`n_sessions=1`** and PASSES (its `close_all()` in the `finally` returns).
The teardown hang only appears with **multiple sessions** (the benchmark used
`n_sessions=7`): `close_all` closes the healthy sessions, then blocks on the one
errored session. A faithful reproducer must use **`n_sessions >= 2`**, error the
engine on *one* session, then call `close_all()` and assert it returns.

## The mechanism to confirm (hypothesis)

`pool/manager.py:220` `close_all()`:
```python
if self.global_session is not None:
    self.global_session.close(wait_status=False)
for session in self.sessions:
    session.close(wait_status=True)      # <-- blocks here for the errored session
```
`sessions/mpi_api_sessions.py:145` `close(wait_status=True)` →
`send_message({"type":"close"}, expect_status=True)` →
`receive_status()` (`:113`) → `_recv(..., tag=0)` (`:87`).

So rank 0 sends the close and **waits for a status (tag 0) the errored engine
never sends**. Candidate root causes to check by stack-sampling the live hung
ranks (the technique that cracked the previous bug — macOS `sample` / Linux
`py-spy dump` / `gdb`, no source changes):
1. **A survivor engine rank is still stuck below Python** (in `lammps_command` /
   an MPI collective) from the earlier error, so it never reaches the run-loop to
   process the `close` — the same trap described in `HANDOFF_recycle_pool_hang.md`
   §"Corrected mechanism", but surfacing at close time on a multi-rank session.
2. **A tag-0/tag-1 status/reply stream desync**: the error path
   (`mpi_api_engine.py:154-161`, the `__handler_error__` reply) may leave the
   per-session status accounting off by one, so the `close`'s `receive_status`
   waits for a status that was already consumed (or never produced) after 100+
   handled errors. Check that every op sends exactly one status (`_send_status`,
   `mpi_api_engine.py:171`) and one reply, including the error case.

Confirm which by sampling: erroring/ stuck engine rank's frame (likely
`MPI_Sendrecv`/`lammps_command` or a barrier) vs rank 0's frame (expected
`MPI_Recv` in `receive_status`).

## A mitigation already exists (use it to bound the damage)

The per-op fix added an **opt-in wall guard**: `config.control.engine_op_timeout_s`
(default `None` = unchanged blocking). When set, `_recv` polls with a deadline and
`MPI.COMM_WORLD.Abort(1)` on timeout. Since this `close_all` hang is rank 0 blocked
in `_recv`/`receive_status`, **setting `engine_op_timeout_s` should turn the hang
into a fast abort** (job fails in minutes, not 16 h). That is a *mitigation*, not a
fix (it aborts rather than shutting down cleanly), but it is worth verifying as the
first step — it both confirms the rank-0-blocked-in-_recv hypothesis and gives an
immediate safety net for recycle runs.

## Suggested fix directions

1. **Make `close` not depend on the errored engine's status.** Close is teardown —
   it doesn't need a clean status. Options: send `close` with `wait_status=False`
   for all sessions (the global session already does), or bound `receive_status`
   during close with a short deadline and proceed/`MPI.Abort` on timeout.
2. **Reset the engine/session to a known-synced state after a handled error** so the
   status/reply stream is balanced before the next op (and before `close`). If
   cause (2) is confirmed, fix the off-by-one in the error reply path.
3. **If cause (1) (survivor stuck in `lammps_command`):** the engine must not be
   left mid-collective — ensure the pre-op symmetric validation also covers the
   `Lost atoms`/`error->all` family on multi-rank sessions, or have `close` force a
   teardown (e.g. `MPI_Abort` the session comm) rather than wait.

## Reproduction

- **Empirical (slow, definitive):** run a recycle config on the 32k benchmark:
  `cd toolkit/profiling && python feature_profile.py --only av+recycle --steps 10
  --run-root <dir>`. It reaches a few steps then freezes at
  `Sending close message to engine at rank N` (rank 0 ~99.9% CPU). Evidence from
  the original observation:
  `benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/07_av+recycle/run_stdout.log`.
- **Targeted test to WRITE (test-first, per project process):** a new MPI test
  (`mpirun -n >= 3`, **`n_sessions=2`**, `engine_use_rank_0=False`): initialize the
  pool, trigger a `minimize`/`minimize_with_results` LAMMPS error on **one**
  session's engine (e.g. a degenerate geometry / NaN coord routed to that session),
  let the manager handle it, then call `manager.close_all()` and assert it returns
  within a wall guard. RED = hang (exit 124) now; GREEN after the fix. Gate it like
  the existing reproducer until fixed (`PYKMC_REPRODUCE_*`), since it hangs.

## Verification (definition of done)

- The new `n_sessions>=2` close-after-error test passes (`close_all()` returns).
- A recycle benchmark config (`av+recycle`) **completes 10 steps and shuts down
  cleanly** (writes `time.txt` with an `Elapsed` line, `status=ok`) — no watchdog
  kill, no `Sending close message ...` freeze.
- `tests/test_kmc.py`, `tests/test_reconstruction.py`,
  `test_engine_error_during_global_minimize_does_not_hang` still pass.

## Key files / lines (branch `bug_recycling_pool_hang`)

| what | location |
|---|---|
| teardown loop (hang site) | `pykmc/enginemanager/lmpi/pool/manager.py:220-227` (`close_all`) |
| session close | `pykmc/enginemanager/lmpi/sessions/mpi_api_sessions.py:145-156` (`close` → `send_message`) |
| status wait | `sessions/mpi_api_sessions.py:113` (`receive_status`), `:87` (`_recv`, the wall-guard poll) |
| wall guard config | `pykmc/config.py` (`control.engine_op_timeout_s`) |
| engine run loop + error reply + status send | `pykmc/enginemanager/lmpi/engines/mpi_api_engine.py:124` (`run_engine_loop`), `:154-161` (error reply), `:171` (`_send_status`), `:58` (`close` handler) |
| existing per-op reproducer (passes; n_sessions=1) | `tests/test_lammps_engine_api_mpi.py::test_engine_error_during_global_minimize_does_not_hang` |
| empirical evidence | `benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/07_av+recycle/run_stdout.log` |
| predecessor handoff (per-op fix, mechanism notes, stack-sampling technique) | `HANDOFF_recycle_pool_hang.md` |

## Out of scope (separate finding, document only)

Full-cell recycle reconstruction is **non-viable on the 32k cell**: AV-off recycle
fails *all* reconstructions (`Lost atoms`) → `All event reconstructions failed` →
early `End of simulation` at step 1. AV-cropped recycle (`av+recycle`) reconstructs
fine (2 fails, reached step 4). This is a recycle physics/scale limitation, not the
teardown bug — note it but don't conflate.
