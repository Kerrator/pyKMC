# Handoff: recycle event reconstruction is non-viable at 32k scale

**Status:** open issue (recycle physics/robustness), **NOT a hang.** The two recycle
*hangs* are already fixed — this is the remaining recycle limitation that those
fixes exposed.
**Severity:** medium — recycle runs no longer crash/hang, but they cannot sustain a
simulation on the large benchmark, so recycle yields no usable speed-up there.
**Found:** 2026-06-19/20, re-running the recycle benchmark configs once both recycle
hangs were fixed.

---

## What this is NOT (already fixed — don't re-do)

1. **Per-op pool hang** (a `Lost atoms` `minimize` error stranding the engine pool) —
   FIXED, see `HANDOFF_recycle_pool_hang.md`.
2. **`close_all()` teardown hang** (multi-session teardown in global mode) — FIXED,
   see `HANDOFF_close_all_teardown_hang.md`.

With both fixed, recycle runs no longer hang. **This issue is the next layer down:**
the recycled-event **reconstructions themselves fail** at 32k scale, so the run runs
out of valid events and ends early — cleanly, but uselessly.

## Symptom

Every recycle config on `Ni_fcc_32000at_4vac+4sia` (10-step request) terminates
after **1–2 steps**, having dropped a huge number of reconstructions:

| config | reconstruction-fails | steps reached | how it ends |
|---|---|---|---|
| `recycle` (AV-off) | **140** | 1 | `All event reconstructions failed` → `End of simulation` |
| `av+recycle` (AV-on) | **96** | 2 | same |

AV-cropping helps (fewer fails, one more step) but does not resolve it. So recycle
cannot currently be benchmarked or used for a real run on this system — the moment a
step has *no* event whose reconstruction succeeds, KMC has no move to make and stops.

## Mechanism

`pykmc/reconstruction.py::reconstruct` (lines ~54–95): to reuse a recycled event it
pushes the saddle's neighbours toward the target minimum by
`config.reconstruction.push_fraction` (default **0.15**), then minimizes:

```python
saddle_toward_min1_pos = push_towards(saddle_positions[neighbors],
                                      supposed_min1_positions,
                                      fraction=self.config.reconstruction.push_fraction,
                                      cell=cell)
tmp_positions[neighbors] = saddle_toward_min1_pos
min1_pos, _ = self.manager.global_minimize_with_results(self.config,
                                                        positions=tmp_positions,
                                                        types=self.types)   # full-cell minimize
```

At 32k full-cell scale the pushed geometry is unstable → LAMMPS `Lost atoms` → the
per-op fix drops that one reconstruction (`Err(RECONSTRUCTION_MINIMIZE_FAILED)`) and
the run continues. But when *every* candidate event's reconstruction fails that way,
the step has no viable event → `All event reconstructions failed` → clean
`End of simulation`.

## Investigation directions

1. **The pushed geometry.** `push_fraction = 0.15` may overshoot when a generic event
   is mapped onto a differently-shaped local environment. Dump a failing
   `tmp_positions` (atom overlaps? out of box?); try smaller / adaptive fractions.
2. **Full-cell vs local minimize.** `reconstruct` minimizes the WHOLE 32k cell from a
   perturbed saddle — numerically fragile and slow. A **local / AV-cropped** minimize
   (relax only the event neighbourhood, freeze the far field) is almost certainly the
   fix — it is *why* `av+recycle` already does better (96 vs 140 fails). Consider
   making reconstruction always minimize a cropped subsystem.
3. **IRA mapping quality.** Recycle reuses a generic event by mapping it onto a new
   site (IRA). If the orientation/registration is slightly off, the pushed geometry is
   unphysical → loses atoms. Correlate reconstruction failures with the IRA match
   score for that mapping.
4. **Intended scale.** Check upstream recycle usage/tests — recycle may have been
   validated only on small/lattice systems; confirm whether full-cell reconstruction
   was ever expected to work at 32k.

## Reproduction

Needs BOTH recycle hang-fixes applied so the run *ends cleanly* instead of hanging:
the per-op fix (committed `8f05963` on `bug_recycling_pool_hang`) AND the `close_all`
teardown fix (on `bug_recycling_pool_hang`; see `HANDOFF_close_all_teardown_hang.md`).

```bash
source pykmc_env/bin/activate
cd toolkit/profiling
python feature_profile.py --only av+recycle --steps 10 \
    --run-root <dir>
```

Watch `<dir>/07_av+recycle/run_stdout.log`: dozens of
`Reconstruction fails (reference event N) : ... Lost atoms`, then
`ERROR : All event reconstructions failed` / `:=> End of simulation` after 1–2 steps.
Captured evidence:
`benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/{04_recycle,07_av+recycle}/run_stdout.log`.

## A targeted test to write (test-first)

A unit/integration test that recycles **one** known generic event onto a perturbed
copy of its own environment and asserts the reconstruction **succeeds** (returns
`Ok`, `delr <= psr.matching_score_thr`), parametrized over increasing system size /
cell, to pin exactly where full-cell reconstruction breaks down — and to lock in the
fix (local-minimize / smaller push).

## Verification (definition of done)

A recycle config completes **≥ 5 steps** on the 32k benchmark with the **majority of
reconstructions succeeding** (no `All event reconstructions failed`), shutting down
cleanly. Ideally recycle then delivers its intended speed-up vs fresh ARTn search.

## Key files

| what | location |
|---|---|
| reconstruction (push + full-cell minimize) | `pykmc/reconstruction.py` (`reconstruct`, ~lines 54–95) |
| push_fraction knob | `pykmc/config.py:662` (`ReconstructionConfig.push_fraction`, default 0.15) |
| recycle feature | `pykmc/event_recycling.py`; `pykmc/config.py:786` (`EventRecyclingConfig`, `style="displacement"`) |
| graceful per-op drop (already fixed) | `pykmc/reconstruction.py` → `Err(RECONSTRUCTION_MINIMIZE_FAILED)` |
| evidence | `benchmarks/Ni_fcc_32000at_4vac+4sia/profiling_runs_postfix/{04_recycle,07_av+recycle}/run_stdout.log` |
| predecessor handoffs (the FIXED hangs) | `HANDOFF_recycle_pool_hang.md`, `HANDOFF_close_all_teardown_hang.md` |
