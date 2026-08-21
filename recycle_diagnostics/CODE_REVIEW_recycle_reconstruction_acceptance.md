# Code review — recycle reconstruction + acceptance-policy branch

**Date:** 2026-06-27
**Branch:** `bug_recycle_reconstruction_nonviable`
**Tip reviewed:** `d193cce` (on `kerrator`)
**Diff range:** `8f05963..d193cce` (full branch contribution: code + tests + diagnostics + docs)
**Method:** multi-angle local review (`/code-review`, xhigh) — correctness + MPI + data-flow +
cleanup/conventions finders, then each high-severity finding re-verified by hand against the
worktree source (line numbers are branch-dependent and were confirmed against `d193cce`).

This document records the review outcome so it survives the session. It is a companion to
`HANDOFF_recycle_reconstruction_nonviable.md` / `HANDOFF_recycle_reconstruction_findings.md` and to
the durable memory note `recycle-reconstruction-neighbor-order-fix.md`.

---

## 1. What holds up (verified correct)

The branch's structural claims were checked against the actual worktree and are genuine:

- **Signature change is clean.** Both real `reconstruct()` callers — `pykmc/kmc.py:751` and
  `pykmc/basins/basin.py:437` — match the new `(min1, min2, saddle, cell, neighbors=None,
  central_atom=None)` signature; `central_atom` and `neighbors` are in scope at each call. The
  dropped `delr_thr` positional is fully gone; the remaining `delr_thr` references in the tree are
  unrelated config fields (`eventsearch.delr_thr`, `partn.delr_thr`).
- **No KeyError on the engine rejection/exception paths.** In `_basin_reconstruct_impl` the
  `proceed["movers"]` / `proceed2["movers"]` reads are short-circuited by the `ok is False` /
  `step == "min_global"` / `None` guards (lammps_operations.py:1114, 1120, 1178) **before** any
  `["movers"]` access. The containment-rejection dict, the exception payloads, and the PSR-miss
  dicts all carry `ok: False` and return early.
- **Geometry helpers exist and align.** `per_atom_displacement`, `minimum_image_distance`,
  `event_movers`, `reconstruction_matches` are all defined in `pykmc/utils/geometry.py` and imported
  correctly in both reconstruction paths. The `movers` row indices correctly index the `discrepancy`
  array (same `neighbors` ordering on both sides). `compute_delr` is still legitimately used
  (event_table.py:978/1030, kmc.py import).
- **Neighbour-ordering data flow is consistent.** `refinement.py:124` stores `neighbors` and then
  crops `saddle_positions` by the same index set; `EventRefinementOutput.neighbors` flows through
  `build_event_series` into the row; `_reconstruction_active_event` and `backfill_refined_prefactors`
  both consume the stored column. Production producers (refinement, basin tmp-event, recycler row
  subset) all populate it.

The six prior-review fixes (whole-shell `shell_tolerance` bound; serial/engine unified policy;
`n_movers` `gt=0`; `central_atom` threading; corrected backfill docstring + repaired htst fixtures;
dead `delr_thr` removal) are present and real.

**Refuted candidate:** the "untyped signature + `_type_`/`_description_` docstring" lint concern on
`reconstruct` is *pre-existing* stub debt, not introduced by this diff (the diff only changed which
params appear on the signature line; the new `event_movers`/`reconstruction_matches` functions are
fully annotated and documented). Not a finding.

---

## 2. Findings (ranked)

| # | Sev | File:line | Finding |
|---|-----|-----------|---------|
| 1 | High | `pykmc/utils/geometry.py:225` | `n_movers=3` cap can accept a wrong final state when an event has >3 real movers |
| 2 | High | `pykmc/enginemanager/lmpi/lammps_operations.py:1192` | Engine min2 block unguarded → rank-0-only raise → `_ensure_full_system(force=…)` divergence → MPI deadlock |
| 3 | Med-High | `pykmc/config.py:671` | `containment_margin` has no `gt=0` and no cross-check vs `rcut`; `margin ≥ rcut` rejects every event |
| 4 | Med | `pykmc/reconstruction.py:113` vs `…/lammps_operations.py:1154` | Serial hardcodes `pbc=True`; engine threads runtime `pbc` → serial≠engine for slabs; acceptance metric is pbc-blind |
| 5 | Med-Low | `pykmc/kmc.py:275` | `EVENT_NOT_CONTAINED` + fail-fast integrity errors flow into the global reference-event/topology purge |
| 6 | Low | `pykmc/reconstruction.py:75` | Containment guard checks only min1 mover radii, never min2 |
| 7 | Low | `pykmc/reconstruction.py:75` / `…/lammps_operations.py:1080` | Containment guard fails open when `central_atom` absent from `neighbors` |
| 8 | Low | `pykmc/utils/geometry.py:223` | Degenerate/empty `event_displacement` → `argmax`/`max` `ValueError` (crash vs graceful `Err`) |
| 9 | Low | `pykmc/event_table.py:903` | Backfill lacks the None-guard its sibling consumer (`_reconstruction_active_event`) got |
| 10 | Cleanup | `pykmc/reconstruction.py:71` + `…/lammps_operations.py:1074` | Containment-guard block (~12 lines) duplicated verbatim across the two paths |
| 11 | Cleanup | `pykmc/event_table.py:884` | `backfill_refined_prefactors` keeps an unused `neighbors_list` param "for signature stability" |
| 12 | Tuning | `pykmc/config.py:675` | `shell_tolerance` default 1.0 Å is absolute, tuned only on Ni (NN≈2.49 Å) |

### Detail

**1 — `n_movers=3` cap can accept a wrong final state.** `event_movers` keeps only the top
`n_movers` (default 3); `reconstruction_matches` checks those tightly (`matching_score_thr` ≈ 0.1 Å)
but every other atom only against the loose `shell_tolerance` (1.0 Å). A genuine 4th+ participant
that lands 0.1–1.0 Å off — relaxing into a *nearby but distinct* site — passes both checks, so the
reconstruction is **accepted onto the wrong state**.

> Failure scenario: `event_disp=[1.5,1.4,1.3,1.2,1.1]` (5 real movers) → movers `[0,1,2]`;
> reconstruction discrepancy `[0,0,0,0.6,0]` (4th real mover idx 3 lands 0.6 Å off). `ok=True`
> because `delr_movers=0.0 ≤ 0.1` and `delr_shell=0.6 ≤ 1.0`.

Especially relevant to the multi-vacancy Ni domain, where concerted/correlated events with >3 movers
are plausible. **Fix:** gate *all* atoms with `event_displacement > matching_thr` at the tight
threshold (drop the fixed top-3 cap, or raise it and tighten the shell bound).

**2 — Engine min2 block unguarded → MPI deadlock.** The PSR (lammps_operations.py:1005) and min1
(1147) rank-0 blocks are wrapped in `try/except` that convert any raise into an `ok: False` payload,
with comments warning that raising past the collective hangs the other ranks. The **min2 validation
block (1192–1222) has no such guard**, yet the diff added new statements to it
(`proceed2["movers"]`, `per_atom_displacement`, `reconstruction_matches`). If rank 0 raises there it
escapes to `basin_reconstruct`'s `except` (951–975), setting `raised=True` **only on rank 0**; the
`finally` then calls `_ensure_full_system(force=raised)`. Rank 0 (`force=True`) unconditionally runs
collective `lmp.command("clear")` + rebuild while the other engine ranks (`force=False`)
short-circuit on the atom-count check and issue no LAMMPS commands → collective divergence → hang
(no `MPI.Abort` boundary in `run.py`). Trigger is rare; impact is a full-job deadlock.
**Fix:** wrap the min2 block in the same try/except-to-payload pattern as its two siblings.

**3 — `containment_margin` unvalidated.** Its two sibling new fields have `gt=0`; this one has
nothing. `rcut_limit = rcut - containment_margin` (reconstruction.py:80): `margin ≥ rcut` makes the
limit ≤ 0, so **every** event with a nonzero-radius mover is rejected
`RECONSTRUCTION_EVENT_NOT_CONTAINED` → run silently purges/stalls everything; a negative margin
silently disables the guard. **Fix:** add `gt=0` plus a `model_validator` requiring
`containment_margin < atomicenvironment.rcut`.

**4 — Serial/engine PBC divergence.** Serial `reconstruct` hardcodes `wrap_positions(…, pbc=True)`
and `push_towards` with `pbc=None` (full PBC); the engine threads the runtime `pbc`. For a slab
(`pbc=[T,T,F]`) the two paths wrap/push differently and can accept/reject the same event differently
— breaking the serial≡engine parity this branch set out to guarantee. Separately,
`per_atom_displacement` is always full-MIC (pbc-blind), so a real across-free-surface displacement is
under-reported in both paths. Benign for bulk (the benchmark); real for slabs. **Fix:** thread `pbc`
through the serial path too, and pass `pbc` into the displacement metric.

**5 — Over-broad purge routing.** Any non-OK result from `_reconstruction_active_event` makes
`reconstruction()` append to `err_reference`/`err_ae`, and the main loop (kmc.py:275–279) then
removes the generic event from `reference_table` and drops its topology from
`visited_environments`. `EVENT_NOT_CONTAINED` joining this is arguably fine (containment is a
site-independent property of the generic event), but the new fail-fast data-integrity guards
(None/length-mismatch `neighbors` → `RECONSTRUCTION_MINIMIZE_FAILED`, kmc.py:705–735) get conflated
with a physical reconstruction failure: a transient row-construction bug silently and permanently
shrinks the catalogue instead of surfacing the corruption. **Fix:** give the integrity guards a
distinct, non-purging disposition (log/abort, not catalogue deletion).

**6 — Containment guard checks only min1.** `max_mover_r` is computed from
`supposed_min1_positions` only. A mover starting just inside `rcut - margin` but moving *outward*
past `rcut` at the final state passes the guard, and the frozen far field then truncates the event
during the min2 minimize — the truncation the guard exists to prevent. **Fix:** check both
endpoints.

**7 — Containment guard fails open.** `if len(central_rows) > 0:` silently skips the entire guard
when the central index isn't in the stored ordering — i.e. exactly the mis-ordered/truncated rows
this fix targets bypass the only geometric sanity check. Normally self is in the rcut list so this
holds; a corrupted/permuted column defeats it without error. **Fix:** treat an absent central row as
an error, not a skip.

**8 — Degenerate/empty shell crashes.** On a zero-length `event_displacement`, `np.argmax` raises
`ValueError`; downstream `max(… for m in movers)` likewise on empty `movers`. It is not a
`RuntimeError`, so the `try/except` around the minimize does not catch it — the step crashes rather
than declining the reconstruction. Low reachability (rcut shell is always ≥1 atom). **Fix:** guard
the empty case and return an `Err`.

**9 — Backfill None-guard asymmetry.** `backfill_refined_prefactors` does
`np.asarray(row["neighbors"], dtype=int)` over `refined=="T"` rows with no None/missing protection,
while `_reconstruction_active_event` gained exactly that guard. A refined row with `neighbors=None`
raises `TypeError` and aborts the run (backfill runs in the main loop, no try/except). Production
paths populate it today, so this is a robustness asymmetry rather than a live bug — but it is the
asymmetric half of the same fix. **Fix:** mirror the None-guard (skip + log).

**10 — Duplicated containment block.** `event_movers`/`reconstruction_matches` were correctly
extracted to `geometry.py`, but the harder-to-keep-in-sync `central_rows`/`max_mover_r`/`rcut_limit`
math was left inline in *both* reconstruction.py and lammps_operations.py — the most likely spot for
host/engine drift, contradicting the branch's own "accept/reject identically" goal. **Fix:** extract
an `event_contained(…)` helper alongside the other two.

**11 — Dead `neighbors_list` param.** `backfill_refined_prefactors` keeps a required positional that
is never read, forcing the one caller (kmc.py:269) to pass a `NeighborsList` for nothing; it was
re-typed to bare `object` (loosening mypy). It is a one-line caller. **Fix:** drop the param.

**12 — `shell_tolerance` not material-portable.** Default 1.0 Å is absolute and tuned on Ni
(NN≈2.49 Å). For a small-NN material 1.0 Å can exceed half the NN spacing (a wrong-site landing
slips under it); for large-NN it may reject valid peripheral relaxation. Benign now (re-validation
showed 0 shell rejections). **Fix:** consider scaling with `rnei`/NN rather than a fixed Å.

---

## 3. Recommendation

- **Fix before relying on the branch:** **#1** (multi-mover acceptance hole — directly relevant to
  multi-vacancy runs) and **#2** (engine min2 deadlock asymmetry). **#3** is a cheap, high-value
  config guard worth doing at the same time.
- **Deliberate design calls:** **#4** (slab parity) and **#5** (purge routing) — decide intent
  rather than patch blindly.
- **Hardening / cleanup:** **#6–#12** as follow-ups.

None of these regress the validated happy path: the core neighbour-ordering fix and the serial≡engine
parity (for periodic cells) are sound. The benchmark used `strategy=serial`, so the engine-path
findings (#2, parts of #4/#10) are exercised only by the MPI tests, not the 32k benchmark.
