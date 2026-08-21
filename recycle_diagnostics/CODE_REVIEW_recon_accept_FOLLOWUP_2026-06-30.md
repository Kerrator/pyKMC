# Code review (max-effort) — recycle reconstruction + acceptance, LIVE branch

**Date:** 2026-06-30
**Branch reviewed:** `SK_basin_htst_binary` @ `36dea73` (the default editable install at `/home/kerr/pykmc/pyKMC`)
**Base:** `SK_HTST_backend` (three-dot diff)
**Scope:** the recycle-reconstruction + acceptance contribution as it exists on the **live** branch —
`reconstruction.py`, `utils/geometry.py`, `config.py` (`ReconstructionConfig`), `event_table.py`,
`kmc.py` reconstruction path, `refinement.py`, `result.py`, and the engine
`lammps_operations.py` `_basin_reconstruct_impl`. Scoped diff:
`scratchpad/recon_accept_scoped.diff` (volatile) was regenerated with
`git diff SK_HTST_backend...SK_basin_htst_binary -- <files>`.
**Method:** 10 independent finder angles (×8) → dedup (56→54) → 1-vote 3-state verify (46 survived) →
gap sweep (+3) → ranked to 14 distinct. **All 14 are CONFIRMED.** Companion to the prior
`CODE_REVIEW_recycle_reconstruction_acceptance.md` (the predecessor `bug_recycle_reconstruction_nonviable`
review). Line numbers are authoritative as of `36dea73`; re-verify with `git show` before editing.

> Why this found more than the prior doc: the prior review was scoped to the **pre-recycle** predecessor
> branch, so it never traced the **recycle × purge** interaction — which is where the worst bugs are.

---

## Root-cause clusters (fix these, not the 14 symptoms one-by-one)

### A. Recycle × purge desynchronisation (4 crashes + 1 silent purge) — HIGHEST PRIORITY
The active table is **persistent** across steps (recycled rows carried over), but
`reference_table.remove()` (fired by ANY reconstruction failure, `kmc.py:277`) deletes reference
events — and its backward sibling (`event_table.py:666 all_refs = idx_refs | backward_refs`) — **without
revalidating the carried-over active rows**. `select_recyclable` (`event_recycling.py:78`) keeps rows on
geometry only (movement/distance), never on ref validity. So orphaned rows pointing at purged refs
survive and later blow up. **Fix direction:** on purge, evict/revalidate active rows whose
`num_reference_event ∈ removed set` (or its backward), OR make the downstream lookups tolerate a missing
ref. One fix kills #1, #3, #5; #2/#9 are the same over-broad purge with no transient-vs-defect
discrimination.

- **#1 (HIGH) `kmc.py:688`** — `reference_table[idx_ref==num_ref_event]['event_id'].values[0]` has no
  empty-result guard → `IndexError` (run dies) when a selected recycled row's ref was purged. Fires
  *before* the graceful `active_table.remove` at `:694`; `reconstruction()` has no try/except.
- **#2 (HIGH) `kmc.py:726`** — new len/None fail-fast guards return `RECONSTRUCTION_MINIMIZE_FAILED`,
  routed through the same purge → permanently deletes a **globally-valid** reference event over a
  per-row `len(neighbors)!=len(saddle_positions)` mismatch (routine when IRA/PSR maps a ref
  neighbourhood of size `N_ref` onto a site with `N_active` rcut neighbours; see `refinement.py:118-125`).
- **#3 (HIGH) `info_simulation.py:211`** — `mapping_event_id[ref]` (built only from the current
  reference table) `KeyError`s on an orphaned recycled row.
- **#5 (HIGH) `info_simulation.py:214`** — `mapping_energy[backward]` `KeyError`s because `remove()`
  drops the backward sibling too.
- **#9 (MED) `kmc.py:689`** — transient `MINIMIZE_FAILED` / `EVENT_NOT_CONTAINED` flow into the same
  unconditional permanent purge; a non-deterministic engine hiccup permanently deletes a correct event.
  No transient-vs-real discrimination.

### B. MPI safety on the basin engine path
- **#4 (HIGH) `lammps_operations.py:~1132`** — `basin_reconstruct` feeds pushed/saddle geometry to
  `set_positions`+`minimize` on the multi-rank pool **without** `_require_finite_positions` (wired only
  into `minimize_with_results` at `:509`). A NaN/inf coord → LAMMPS `error->one` on the owning rank only
  → survivors trapped in liblammps collectives → multi-hour pool hang (the `HANDOFF_recycle_pool_hang.md`
  incident). Mirror the host guard before the collective entry points (~1118/1131/1181).
- **#11 (MED) `lammps_operations.py:~1221`** — the rank-0-only min2 validation block
  (`wrap_positions`/`per_atom_displacement`/`reconstruction_matches` at ~1194-1197) lacks the
  try/except-to-payload that min1/PSR have. A rank-0-only raise → `raised=True` on rank 0 → `finally`
  calls `_ensure_full_system(force=True)` → collective LAMMPS rebuild on rank 0 alone while others
  short-circuit → collective mismatch/hang + mislabeled error. Structurally present; latent today
  (reachability needs a rank-0-only raise) but one config/edit away. Fix: wrap the block like its
  siblings; add an `MPI.Abort` boundary in `run.py`.

### C. Dedup neighbour-ordering (same class as the landed neighbour-order fix, untouched paths)
- **#6 (MED) `event_table.py:998`** — `remove_duplicates` part-2 re-derives `get_neighbors('rcut')`
  ordering instead of the stored `neighbors` column → recycled rows compare non-corresponding atoms →
  duplicate kept (rate double-counted) or distinct event wrongly removed.
- **#8 (MED) `event_table.py:973`** — `remove_duplicates` part-1 compares `saddle_positions`
  element-wise via `compute_delr` assuming identical ordering; same hazard. (backfill/refinement honour
  the stored ordering; dedup does not — the asymmetry is the tell.)

### D. Serial ≡ engine acceptance parity (stated invariant, violated)
- **#12 (MED) `lammps_operations.py:1132`** — engine basin reconstruction applies the AV outer-sphere
  freeze (`_minimize_freeze_outer_sphere`) during min1/min2; serial `Reconstruction.reconstruct` does an
  unconstrained `global_minimize` → divergent geometries → opposite accept/reject under `active_volume=True`.
- **#14 (MED) `lammps_operations.py:1117`** — same divergence for `basin.style=='global'`
  (skip-reconstruction short-circuit): engine freezes, serial (`basin.py:426`) does not.
- **#13 (MED) `reconstruction.py:114`** — serial wraps with `pbc=True` and pushes with default
  `pbc=[T,T,T]`; engine threads the runtime `pbc`. On a slab (`pbc=[T,T,F]`, see `basin.py:744`) a
  near-z-boundary mover wraps/pushes differently → the `shell_tolerance`/`matching_thr` gate flips →
  serial vs MPI reach opposite verdicts. (Bulk fully-periodic is unaffected.)

### E. Acceptance gate (carried over from prior doc, re-confirmed)
- **#7 (MED) `reconstruction.py:77`** — containment guard measures mover radius from `supposed_min1`
  only; an outward event (inside rcut at min1, near/over rcut at min2) passes containment but its min2
  relaxation is truncated by the frozen far field → `INVALID_MIN2` → valid event purged. Check max
  extent over min1+saddle+min2.
- **#10 (MED) `utils/geometry.py:225`** — `event_movers` caps the tight-checked set to top `n_movers`
  (default 3); a genuine 4th+ participant in a collective event is bounded only by `shell_tolerance`
  (1.0 Å) and can mis-land ~0.7 Å while accepted (also advances the KMC clock with the wrong barrier).
  Mitigable without code change (raise `n_movers` / lower `shell_tolerance`) but the default leaves the
  (0.1, 1.0] Å band open. Masked for FCC-Ni NN hops; real for sub-Å-distinct minima.

---

## Suggested fix grouping (focused commits on `SK_basin_htst_binary`)
1. **recycle/purge desync** — #1, #2, #3, #5, #9 (highest impact; one root cause)
2. **MPI safety** — #4 (finite guard), #11 (min2 guard + `MPI.Abort`)
3. **dedup neighbour-ordering** — #6, #8
4. **serial≡engine parity** — #12, #13, #14
5. **acceptance** — #7 (containment over min2), #10 (adaptive n_movers / material-aware shell)

Validate each with the repo gates (see `verify-pykmc-change` skill): `ruff`/`mypy` are local-only;
recycle/basin MPI suites need `mpirun -n 8+`. The 32k recycle benchmark
(`benchmarks/Ni_fcc_32000at_4vac+4sia/`, `toolkit/profiling/feature_profile.py --only recycle,av+recycle`)
is the integration check — note the recycle×purge crashes (#1/#3/#5) need a **reconstruction failure +
later selection of an orphaned recycled row**, which the clean 16/16 validation runs may not have
exercised (they had 0 fails post-neighbour-fix), so add a targeted unit test that purges a ref then
selects/reconstructs a row still pointing at it.
