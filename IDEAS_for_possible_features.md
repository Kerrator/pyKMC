# Ideas for possible features

Speculative / parked ideas. Each entry: the idea, what it would change vs. current behaviour,
and the known blockers (so we don't re-derive them). Not a commitment to build.

---

## Basin recycling — cache the basin shape on exit for near-instant re-entry

**Status:** parked (2026-06-30) — likely not viable; see blockers. Captured for the record.

**Idea.** When KMC leaves a basin, record the "shape" of the basin (its explored states +
connectivity table + absorbing/exit states) so that if the trajectory later re-enters the *same*
basin, we skip the full BFS exploration and get an almost-instant basin "search" by replaying the
cached structure. As a safety step, **map the chosen exit state** to confirm we are not leaving a
low-energy barrier behind us (i.e. an exit that would immediately re-trap into another basin).

Two scopes:
- **Narrow (re-entry, same site):** reuse the cache only when we re-enter the identical trapped
  configuration at the same atoms. Key = entry-state fingerprint (`basins/fingerprinting.py`
  already provides `compute_fingerprint`).
- **Broad (generic super-event):** canonicalize the basin to relative coords + a topology key so it
  can be replayed at any symmetry-/chemically-equivalent site, the way generic *events* are reused
  across sites today. Far bigger payoff, far harder.

**What it would change.** Today there is **no basin cache** — re-entering the "same" basin re-runs
the entire BFS from scratch every time. The `BasinsGenericEvents` object is a per-step throwaway
(`kmc.py:300`), GC'd at step end; intra-basin state fingerprints reset on every entry
(`basin.py:177`). The connectivity table *is* already pickled each step
(`basin_connectivity_<step>.pickle`, `kmc.py:343`) but is **never read back** — so a cache substrate
half-exists.

**Exit-safety sub-idea (worth it independently of caching).** There is currently **no** check that a
basin exit avoids leaving a low barrier behind. `config.basin.energy_thr` only *classifies*
transient-vs-absorbing states at entry/exploration (`detection.py:37`, `exploration.py:89`); the FPTA
exit draw (`selection.py:212-280`) applies no barrier condition (its only filter is
failed-reconstruction `excluded_states`). Mapping the exit state's onward barriers before committing
would close that hole — and is a prerequisite for any caching version.

### Why this probably won't work out (blockers)

1. **No basin identity to key on.** Two basin entries being "the same basin" is not well-defined.
   A defect's local entry configuration does not uniquely determine the whole multi-atom super-event;
   the surrounding environment can differ. Narrow keying on the entry-state fingerprint is cheap but
   likely matches rarely (how often does a trajectory re-enter a *byte-identical* trap?).
2. **Stochastic exit can't be cached.** FPTA draws the exit time and exit state with **unseeded**
   `np.random.random()` (`selection.py:175`, `:277`). KMC correctness requires drawing the exit
   *fresh* each visit from the proper time/probability distribution — you may cache the *structure*
   (states/rates/generator) but you must **re-draw** the exit, so the speed-up is only the BFS, not
   the selection.
3. **Staleness / invalidation.** Between visits the surrounding atoms may have moved, changing
   barriers and connectivity. A cached basin needs a validity condition (environment unchanged within
   some radius), which erodes the narrow hit rate further.
4. **Broad reuse needs whole-basin reconstruction.** Replaying a basin at a different site means
   IRA-style reconstruction of an *entire* basin's geometry (every state) onto a new neighbourhood,
   under the existing species-blind grey-alloy matching — much harder than reconstructing a single
   event, and a new correctness surface.
5. **Payoff is uncertain.** The current basin bottleneck is dedup + reconstruct cost *within* one
   BFS, which the fingerprint pre-filter + MPI wavefront already attack. Caching only helps on
   genuine re-entry, which may be rare in practice.

### If revisited, do this first (cheapest valuable slice)
- Build the **exit-safety map** alone (blocker-free, useful on its own): after FPTA picks an exit
  state, check its outgoing barriers against `energy_thr` and reject/re-draw if it leaves a sub-
  threshold escape behind. This is the one piece that stands on its own.
- Only then attempt narrow re-entry caching, measuring the actual re-entry hit rate on a real run
  before investing in the broad generic-super-event version.

**Relevant code:** `basins/basin.py` (`_initialize` :170-185, exit hook), `basins/exploration.py`,
`basins/selection.py` (FPTA exit draw), `basins/detection.py` (`energy_thr` classification),
`basins/fingerprinting.py` (entry key), `kmc.py:296-350` (basin hook, connectivity pickle, recycler
detach).
