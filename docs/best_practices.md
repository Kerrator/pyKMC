# Best Practices

This page collects parameter choices that have been measured, and says on which systems.
It complements the [parameter reference](parameters.md), which lists every key and its default.

The numbers come from benchmark campaigns run in November 2026 on bulk fcc Ni (EAM, 2000 to 4000 atoms) and bulk
bcc Fe (EAM, 2000 to 16,000 atoms) with vacancies and self-interstitials, on 8 MPI ranks of one workstation, plus
100-step runs of the shipped [examples](https://github.com/hugomoison/pyKMC/tree/develop/examples).
**A setting measured on one kind of system is not evidence for another.**
Nothing here was measured on alloys, surfaces, grain boundaries, silicon or copper; for those, treat the values in
the examples as starting points.

## Summary

| Setting | Recommendation | Measured on |
|---|---|---|
| MPI size | `n_sessions = 7`, `mpirun -n 8` | every run |
| `[RateConstant]` | `style = constant`, `k0 = 10` | every run |
| `[AtomicEnvironment]` | fcc Ni: `rnei = 3.0`, `rcut = 6.5`; bcc Fe: `rnei = 3.46`, `rcut = 6.5` | Ni, Fe |
| `[EventSearch] nsearch` | 10 to 20 for isolated point defects | Ni, Fe |
| `[Control] active_volume` | `True` from about 2000 atoms, with localized defects | Ni, Fe |
| `[ActiveVolume]` | fcc Ni: `rmov = 9`, `ract = 16`; bcc Fe vacancy: `rmov = 7.7`, `ract = 14` or 9 / 16 | Ni, Fe |
| `[pARTn]` | defaults for fcc Ni; the defaults fail about 70 % of searches in bcc Fe | Ni, Fe |
| `[Control] basin` | `False` unless the system flickers and you have checked the basin run | Fe and NiCr examples |
| `[Control] recycle` | `False` by default; see [below](#event-recycling) | Ni, Fe |

## Parallel size

All timings on this page are for `n_sessions = 7` on 8 MPI ranks (one driver rank plus one rank per session).
Event searches and refinements are distributed one per session, so the wall time of a step falls with the number of
sessions until there are fewer tasks than sessions.
The number of sessions was not varied in the benchmarks: 7 is the measured configuration, not a tuned optimum.

```bash
mpirun -n 8 python -m pykmc -in input.in
```

For timings, pin the run to cores of one kind and keep the rest of the machine idle: on a hybrid CPU, load on the
other cores slowed pinned runs by 10 to 24 %.

## What a step costs

For an isolated defect in a bulk crystal, event searches happen on the first step (and on the first visit to each
new environment).
After that a step is the refinement of the known events: about 0.05 s per refined event in fcc Ni on 7 sessions, so
the step time grows linearly with the number of defects (0.80, 1.40, 2.37 and 4.63 s for 1, 2, 4 and 8 Ni vacancies
in 4000 atoms).
Search cost returns when defects interact or when a defect keeps meeting new environments: a vacancy plus a
self-interstitial in bcc Fe at 800 K found 181 new environments in 100 steps, and searching was most of the run.

## Event search

`nsearch` is the number of searches launched for each new environment.
With 10 to 20 searches the catalogue of an isolated vacancy was complete in both fcc Ni (12 hops) and bcc Fe (8 hops),
even where most searches failed.
No convergence study against `nsearch` was made for interstitials or interacting defects.

The default `[pARTn]` settings work for vacancies and self-interstitials in fcc Ni.
In bcc Fe they do not: 69 to 73 % of the searches stopped with `EIGENVALUE LOST`, for the vacancy, for the
split-vacancy state and for the dumbbell.
A failed search costs about half a successful one, so the catalogue was still correct, at roughly three times the
search cost.
When the pARTn output shows this message often, raise `nnewchance` (the number of times a search may cross a convex
region) before raising `nsearch`.

A key placed in the wrong section is ignored without a warning.
`nnewchance`, `r_nnewchance` and the other pARTn keys belong in `[pARTn]`, not in `[EventSearch]`.

## Active volume

See [Active Volumes](active_volumes.md) for how the feature works.

Use it when the cell is large compared with the region a defect disturbs.
In bcc Fe a search on the whole 1999-atom cell cost 2.2 to 2.4 times a search in a 7.7 / 14 Å active volume, and with
the volume fixed a cell 4.6 times larger cost 1.5 times more per step.

Choosing the radii:

- `rmov` must be at least `rcut` of `[AtomicEnvironment]`, otherwise refinements fail.
- `ract - rmov` must be at least the cutoff of the interatomic potential, so that movable atoms feel correct forces.
- `ract` must be smaller than half of the shortest periodic cell length.
- Measured values: `rmov = 9`, `ract = 16` (fcc Ni and bcc Fe, `rcut = 6.5`); `rmov = 7.7`, `ract = 14` (bcc Fe
  vacancy); `rmov = 11.4`, `ract = 17` (bcc Fe dumbbell, one search batch).

The radii were not swept. The energy barriers obtained with them agree with full-cell runs to within 5 meV for the
bcc Fe vacancy.
The code defaults (`rmov = 4`, `ract = 6`) are smaller than any `rcut` in normal use: always set both keys.

A small cell cannot hold an active volume: the 2047-atom Ni example has a half cell length of 14 Å, below
`ract = 16`.

## Event recycling

See [Event Recycling](recycling.md) for how the feature works.

Recycling carries refined events from one step to the next, so it saves refinements, never searches.
It does nothing for a single defect (every event is within `distance_thr` of the executed one).
With several vacancies far apart and an active volume, the step time fell to 0.58, 0.37 and 0.21 of the
non-recycled value for 2, 4 and 8 vacancies in fcc Ni, and to 0.71, 0.45 and 0.23 in bcc Fe, with the same catalogue
and the same total rate.

The examples ship with `recycle = False`.
The gain above was measured only for bulk vacancies that stay farther apart than `distance_thr`, and only with the
active volume on.
Without an active volume, and when two defects approach each other, recycled runs lost events in the benchmarks.
`distance_thr` was not calibrated; do not lower it below the default.

## Basin acceleration

See [Basins](basins.md).
Keep `basin = False` unless the run flickers between a few states, and check a basin run before trusting it: with
`basin = True` the Fe four-vacancy example needed many hours for its first steps, and in the NiCr slab example most
basin exits failed, so little time was gained.

## Checking a run

A run that ends with exit code 0 is not necessarily healthy.
Check at least:

- the number of steps written to `pykmc.out` (a run whose first searches all fail stops at step 1 with code 0);
- `Reconstruction fails` lines in `pykmc.log`;
- `EIGENVALUE LOST` in the pARTn output, as a share of the searches;
- `k_tot` against what the defects should give (for isolated vacancies, the number of hops per vacancy times the
  hop rate): a value that jumps by a factor of 1.5 or 2 means duplicated events;
- that the energy barriers of the executed events are the ones you expect for the defect.
