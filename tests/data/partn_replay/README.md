# Recorded pARTn searches for the shared-rules replay test

Twelve event searches recorded through `LammpsEngine.partn_search` on pyKMC `develop` 1f633fc with pARTn v1.1.0
(commit 9b753517). Each file holds what pARTn returned, every other input the search rules read, and the result
or error pyKMC built. A replay test feeds the raw values and inputs to the rules and checks it gets the recorded
result back exactly. Nothing here needs LAMMPS or pARTn.

- Cell: bcc Fe, 8x8x8 conventional cells, a = 2.855312 Å, box 22.842496 Å, 1023 atoms, one vacancy. The lattice
  is shifted by -0.25 Å in x, so four of the vacancy's first neighbours (atoms 986, 987, 994, 995) sit just
  inside the x face, on the far side from the vacancy.
- Potential: `FeP_mm.eam.fs` from LAMMPS `potentials/` (A04), `pair_style eam/fs`.
- Settings: `rcut = 6.5`, `eventsearch.delr_thr = 0.5`, pARTn at pyKMC defaults with `zseed = 0`; active volume
  `rmov = 7.7`, `ract = 11.4` when on.
- Total size: 1.0 MB (12 `.npz` files plus `manifest.json`).

## Cases

| File | Active volume | Rule branch |
|---|---|---|
| `avon_partn_failure`, `avoff_partn_failure` | on, off | pARTn error code -10 gives `EVENT_NOT_FOUND` |
| `avon_rejected`, `avoff_rejected` | on, off | both distances at or above `delr_thr` give `EVENT_MINIMA_NOT_MATCH_POSITIONS` |
| `avon_plain`, `avoff_plain` | on, off | accepted, pARTn's first minimum is the initial state |
| `avon_swapped`, `avoff_swapped` | on, off | accepted, pARTn's second minimum is closer, so the minima and barriers are swapped |
| `*_plain_boundary_bystander`, `*_swapped_boundary_bystander` | on, off | accepted; the hopping atom's displacement from pARTn's first minimum to the saddle crosses the x face, is above `rcut` and is zeroed, so the move atom is a bystander (atom 915 or 987 instead of the hopper 987 or 922) |

`manifest.json` lists each file's central atom, pARTn error, pyKMC result, the raw pick and the atom with the
largest minimum-image displacement.

## Format

Each file is an `np.load(..., allow_pickle=False)` archive. `meta` is a JSON string; floats in it are exact.

- `meta["in"]`: central atom, whether positions and cell were passed, box tilt, periodicity, `is_orthorhombic`.
- `meta["config"]`: `active_volume`, `eventsearch.delr_thr`, `atomicenvironment.rcut`, `partn.delr_thr`,
  `partn.zseed`, `activevolume.ract`, `activevolume.rmov`.
- `meta["err"]`: pARTn's `get_error()` as `[code, message]`. `meta["calls"]`: the pARTn values pyKMC read, in order.
- `meta["raw_scalars"]`: `delr_min1`, `delr_min2` and, when accepted, `etot_sad`, `etot_min1`, `etot_min2`.
  `meta["extra_scalars"]`: the same values when pyKMC did not read them (a rejected search), fetched after the
  search returned.
- `meta["result"]`: pyKMC's `Ok` (central atom, `dE_forward`, `dE_backward`, `move_atom_index`) or `Err` (type,
  message, details, variables).
- Arrays: `in_positions`, `in_types`, `in_cell` (active volume on only; off passes no cell), `box_lo`, `box_hi`,
  `Q`; `atom_map` and `central_lammps_id` (active volume on); `raw_tau_min1`, `raw_tau_min2`, `raw_tau_sad`
  (`extra_` prefix when pyKMC did not read them); `res_min1`, `res_saddle`, `res_min2`, `res_types` when accepted.
  With the active volume on, the `tau` arrays are in active-volume order and the `res` arrays are full-system.

## How these were made

Recorded on 2026-11-16 on an Apple M1 Max (macOS, Open MPI 5.0.7, gfortran 15.2, Accelerate LAPACK), LAMMPS
stable_22Jul2025, by a dump hook kept outside pyKMC, which also recorded 424 more searches. A rebuild of all 436
results from their raw values with the rules as they were on `develop` matched bit for bit. Rerunning gives
different searches (`zseed = 0`), so new recordings mean a new set and one file chosen per case.

In this copy the machine-specific fields of `meta` (`host`, and the pARTn library's directory in `artn_lib`) are
removed; every array and every other field is as recorded. `tests/test_search_rules.py` replays them.
