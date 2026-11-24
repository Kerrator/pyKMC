# pARTn v1.1.0 step recordings, cut to the initial-push window (search 0: the first Lanczos block)

The eight Fig. 2 event searches of the Python ARTn plan, recorded at every force evaluation of pARTn v1.1.0 (commit
9b753517) under pyKMC 745527c, LAMMPS 22Jul2025, one rank (issue 02). Searches 1–7 are cut to their first 15
evaluations: the start, two initial pushes with three perpendicular FIRE steps each, and the first Lanczos probe.
Search 0 is cut to its first 37 evaluations, through its first Lanczos block (11 probes) and the move back, so the
Lanczos block tests run on it in CI. `tests/test_artn_replay.py` replays them through `pykmc.artn.artn_search`
with `tests/artn_replay.py`. Nothing here needs LAMMPS or pARTn.

- Cell: the gate's Fe vacancy cell (bcc Fe, `FeP_mm.eam.fs`, `pair_style eam/fs`), active volume on with
  `rmov = 10.7`, `ract = 14.2`: 994 atoms in tag order, 440 movable, 554 buffer.
- Settings: the Table I `[pARTn]` block (`push_dist_thr 4.0`, `push_step_size 0.3`, `eigval_thr -0.5`,
  `nnewchance 3`, the rest pyKMC defaults); every `set` call pyKMC made is in each file's `meta_json`.
- Arrays per evaluation: `x`, `f`, `v_in`, `scalars_in`, `scalars_out`, `partn_step`, `repeat`, `kind`. Once per
  search: `draw_initial_push` (bohr), `draw_first_lanczos_vector`, `movable`, `cell`, `pbc`, `meta_json`. Search 0
  also keeps `eigenvec_index` per evaluation and the 12 `eigenvec` versions it points to. Layout and units as in
  the full recordings.
- Made by `make_replay_set.py` from the full recordings (611 MB, kept outside git); home folders in `meta_json`
  are written as `~`. Total size: 3.9 MB (search 0 1.35 MB, the others 0.38–0.39 MB each).

| File | Central atom (full cell) | Seed |
|---|---:|---:|
| `search0.npz` | 1333 | 494828123 |
| `search1.npz` | 1333 | 712798592 |
| `search2.npz` | 1531 | 2540808125 |
| `search3.npz` | 1513 | 348537163 |
| `search4.npz` | 1513 | 875863341 |
| `search5.npz` | 1311 | 1934173456 |
| `search6.npz` | 1333 | 900186024 |
| `search7.npz` | 1311 | 2134761458 |

To replay the full searches, set `PYKMC_ARTN_RECORDINGS` to the folder of full recordings.
