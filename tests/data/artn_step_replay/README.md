# pARTn v1.1.0 step replay fixtures

Search 0 is a complete successful search. The default LAMMPS-free tests replay
all 453 numbered pARTn steps in its 604 recorded evaluations, from the initial
push through the saddle and both minima. Searches 1–7 keep their first 15
evaluations, through the first Lanczos probe. Set `PYKMC_ARTN_RECORDINGS` to the
directory containing all eight full recordings to run the eight-search gate.

Issue 02 recorded the eight Fig. 2 searches with pARTn v1.1.0 (`9b753517`),
pyKMC `745527c` and LAMMPS 22Jul2025 on one rank. The Fe vacancy cell used
`FeP_mm.eam.fs` with `pair_style eam/fs`. Its active volume had 994 atoms in
tag order, 440 movable and 554 in the buffer (`rmov = 10.7`, `ract = 14.2`).
Each file's `meta_json` records the Table I settings, including every pARTn
`set` call.

Each file stores `x`, `f`, `v_in`, `scalars_in`, `scalars_out`, `partn_step`,
`repeat` and `kind` for each evaluation. It also stores the initial push
(in bohr), first Lanczos vector, movable mask, cell, periodicity and metadata.
Search 0 includes the eigenvector history and its index, the new-chance draw
array, the saved starting, saddle and minimum positions, and the saddle mode.
The NPZ stores the recording's original numeric arrays losslessly. Search 0
takes 26,931,166 bytes. Searches 1–7 take 0.38–0.39 MB each.

To reproduce the fixtures, give the generator the full recordings directory
and an existing output directory. Search 0's recorded commands and provenance
identify local files by basename. The prefixes retain their original metadata.

```sh
python -I tests/data/artn_step_replay/make_replay_set.py SRC DST
```

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
