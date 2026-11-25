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
The NPZ keeps the recording's numeric arrays unchanged. Search 0
takes 26,931,166 bytes. Searches 1–7 take 0.38–0.39 MB each.

Issue 02's full recordings (611 MB) are kept outside git in the
`2026-11-24-partn-step-recordings` benchmark archive. To reproduce the fixtures,
use its `recordings` directory for `SRC` and an existing output directory for
`DST`. Search 0's recorded commands and provenance name local files by
basename only. Searches 1–7 keep their original metadata.

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

Use these SHA-256 digests to verify the archive inputs and the generated files.

| File | Full recording (SRC) | Committed fixture |
|---|---|---|
| `search0.npz` | `d0764e91f600718c92f07f8f26afa3039ef2df53dfdb9a6a5a0785c0138b57f3` | `79b97b343f130b90956b9151b4c14103831890f89df94dfd42df28c98b4ce68d` |
| `search1.npz` | `13724b5b7d4a3faf44bbcd876cd0449db35e2c327abd5982f7b62cf486749d96` | `f2aeacc21ffd2c91791a9e70ce07a63866142ac882b89e6c7cf0caee4b714c20` |
| `search2.npz` | `8e090f173b082ea1d3bc5026b0365c3e3cb7cb362a22c9c01d81057e9bc9de99` | `47a45f01f67d6b0ce81a44dd39c8823cebb13572c11a8a83130cfae5c182b0f5` |
| `search3.npz` | `e5213bd8d14e18c320643189fc2d1ef9d9b5231e1f68910cc0acd00523a63614` | `419a82e0fc7c4033cc125f206e76a9058c7926f00d6bb29a464d7fb9fb93ae5f` |
| `search4.npz` | `483417191daca3b6449afa16b40ab3f18595ce830e12d5435030286fc50e50e1` | `32cb1f0e6edc187dea126c3a0c1560d53fb7fd4f24001fc3e21afcdaf5a7391c` |
| `search5.npz` | `3145342e674aeeef9242b3bf5083c47cd33e6197082fa292546819a3435bb74f` | `8c8918b4c3203893d23954ec6ffad3770002cf6d46a317e54e08eaa68b295b3e` |
| `search6.npz` | `3a11f470036adf14e3fad9c44c04632b48e7265447a9c9ed99bd75fb0953dbd3` | `abd330e0179351276de3b048c6684ddff90b6e38886e4c6fd1840b6489aaa96c` |
| `search7.npz` | `d48c66bb82ab53bcca56e08eed9b939c802aeabfb67084f2360cf708d7bbfe84` | `e00b2afe88e9c02ce72ed4fef1d48c972b0b09008f2086511b067f0f019884ae` |
