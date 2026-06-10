# HTST prefactor profiling — eskm vs Python-FD

Benchmarks the two ways pyKMC builds the mass-weighted Hessians that feed the
Vineyard ν₀ prefactor, so we can pick the fastest for the production path (HTST
runs only when an event is *added to the reference table*, before refinement):

| Mode | How the Hessian is built | Per Hessian |
|---|---|---|
| **eskm** | LAMMPS PHONON `dynamical_matrix <group> eskm <dx> file` | 1 LAMMPS command + 1 file read |
| **fd** | `pykmc.htst.hessian.mass_weighted_partial_hessian` | `2·3F` `get_forces` calls |

Both feed the identical numpy `eigh` / `vineyard_prefactor` math. The harness
runs both on **one** serial LAMMPS engine per event with identical `free_indices`
and `dx`, so the comparison is apples-to-apples.

## Run

```bash
cd /Users/stephenkerr/kmc/pyKMC-htst-work
source /Users/stephenkerr/kmc/pykmc_env/bin/activate

python benchmarks/htst_prefactor/run_profile.py --system ni100 \
    --modes fd,eskm --radii 3.0,4.0,5.0,6.0,7.5 --repeats 5 \
    --out benchmarks/htst_prefactor/results/ni100.csv

python benchmarks/htst_prefactor/run_profile.py --system ni4000 \
    --modes fd,eskm --radii 3.0,4.0,5.0,6.0,7.5 --repeats 5 \
    --potential tests/data/Ni_v6_2.0_LKBeland2016.eam \
    --out benchmarks/htst_prefactor/results/ni4000.csv

python benchmarks/htst_prefactor/compare.py benchmarks/htst_prefactor/results/*.csv
```

## Systems & potentials

- **ni100** — `tests/data/htst_ni100_surface_hop.npz` (130-atom Ni(100) surface hop, 1 event).
  Canonical ν₀ ≈ 12.6 THz @ `free_radius=4.0` with `NiAlH_jea.eam`. That potential is **not**
  committed; drop it in `basin_testing/NiAlH_jea.eam` to reproduce the absolute canon (the run
  falls back to the tracked Ni_v6 EAM otherwise — the eskm-vs-fd comparison is potential-independent).
- **ni4000** — the 2 stored events of
  `tests/data/reference_table_Ni_fcc_4000at_monovacancy+sia.pickle` (86/105-atom neighbour
  clusters), with the tracked `tests/data/Ni_v6_2.0_LKBeland2016.eam`.

## Reading the report

- `speedup fd/eskm` > 1 means eskm is faster in serial wall time.
- `rel_err` is the cross-mode ν₀ disagreement; it should be < 1% (flagged otherwise).
- `rt fd` / `rt eskm` are **engine round-trips per event** (eskm = 3; fd = `2·3·n_free` per
  Hessian × 3 Hessians = `18·n_free`). This is the
  number that dominates once the prefactor runs through a remote/global engine (the Option A
  backend), where each round-trip is a messenger send/recv — eskm's 3-vs-hundreds advantage
  usually decides the production choice independent of local compute speed.

`compare.py` prints, per system, the serial-fastest mode at the production `free_radius=6.0` plus
the round-trip ratio.

## Notes

- At `free_radius ≥ 6` the finite 86/105-atom `ni4000` clusters saturate (boundary atoms get
  pulled into the free region); the eskm-vs-fd *comparison* stays valid but the absolute ν₀ there
  is less physically meaningful.
- The eskm timer writes/reads `/tmp/pykmc_dynmat_profile.*.dat` and removes it per Hessian.
