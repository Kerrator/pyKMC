
# Examples : 

This directory contains example atomistic systems used to test and illustrate the code.  

Each example is provided as a separate subdirectory.

## How to run an example

Most example directories contain an `input.in.template` file.

To run a simulation:
1. Copy the template input file:
```bash
cp input.in.template input.in
``` 
`input.in` files are user-specific and are not tracked. Do not modify input.in.template.

Four older examples ship a tracked `input.in` instead of a template
(`Ni_fcc_4000at_monovacancy+sia`, `Cu_fcc_sia`, `Fe_bcc_6746at_bivacancies`, `NiSi_Surface`): run those directly.

2. Edit ```input.in``` if needed. 
3. Run the simulation. Default template values are intended to be run in parallel with mpi using at least 8 cores
(`n_sessions = 7` plus one driver rank). For example : 
```bash
mpirun -n 8 python -m pykmc -in input.in 
``` 

Examples with an `expected_outputs/` directory ship the output files of a reference run. Event searches are random, so
your run will not match them step for step: compare the energy barriers and the events found, not the trajectory.

## Parameter choices

The inputs follow the recommendations of the [best practices](../docs/best_practices.md) page of the documentation:
8 MPI ranks, a constant prefactor, `basin = False` and `recycle = False` by default, and an active volume only where
the cell is large enough for one. That page also says which settings have been measured on which kind of system;
several examples (Cu, Si, the alloy slab, the NiSi surface) use settings that have not been benchmarked.

## Checking the examples

Two checks cover the examples. The first runs no simulation:

```bash
python -m pytest tests/test_examples_inputs.py
```

It checks that every input loads, that the parser ignores none of its keys, that the files it points to exist, that
this README lists the example, and that `expected_outputs/` was made with the shipped parameters.

The second runs the shipped inputs for 3 KMC steps on 8 MPI ranks and compares each run with
`scripts/examples_baseline.json`:

```bash
python scripts/examples_ratchet.py          # the four single-vacancy examples, about one minute
python scripts/examples_ratchet.py --all    # adds the two self-interstitial examples, about four minutes
```

A run fails if it does not finish, if its first selected barrier is not the baseline one, or if its event searches or
refinements succeed clearly less often than in the baseline. Searches are random, so a failed example is run a second
time before the failure counts. After an intended change, `--update` rewrites the baseline from three runs.

# Nickel systems : 

Nickel systems are the simplest examples provided. 
Nickel has relatively weak elastic effects and is well suited for basic testing and validation 

- `./Ni_fcc_2047at_monovacancy` : 
Simplest example, system with 1 vacancy. 

- `./Ni_fcc_4001at_sia`: 
Isolated SIA, test of intersitial handling and symmetries.

- `./Ni_fcc_4000at_monovacancy+sia` : 
Vacancy-SIA recombination test, run with an active volume (`rmov = 9`, `ract = 16`).


# Iron systems : 

- `./Fe_bcc_1999at_monovacancy` : 
1 vacancy in bcc iron.

- `./Fe_bcc_6746at_bivacancies` : 
4 vacancies in a 2x1NN initial configuration. Well suited to study basins and flickering.
The input ships with `basin = False`: with the default settings the run completes but flickers between a few
states, and with `basin = True` (`[Basin] energy_thr = 0.6`) the basin exploration is currently very slow.

# Copper systems : 

- `./Cu_fcc_2047at_monovacancy` : 
1 vacancy in fcc copper.

- `./Cu_fcc_2049at_sia` : 
Isolated SIA in fcc copper.

- `./Cu_fcc_sia` : 
Vacancy-SIA recombination test. Stronger elastic effects than in Nickel systems.

# Silicon system : 

- `./cSi_511at_monovacancy` : 
1 vacancy in diamond silicon (Stillinger-Weber), with the `diamond/graph` atomic environment.

# Nickel-Chromium alloy system : 

- `./NiCr_fcc_447at_slab_monovacancy` : 
First alloy example: an FCC(100) surface slab with 1 top-surface vacancy, ~15% random
Cr, and an alternating Cr/Ni shell around the vacancy. A hands-on demonstration of
`atom_coloring_mode`: with the default species-resolved `full` the run is clean, while
species-blind `grey` mis-reconstructs catalogued events onto sites of the wrong species
(`Reconstruction fails (reference event N)` in `pykmc.log`). The system is intentionally
constructed to make this happen, so it doubles as a regression check for the coloring
feature. The contrast only appears with basin acceleration on; the template ships with
`basin = False` (the default), so set `basin = True` to try it -- the header of
`input.in.template` walks you through it.

With `basin = True` the example works but is slower: about 70 s per KMC step on 7 MPI ranks
(the template's 10 steps take about 17 min).

# Nickel on silicon surface : 

- `./NiSi_Surface` : 
Ni adatom on a silicon surface with a machine-learned potential (`pair_style mtp/extrapolation`, needs a LAMMPS
build with the MLIP interface), frozen and inactive bottom layers, and a direction bias (`bias = True`).
