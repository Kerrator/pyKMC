## `Control` Section (mandatory)

<details><summary>Section Overview</summary>
  Core simulation control parameters.
</details>

- **`initial_config`** : `str`, mandatory
  <details><summary>Description</summary>
  File path for the initial atomic structure. This file should be parseable by `ase.io.read()` and contain atom types, positions, simulation cell, and periodic boundary conditions.
  </details>
- **`trajectory_output`** : `str`, default = `'./trajkmc.xyz'`
  <details><summary>Description</summary>
  File path where the simulation trajectory will be saved. The file should be writable by `ase.io.write` using `append=True `
  </details>
- **`reference_table_output`** : `str`, default = `'./reference_table.pickle'`
  <details><summary>Description</summary>
  File path where the reference table will be store in pickle format.
  </details>
- **`visited_environments_output`** : `str`, default = `'./visited_environments.pickle'`
  <details><summary>Description</summary>
  File path where the list of atomic environments that have been explored will be sore in pickle format.
  </details>
- **`reference_table`** : `str`, optional
  <details><summary>Description</summary>
  Path to a reference table generated from a previous simulation. With the `htst`/`rpa` rate styles a table written before the current schema (or without per-calculation producing provenance) is loaded with its event geometry kept but every accepted prefactor demoted to status `legacy` with the `k0` fallback; one `WARNING` names the file, the schema versions, the counts and the recovery path (run with a prefactor service so refined sites get site estimates, or regenerate the catalogue). Re-saving never restores those estimates. See the *Reference estimate persistence* section of the physical descriptors page.
  </details>
- **`visited_environments`** : `str`, optional
  <details><summary>Description</summary>
  Path to a list of visited environment generated from a previous simulation.
  </details>
- **`restart_file`** : `str`, optional
  <details><summary>Description</summary>
  Restart file `restart_<step>.npz` written at the end of a previous run. It holds only the last step number and the simulated time in seconds (`last_step`, `last_time`); the atomic configuration is taken from `initial_config` (the last saved snapshot) and the catalogue from `reference_table` and `visited_environments`. The step counter continues from `last_step + 1` and the saved time is carried over once; the saved configuration is not re-minimized, its energy is evaluated as saved. The random-number streams are not saved: with `seed` set the generators are re-seeded from that value, so a restarted run is reproducible on its own but does not continue the interrupted run's sequence of draws.
  </details>
- **`reconstruction`** : `bool`, default = `True`
  <details><summary>Description</summary>
  If at each KMC step we reconstruct generic events.
   NOT WORKING
  </details>
- **`n_steps`** : `int`, mandatory
  <details><summary>Description</summary>
  Total number of simulation steps to run.
  </details>
- **`engine`** : `Literal['lammps']`, mandatory
  <details><summary>Description</summary>
  Which E/F Engine to use. Note : Only lammps is implemented.
  </details>
- **`n_sessions`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of Sessions
  </details>
- **`group_size`** : `int`, default = `-1`
  <details><summary>Description</summary>
  Number of MPI worker ranks in the group communicator (replaces old global mode). -1 means all worker ranks.
  </details>
- **`engine_use_rank_0`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Deprecated : If use mpi rank 0 or not.
  </details>
- **`verbosity`** : `int`, default = `1`
  <details><summary>Description</summary>
  Controls the level of detail in the simulation output.
  </details>
- **`refine_thr`** : `float`, default = `0.9999`
  <details><summary>Description</summary>
  Refinement coverage target, as a fraction of the total rate. constant style: reference events whose barrier lies within 0.1 eV of the fastest event below this fraction of the estimated total rate are refined. htst/rpa styles: cumulative pre-dispatch rate coverage over the candidate ledger of retained active channels and PSR-valid reference applications, grouped by reference event and ranked by rate (groups tied at the cut are included); 1 refines every group with a positive rate, a zero total refines nothing.
  </details>
- **`basin`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Basin mode
  </details>
- **`active_volume`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Incorporate AV's into simulations, recommended for large systems
  </details>
- **`recycle`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Recycle non-perturbed events from the previous KMC step instead of re-searching them. Requires an [EventRecycling] section.
  </details>
- **`bias`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Enable event selection bias. Requires a [Bias] section.
  </details>
- **`max_physical_time`** : `float`, optional
  <details><summary>Description</summary>
  Maximum physical (simulated) time in ps. If set, the simulation stops once this value is reached. Defaults to None (no time limit).
  </details>
- **`seed`** : `int`, optional
  <details><summary>Description</summary>
  Reproducibility knob. When set, the Python `random` module and NumPy's global random generator are seeded once at KMC construction, so the choice of the atoms searched per new environment (`central_atoms_research`), the rejection-free (BKL) event and time draws and the basin exit draws repeat between runs; the new-environment list is sorted, so `PYTHONHASHSEED` is not needed. It does not seed the saddle-point search: pARTn's own stream is `[pARTn] zseed`, and the saddle instance a search returns can still differ between runs. On a restart the generators are seeded afresh from the same value; the streams of the interrupted run are not restored. Must be between 0 and 2**32 - 1, inclusive. Defaults to None (unseeded).
  </details>

---

## `Atomicenvironment` Section (mandatory)

<details><summary>Section Overview</summary>
  Atomic environments parameters.
</details>

- **`style`** : `Literal['cna', 'graph', 'cna/graph', 'diamond/graph', 'coordination', 'coordination/graph']`, mandatory
  <details><summary>Description</summary>
  Method used to characterize and assign an ID to an atom's local atomic environment. 'coordination' classifies atoms based on nearest-neighbor count against a threshold. 'coordination/graph' first filters by coordination, then computes graph IDs for non-crystal atoms.
  </details>
- **`rnei`** : `float`, mandatory
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors.
  </details>
- **`rcut`** : `float`, optional
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the local atomic environment.
  </details>
- **`neighbors_add`** : `int`, default = `0`
  <details><summary>Description</summary>
  When `style` is 'cna/graph', specifies the N-th shell of neighbors whose graph IDs should also be computed.
  </details>
- **`coordination_threshold`** : `int`, optional
  <details><summary>Description</summary>
  When style is 'coordination' or 'coordination/graph', atoms with fewer neighbors (within rnei) than this value are classified as 'noncrystal'. Atoms with this many or more neighbors are classified as 'crystal'. Required when style is 'coordination' or 'coordination/graph'.
  </details>
- **`atom_coloring_mode`** : `Literal['grey', 'full']`, default = `'full'`
  <details><summary>Description</summary>
  Controls whether element types are used in environment matching. Defaults to 'full' (species-resolved). 'grey': all atoms treated identically (grey alloy approximation). 'full': element types used in graph hashing, PSR matching, and symmetry detection.
  </details>

---

## `Eventsearch` Section (mandatory)

<details><summary>Section Overview</summary>
  Event search parameters.
</details>

- **`style`** : `Literal['partn']`, mandatory
  <details><summary>Description</summary>
  Method used to find events.
  </details>
- **`nsearch`** : `int`, mandatory
  <details><summary>Description</summary>
  Number of event searches to perform per unique atomic environment.
  </details>
- **`emax_event`** : `float`, default = `5.0`
  <details><summary>Description</summary>
  Maximum energy barrier (in eV) for an event to be added to the reference table.
  </details>
- **`emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum energy forward and backward barrier (in eV) for an event to be added to the reference table.
  </details>
- **`backward_emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Lower bound (in eV) of the backward barrier used together with `energy_asymmetry`.
  </details>
- **`energy_asymmetry`** : `int`, default = `5`
  <details><summary>Description</summary>
  Prevent highly asymmetric events from being added to the reference table: an event whose forward barrier exceeds `energy_asymmetry` x `backward_emin_event` is rejected unless its backward barrier is also above `backward_emin_event`.
  </details>
- **`refined_minimum_delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Refinement is accepted only if the central atom moves less than this distance between the current position and the refined minimum.
  </details>
- **`refined_energy_thr`** : `float`, default = `0.05`
  <details><summary>Description</summary>
  Maximumallowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy.
  </details>
- **`delr_thr`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  delr threshold between one minima and the intial configuration to consider the event valid.
  </details>

---

## `Psr` Section (mandatory)

<details><summary>Section Overview</summary>
  Point set registration parameters.
</details>

- **`style`** : `Literal['ira']`, mandatory
  <details><summary>Description</summary>
  Method used for the point set registration (shape matching) between reference events and atomic environment of an atom having the same atomic environement ID of the event. This method is also used to find atomic environment symmetries.
  </details>
- **`matching_score_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Maximum value of the matching score of the algorithm used.
  </details>

---

## `Rateconstant` Section (mandatory)

<details><summary>Section Overview</summary>
  Rate constant computation parameters.
  
  The rate of an event is ``k = k_prefactor * exp(-dE / (kb * T))`` with every
  rate and prefactor in ps^-1. ``style`` selects the prefactor backend in
  ``pykmc.rate_constant.backends``:
  
  - ``constant``: ``k_prefactor = k0`` for every event.
  - ``htst``: harmonic transition state theory; ``k_prefactor`` is the
    per-event Vineyard frequency ``nu0`` (computed in Hz by the HTST kernel
    and converted to ps^-1 once), falling back to ``k0`` when no estimate is
    available for that event.
  - ``rpa``: registered alias of ``htst``; bare Vineyard, no recrossing
    correction is implemented.
  
  The HTST-only fields (``free_radius``, ``free_region_center``, ``fd_step``,
  ``zone_radius``, ``interaction_range``, ``nu0_min_THz``, ``nu0_max_THz``,
  ``premin``) are validated for every style and ignored by ``constant``.
</details>

- **`style`** : `Literal['constant', 'htst', 'rpa']`, mandatory
  <details><summary>Description</summary>
  Method used to compute the prefactor of the rate constant: 'constant' (fixed `k0`), 'htst' (per-event harmonic TST / Vineyard prefactor with `k0` as the fallback) or 'rpa' (alias of 'htst': bare Vineyard, no recrossing correction is implemented).
  </details>
- **`k0`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Prefactor in ps^-1 (1.0 = 1 THz). When `style` is **'constant'** it is used directly as the pre-exponential factor ($k_0$); for **'htst'** and **'rpa'** it is the per-event fallback when no Vineyard prefactor is available. For 'htst'/'rpa' a value above 1e4 ps^-1 is rejected because it was almost certainly entered in Hz.
  $$ k = k_{0} \exp\left(-\frac{\Delta E}{k_{b}T}\right) $$
  </details>
- **`T`** : `float`, default = `300.0`
  <details><summary>Description</summary>
  Temperature (in Kelvin) used for computing rate constants.
  </details>
- **`free_radius`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  HTST: radius (Angstrom) around the moving atom selecting the free (movable) atoms of the partial Hessian; every other atom is frozen. User-frozen atoms and, under active volume, the shell atoms beyond `rmov` that fall inside this radius are excluded from the free set as well (the search held them, so they carry residual forces); the log reports the free-set size and the excluded shell count per request.
  </details>
- **`free_region_center`** : `Literal['saddle', 'min1']`, default = `'saddle'`
  <details><summary>Description</summary>
  HTST: geometry in which the free (movable) region of the partial Hessians is selected around the moving atom. 'saddle' (default) centres the one common free region on the atom's saddle-point position, which is symmetric between the two minima by construction. 'min1' centres it on the atom's initial position (the original model); on the symmetric SW-Si vacancy hop this gives forward and backward prefactors that differ by 20 percent (23.6 vs 19.6 THz) purely through the frozen-boundary choice, so 'min1' exists only for comparison with older results.
  </details>
- **`fd_step`** : `float`, default = `0.01`
  <details><summary>Description</summary>
  HTST: central finite-difference displacement (Angstrom) used to build the Hessian.
  </details>
- **`force_tol`** : `float`, default = `0.005`
  <details><summary>Description</summary>
  HTST: maximum raw force norm (eV/Angstrom) on any atom of the common vibrational set at each stationary geometry, after premin and cropping. Fixed-atom reaction forces are excluded. Larger or nonfinite forces reject the native prefactor calculation.
  </details>
- **`zone_radius`** : `float`, optional
  <details><summary>Description</summary>
  HTST: optional radius (Angstrom) around the moving atom used to crop the scratch system on which the Hessians are computed. None (default) uses the full system.
  </details>
- **`interaction_range`** : `float`, default = `13.0`
  <details><summary>Description</summary>
  HTST: interaction range (Angstrom) of the force model, the largest distance over which a fixed atom's position enters the partial Hessian of a free atom: the pair-style cutoff for a pair potential, up to twice the cutoff for an embedded-atom, moment-tensor or three-body potential. Used only when recycling active events between steps: a stored site prefactor depends on every atom within `free_radius + interaction_range` of the moving atom (within `zone_radius` when the calculation was zone-cropped) and any motion of one of them invalidates the recycled row. It never enters a computed prefactor. The default is twice the 6.5 Angstrom cutoff of the Ni EAM potential shipped with the tests, the largest of the shipped potentials; a value matched to the potential keeps more recycled rows valid.
  </details>
- **`nu0_min_THz`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  HTST: lower bound (THz) of the acceptance window for the Vineyard prefactor nu0. The window is applied by the HTST kernel: an estimate below it is rejected and the event falls back to `k0`. Must be < `nu0_max_THz`.
  </details>
- **`nu0_max_THz`** : `float`, default = `100.0`
  <details><summary>Description</summary>
  HTST: upper bound (THz) of the acceptance window for the Vineyard prefactor nu0. The window is applied by the HTST kernel: an estimate above it is rejected and the event falls back to `k0`. Must be > `nu0_min_THz`.
  </details>
- **`premin`** : `bool`, default = `False`
  <details><summary>Description</summary>
  HTST: relax the surroundings of the event with the event core frozen before computing the Hessians.
  </details>

---

## `Lammps` Section (optional)

<details><summary>Section Overview</summary>
  Lammps parameters.
</details>

- **`pair_style`** : `str`, mandatory
  <details><summary>Description</summary>
  Lammps pair_style command.
  </details>
- **`pair_coeff`** : `str`, mandatory
  <details><summary>Description</summary>
  Lammps pair_coeff command.
  </details>
- **`min_style`** : `str`, default = `'cg'`
  <details><summary>Description</summary>
  Lammps min_style command.
  </details>
- **`minimize`** : `str`, default = `'1.0e-6 1.0e-8 1000 1000'`
  <details><summary>Description</summary>
  Lammps minimize command
  </details>
- **`frz_min`** : `str`, default = `'1.0e-6 1.0e-8 10 10'`
  <details><summary>Description</summary>
  Lammps minimize command with frozen core
  </details>
- **`verbosity`** : `int`, optional
  <details><summary>Description</summary>
  LAMMPS log verbosity. None inherits control.verbosity. 0 disables log file.
  </details>

---

## `Partn` Section (optional)

<details><summary>Section Overview</summary>
  pARTn parameters.
</details>

- **`verbosity`** : `int`, default = `2`
  <details><summary>Description</summary>
  pARTn verbosity
  </details>
- **`delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Threshold at which an atom is considered to have moved. This threshold affects the npart parameter in the artn.out output.
  </details>
- **`zseed`** : `int`, default = `0`
  <details><summary>Description</summary>
  The value of zseed is used to seed the random number generator. If the value equals 0, a new radom seed gets geenrated. The exact zseed value of each research is written in file zseed.dat, which can be useful for debugging, or re-running exact same pARTn runs.
  </details>
- **`push_mode`** : `Literal['list', 'rad']`, default = `'rad'`
  <details><summary>Description</summary>
  Determines how the initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) from the central atom.
  </details>
- **`push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which all atoms receive an initial displacement.
  </details>
- **`push_step_size`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- **`ninit`** : `int`, default = `2`
  <details><summary>Description</summary>
  Specify the minimal number of pushes with the initial push vector.
  </details>
- **`lanczos_min_size`** : `int`, default = `10`
  <details><summary>Description</summary>
  Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`lanczos_max_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Maximum number of Lanczos iterations.
  </details>
- **`lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- **`eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- **`eigen_step_size`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  The limit to the maximum size of the displacement with eigenvector.
  </details>
- **`nsmooth`** : `int`, default = `3`
  <details><summary>Description</summary>
  Number of smoothing steps from initial displacement to eigenvector.
  </details>
- **`neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of pushes along the eignevector before starting a perpendicular relax.
  </details>
- **`alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- **`nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Number of times a research is allowed to cross a convex region (without counting the starting convex region).
  </details>
- **`nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Control the perpendicular relaxation.
  </details>
- **`nperp_limitation`** : `list[int]`, default = `[4, 8, 12, 16, -1]`
  <details><summary>Description</summary>
  Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- **`forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- **`convergence_property`** : `Literal['maxval', 'norm']`, default = `'maxval'`
  <details><summary>Description</summary>
  Specify how pARTn tests convergence of the forces (transmitted as pARTn's converge_property for both event searches and refinements). 'maxval': the convergence will be tested by MAXVAL( ABS( force ) ); 'norm': the convergence will be tested by NORM2( force ).
  </details>
- **`nevalf_max`** : `int`, default = `9999`
  <details><summary>Description</summary>
  Stop an artn search before end when the number of force evaluations by the force engine is greater to nevalf_max
  </details>
- **`push_over`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Factor that scales the displacement vector used to push the system from the saddle point towards a local energy minimum. 
  $$ \text{displacement} = \text{push_factor} \times v_0 \times \text{eigen_step_size} \times \text{push_over} \times 0.8 $$
  </details>
- **`dmax`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.
  </details>
- **`r_nevalf_max`** : `int`, default = `300`
  <details><summary>Description</summary>
  Stop an artn refinement before end when the number of force evaluations by the force engine is greater to nevalf_max.
  </details>
- **`r_max_attempts`** : `int`, default = `5`
  <details><summary>Description</summary>
  When adjusting the saddle energy and positions, in some rare cases partn has trouble finding the saddle point and goes back to the minium.In that case, we do another attempt with a different seed.
  </details>
- **`r_delr_sad_thr`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  Acceptance threshold (in Angstrom) for a refined saddle point. A refinement run starts from the expected saddle position, and artn delr_sad measures how far the converged saddle has moved from that starting configuration. If delr_sad is strictly below this threshold (delr_sad < r_delr_sad_thr), the refined saddle stayed close to the expected saddle and is accepted; otherwise (e.g. the search fell back to the minimum), a new attempt is made, up to r_max_attempts.
  </details>
- **`r_push_mode`** : `Literal['list', 'rad']`, default = `'list'`
  <details><summary>Description</summary>
  Determines how the initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) from the central atom.
  </details>
- **`r_push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which all atoms receive an initial displacement.
  </details>
- **`r_push_step_size`** : `float`, default = `0.0001`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- **`r_ninit`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Specify the minimal number of pushes with the initial push vector.
  </details>
- **`r_lanczos_min_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Refinement: Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`r_lanczos_max_size`** : `int`, default = `50`
  <details><summary>Description</summary>
  Refinement: Maximum number of Lanczos iterations.
  </details>
- **`r_lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Refinement: Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`r_lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- **`r_eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Refinement: threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- **`r_eigen_step_size`** : `float`, default = `0.005`
  <details><summary>Description</summary>
  Refinement: The limit to the maximum size of the displacement with eigenvector.
  </details>
- **`r_nsmooth`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of smoothing steps from initial displacement to eigenvector.
  </details>
- **`r_neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Refinement: Number of pushes along the eignevector before starting a perpendicular relax.
  </details>
- **`r_alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  Refinement: This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- **`r_nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of times a research is allowed to cross a convex region (without counting the starting convex region).
  </details>
- **`r_nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Refinement: Control the perpendicular relaxation.
  </details>
- **`r_nperp_limitation`** : `list[int]`, default = `[100]`
  <details><summary>Description</summary>
  Refinement: Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- **`r_forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Refinement: The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- **`r_dmax`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Refinement: dmax parameter used in fix ID all artn dmax value lammps command. should be higher than push_step_size.
  </details>

---

## `Ira` Section (optional)

<details><summary>Section Overview</summary>
  IRA parameters.
</details>

- **`kmax_factor`** : `float`, default = `1.8`
  <details><summary>Description</summary>
  Multiplicative factor that needs to be larger than 1.0. Larger value increases the search space of the rotations.
  </details>
- **`sym_thr`** : `float`, default = `0.01`
  <details><summary>Description</summary>
  Threshold in terms of the Hausdorff distance. If an operation returns a distance value beyond sym_thr, then SOFI will not consider that operation as a symmetry operation.
  </details>

---

## `Basin` Section (optional)

<details><summary>Section Overview</summary>
  Basin parameters
</details>

- **`style`** : `Literal['global', 'global/reconstruction']`, default = `'global'`
  <details><summary>Description</summary>
  Basin style used.
  </details>
- **`energy_thr`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Energy threshold
  </details>

---

## `Reconstruction` Section (mandatory)

<details><summary>Section Overview</summary>
  Reconstruction parameters.
</details>

- **`push_fraction`** : `float`, default = `0.15`
  <details><summary>Description</summary>
  Fraction used to push the system from the saddle point toward each minimum during reconstruction.
  </details>
- **`n_movers`** : `int`, default = `3`
  <details><summary>Description</summary>
  Size of the core of an event, its most-displaced atoms (min1->min2). Every atom whose event displacement exceeds psr.matching_score_thr is tight-checked against that threshold, however many there are; only when no atom exceeds it (a sub-threshold event) are the top n_movers atoms tight-checked instead. Peripheral atoms that did not move during the event do not veto the match. The top-n_movers core is also the set measured by the rcut containment guard, so a collective event's small elastic ripple near the shell edge does not count against containment.
  </details>
- **`containment_margin`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Radius margin (Angstrom): the event movers must sit within (atomicenvironment.rcut - containment_margin) of the central atom at min1, the saddle, AND min2, else the event is judged too large for the rcut neighbourhood and reconstruction is rejected as not contained. Must be > 0 and < atomicenvironment.rcut.
  </details>
- **`shell_tolerance`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Looser whole-rcut-shell acceptance bound (Angstrom). On top of the tight n_movers check, EVERY atom in the rcut shell must land within shell_tolerance of its expected min1/min2 position. This catches a peripheral (non-mover) atom that relaxed into a distinct site (a large displacement) while tolerating the small wiggle of atoms that merely settled around the event; the movers-only check alone would accept such a wrong overall state. Set well above the expected peripheral relaxation (~tenths of an Angstrom) but below a nearest-neighbour site change.
  </details>

---

## `Activevolume` Section (optional)

<details><summary>Section Overview</summary>
  Active Volume Parameters
</details>

- **`ract`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  Radius of entire active volume, spherical
  </details>
- **`rmov`** : `float`, default = `4.0`
  <details><summary>Description</summary>
  Radius of movable atoms in active volume, spherical
  </details>
- **`AV_debug`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Debug flag for active volume size checks
  </details>

---

## `Eventrecycling` Section (optional)

<details><summary>Section Overview</summary>
  Event recycling parameters. Required when control.recycle = True.
</details>

- **`style`** : `Literal['displacement']`, mandatory
  <details><summary>Description</summary>
  Method used to decide which events can be recycled. 'displacement' = central atom moved less than movement_thr AND is farther than distance_thr from the executed event.
  </details>
- **`movement_thr`** : `float`, default = `0.02`
  <details><summary>Description</summary>
  Angstroms. Central atoms whose displacement from pre- to post-execution is below this are considered 'unmoved'.
  </details>
- **`distance_thr`** : `float`, default = `10.0`
  <details><summary>Description</summary>
  Angstroms. Candidate events whose central atom is farther than this (PBC-aware minimum-image) from the executed event's central atom pass the distance check.
  </details>

---

## `Inactive_atoms` Section (optional)

<details><summary>Section Overview</summary>
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/region.py``.
</details>

- **`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- **`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- **`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- **`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- **`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- **`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- **`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- **`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- **`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- **`types`** : `list[str]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- **`indices`** : `list[int]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

## `Frozen_atoms` Section (optional)

<details><summary>Section Overview</summary>
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/region.py``.
</details>

- **`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- **`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- **`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- **`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- **`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- **`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- **`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- **`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- **`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- **`types`** : `list[str]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- **`indices`** : `list[int]`, default = `PydanticUndefined`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

## `Bias` Section (optional)

<details><summary>Section Overview</summary>
  Event selection bias parameters.
</details>

- **`style`** : `Literal['direction', 'point', 'topo']`, mandatory
  <details><summary>Description</summary>
  Bias style: 'direction' (DirectionBias), 'point' (PointBias), or 'topo' (TopoBias).
  </details>
- **`mode`** : `Literal['filter', 'boost']`, default = `'filter'`
  <details><summary>Description</summary>
  Selection mode. 'filter': rejection-loop removes non-accepted events. 'boost': multiplies desired event rates by a dynamic factor so they fire with probability bias_weight, without blocking other events.
  </details>
- **`bias_weight`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  Target probability in (0, 1) that a desired event is selected at each step. Only used in boost mode.
  </details>
- **`pass_unlisted`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Whether atoms not in atom_indices pass through the bias predicate unchanged. False (default): non-listed atoms are rejected/undesired. True: non-listed atoms always pass; only valid in filter mode.
  </details>
- **`direction`** : `list[float]`, optional
  <details><summary>Description</summary>
  Direction vector [x, y, z] for 'direction' bias.
  </details>
- **`target_point`** : `list[float]`, optional
  <details><summary>Description</summary>
  Target point [x, y, z] for 'point' bias.
  </details>
- **`atom_indices`** : `list[int]`, optional
  <details><summary>Description</summary>
  Global atom indices to bias. None means all atoms.
  </details>
- **`threshold`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum projection onto the bias direction for acceptance.
  </details>
- **`topo_source`** : `str`, optional
  <details><summary>Description</summary>
  Source topology ID for 'topo' bias (e.g. vacancy).
  </details>
- **`topo_target`** : `str`, optional
  <details><summary>Description</summary>
  Target topology ID for 'topo' bias (e.g. interstitial).
  </details>

---
