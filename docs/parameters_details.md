<a id="section-control"></a>

## `Control` Section (mandatory)

<details><summary>Section Overview</summary>
  Core simulation control parameters.
</details>

- <a id="control-initial_config"></a>**`initial_config`** : `str`, mandatory
  <details><summary>Description</summary>
  File path for the initial atomic structure. This file should be parseable by `ase.io.read()` and contain atom types, positions, simulation cell, and periodic boundary conditions.
  </details>
- <a id="control-trajectory_output"></a>**`trajectory_output`** : `str`, default = `'./trajkmc.xyz'`
  <details><summary>Description</summary>
  Trajectory path written by `ase.io.write(..., append=True)`.
  </details>
- <a id="control-reference_table_output"></a>**`reference_table_output`** : `str`, default = `'./reference_table.pickle'`
  <details><summary>Description</summary>
  Reserved for a configurable reference-table output path; currently unused. The table is always written to `reference_table.pickle`.
  </details>
- <a id="control-visited_environments_output"></a>**`visited_environments_output`** : `str`, default = `'./visited_environments.pickle'`
  <details><summary>Description</summary>
  File path where the set of atomic environments that have been explored will be stored in pickle format.
  </details>
- <a id="control-reference_table"></a>**`reference_table`** : `str`, optional
  <details><summary>Description</summary>
  Path to a reference table generated from a previous simulation.
  </details>
- <a id="control-visited_environments"></a>**`visited_environments`** : `str`, optional
  <details><summary>Description</summary>
  Path to a set of visited environments generated from a previous simulation.
  </details>
- <a id="control-restart_file"></a>**`restart_file`** : `str`, optional
  <details><summary>Description</summary>
  Path to a pyKMC restart metadata file.
  </details>
- <a id="control-reconstruction"></a>**`reconstruction`** : `bool`, default = `True`
  <details><summary>Description</summary>
  Reserved for optional reconstruction; currently unused by the main KMC loop. Selected events are always reconstructed.
  </details>
- <a id="control-n_steps"></a>**`n_steps`** : `int`, mandatory
  <details><summary>Description</summary>
  Total number of simulation steps to run.
  </details>
- <a id="control-engine"></a>**`engine`** : `Literal['lammps']`, mandatory
  <details><summary>Description</summary>
  Energy/force engine. Currently only `lammps` is implemented.
  </details>
- <a id="control-n_sessions"></a>**`n_sessions`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of parallel LAMMPS engine sessions.
  </details>
- <a id="control-engine_use_rank_0"></a>**`engine_use_rank_0`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Deprecated: whether MPI rank 0 also hosts an engine session. When False, rank 0 is reserved for orchestration.
  </details>
- <a id="control-verbosity"></a>**`verbosity`** : `int`, default = `1`
  <details><summary>Description</summary>
  Controls the level of detail in the simulation output.
  </details>
- <a id="control-refine_thr"></a>**`refine_thr`** : `float`, default = `0.9999`
  <details><summary>Description</summary>
  Event constributing to this percent of ktot are refined.
  </details>
- <a id="control-basin"></a>**`basin`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Basin mode
  </details>
- <a id="control-active_volume"></a>**`active_volume`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Enable Active Volume mode; recommended for large single-element systems.
  </details>
- <a id="control-recycle"></a>**`recycle`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Recycle non-perturbed events from the previous KMC step instead of re-searching them. Requires an [EventRecycling] section.
  </details>
- <a id="control-bias"></a>**`bias`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Enable event selection bias. Requires a [Bias] section.
  </details>

---

<a id="section-atomicenvironment"></a>

## `AtomicEnvironment` Section (mandatory)

<details><summary>Section Overview</summary>
  Parameters defining the local atomic environments and the method used to define them.
  
  Atomic environments parameters.
</details>

- <a id="atomicenvironment-style"></a>**`style`** : `Literal['cna', 'graph', 'cna/graph', 'diamond/graph', 'coordination', 'coordination/graph']`, mandatory
  <details><summary>Description</summary>
  Method used to characterize and assign an ID to an atom's local atomic environment. 'coordination' classifies atoms based on nearest-neighbor count against a threshold. 'coordination/graph' first filters by coordination, then computes graph IDs for non-crystal atoms.
  </details>
- <a id="atomicenvironment-rnei"></a>**`rnei`** : `float`, mandatory
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the first nearest neighbors of an atom. Atoms within this distance are considered direct neighbors.
  </details>
- <a id="atomicenvironment-rcut"></a>**`rcut`** : `float`, optional
  <details><summary>Description</summary>
  Radius cutoff (in Angstrom) for defining the local atomic environment. Required by the KMC event-matching, refinement, and reconstruction paths.
  </details>
- <a id="atomicenvironment-neighbors_add"></a>**`neighbors_add`** : `int`, default = `0`
  <details><summary>Description</summary>
  For the hybrid graph styles ('cna/graph', 'coordination/graph', 'diamond/graph'): 0 limits graph IDs to noncrystalline atoms; any positive value also computes graph IDs for their immediate neighbors. Values greater than one do not currently add further shells.
  </details>
- <a id="atomicenvironment-coordination_threshold"></a>**`coordination_threshold`** : `int`, optional
  <details><summary>Description</summary>
  When style is 'coordination' or 'coordination/graph', atoms with fewer neighbors (within rnei) than this value are classified as 'noncrystal'. Atoms with this many or more neighbors are classified as 'crystal'. Required when style is 'coordination' or 'coordination/graph'.
  </details>
- <a id="atomicenvironment-atom_coloring_mode"></a>**`atom_coloring_mode`** : `Literal['grey', 'full']`, default = `'full'`
  <details><summary>Description</summary>
  Controls whether element types are used in environment matching. Defaults to 'full' (species-resolved). 'grey': all atoms treated identically (grey alloy approximation). 'full': element types used in graph hashing, PSR matching, and symmetry detection.
  </details>

---

<a id="section-eventsearch"></a>

## `EventSearch` Section (mandatory)

<details><summary>Section Overview</summary>
  Parameter controling the event searches.
  
  Event search parameters.
</details>

- <a id="eventsearch-style"></a>**`style`** : `Literal['partn']`, mandatory
  <details><summary>Description</summary>
  Method used to find events.
  </details>
- <a id="eventsearch-nsearch"></a>**`nsearch`** : `int`, mandatory
  <details><summary>Description</summary>
  Number of event searches to perform per unique atomic environment.
  </details>
- <a id="eventsearch-emax_event"></a>**`emax_event`** : `float`, default = `5.0`
  <details><summary>Description</summary>
  Maximum energy barrier (in eV) for an event to be added to the reference table.
  </details>
- <a id="eventsearch-emin_event"></a>**`emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum energy forward and backward barrier (in eV) for an event to be added to the reference table.
  </details>
- <a id="eventsearch-backward_emin_event"></a>**`backward_emin_event`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Backward-barrier threshold used by the asymmetric-event rejection rule (see `energy_asymmetry`).
  </details>
- <a id="eventsearch-energy_asymmetry"></a>**`energy_asymmetry`** : `int`, default = `5`
  <details><summary>Description</summary>
  Reject an event when its forward barrier exceeds `energy_asymmetry * backward_emin_event` and its backward barrier is below `backward_emin_event`.
  </details>
- <a id="eventsearch-refined_minimum_delr_thr"></a>**`refined_minimum_delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Refinement is accepted only if the central atom moves less than this distance between the current position and the refined minimum.
  </details>
- <a id="eventsearch-refined_energy_thr"></a>**`refined_energy_thr`** : `float`, default = `0.05`
  <details><summary>Description</summary>
  Maximum allowed difference (in eV) between a reference event's initial barrier energy and its refined barrier energy.
  </details>
- <a id="eventsearch-delr_thr"></a>**`delr_thr`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  Maximum pARTn displacement between the initial configuration and at least one returned minimum for accepting an event-search result.
  </details>

---

<a id="section-psr"></a>

## `PSR` Section (mandatory)

<details><summary>Section Overview</summary>
  Parameter controlling the point set registration algorithm.
  
  Point set registration parameters.
</details>

- <a id="psr-style"></a>**`style`** : `Literal['ira']`, mandatory
  <details><summary>Description</summary>
  Method used for the point set registration (shape matching) between reference events and atomic environment of an atom having the same atomic environement ID of the event. This method is also used to find atomic environment symmetries.
  </details>
- <a id="psr-matching_score_thr"></a>**`matching_score_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Maximum value of the matching score of the algorithm used.
  </details>

---

<a id="section-rateconstant"></a>

## `RateConstant` Section (mandatory)

<details><summary>Section Overview</summary>
  Parameters used to compute rate constants.
  
  Rate constant computation parameters.
</details>

- <a id="rateconstant-style"></a>**`style`** : `Literal['constant']`, mandatory
  <details><summary>Description</summary>
  Method used to compute the prefactor of the rate constant. 
  </details>
- <a id="rateconstant-k0"></a>**`k0`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  When `style` is set to **'constant'**, this value is used directly as the pre-exponential factor ($k_0$) 
  $$ k = k_{0} \exp\left(-\frac{\Delta E}{k_{b}T}\right) $$
  </details>
- <a id="rateconstant-T"></a>**`T`** : `float`, default = `300`
  <details><summary>Description</summary>
  Temperature (in Kelvin) used for computing rate constants.
  </details>

---

<a id="section-lammps"></a>

## `LAMMPS` Section (conditionally required — required when `[Control]` `engine = lammps`)

<details><summary>Section Overview</summary>
  LAMMPS-specific parameters. Required if engine == lammps.
  
  LAMMPS parameters.
</details>

- <a id="lammps-pair_style"></a>**`pair_style`** : `str`, mandatory
  <details><summary>Description</summary>
  LAMMPS pair_style command.
  </details>
- <a id="lammps-pair_coeff"></a>**`pair_coeff`** : `str`, mandatory
  <details><summary>Description</summary>
  LAMMPS pair_coeff command.
  </details>
- <a id="lammps-min_style"></a>**`min_style`** : `str`, default = `'cg'`
  <details><summary>Description</summary>
  LAMMPS min_style command.
  </details>
- <a id="lammps-minimize"></a>**`minimize`** : `str`, default = `'1.0e-6 1.0e-8 1000 1000'`
  <details><summary>Description</summary>
  LAMMPS minimize command.
  </details>
- <a id="lammps-frz_min"></a>**`frz_min`** : `str`, default = `'1.0e-6 1.0e-8 10 10'`
  <details><summary>Description</summary>
  LAMMPS minimize command with frozen core.
  </details>

---

<a id="section-partn"></a>

## `pARTn` Section (conditionally required — required when `[EventSearch]` `style = partn`)

<details><summary>Section Overview</summary>
  pARTn parameters controling the event searches
  
  pARTn parameters.
</details>

- <a id="partn-verbosity"></a>**`verbosity`** : `int`, default = `2`
  <details><summary>Description</summary>
  pARTn verbosity
  </details>
- <a id="partn-delr_thr"></a>**`delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Threshold at which an atom is considered to have moved. This threshold affects the npart parameter in the artn.out output.
  </details>
- <a id="partn-zseed"></a>**`zseed`** : `int`, default = `0`
  <details><summary>Description</summary>
  The value of zseed is used to seed the random number generator. If the value equals 0, a new random seed is generated. The exact zseed value of each search is written in file zseed.dat, which can be useful for debugging, or re-running exact same pARTn runs.
  </details>
- <a id="partn-push_mode"></a>**`push_mode`** : `Literal['list', 'rad']`, default = `'rad'`
  <details><summary>Description</summary>
  Determines how the initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`push_dist_thr`) from the central atom.
  </details>
- <a id="partn-push_dist_thr"></a>**`push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  If `push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which all atoms receive an initial displacement.
  </details>
- <a id="partn-push_step_size"></a>**`push_step_size`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- <a id="partn-ninit"></a>**`ninit`** : `int`, default = `2`
  <details><summary>Description</summary>
  Minimum number of pushes with the initial push vector.
  </details>
- <a id="partn-lanczos_min_size"></a>**`lanczos_min_size`** : `int`, default = `10`
  <details><summary>Description</summary>
  Enforce Lanczos to always do at least this number of iterations.
  </details>
- <a id="partn-lanczos_max_size"></a>**`lanczos_max_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Maximum number of Lanczos iterations.
  </details>
- <a id="partn-lanczos_disp"></a>**`lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Scaling factor for displacement during the Lanczos algorithm
  </details>
- <a id="partn-lanczos_eval_conv_thr"></a>**`lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- <a id="partn-eigval_thr"></a>**`eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- <a id="partn-eigen_step_size"></a>**`eigen_step_size`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  The limit to the maximum size of the displacement with eigenvector.
  </details>
- <a id="partn-nsmooth"></a>**`nsmooth`** : `int`, default = `3`
  <details><summary>Description</summary>
  Number of smoothing steps from initial displacement to eigenvector.
  </details>
- <a id="partn-neigen"></a>**`neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Number of pushes along the eigenvector before starting a perpendicular relax.
  </details>
- <a id="partn-alpha_mix_cr"></a>**`alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- <a id="partn-nnewchance"></a>**`nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Number of times a search is allowed to cross a convex region (without counting the starting convex region).
  </details>
- <a id="partn-nperp"></a>**`nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Control the perpendicular relaxation.
  </details>
- <a id="partn-nperp_limitation"></a>**`nperp_limitation`** : `list[int]`, default = `[4, 8, 12, 16, -1]`
  <details><summary>Description</summary>
  Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- <a id="partn-forc_thr"></a>**`forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- <a id="partn-convergence_property"></a>**`convergence_property`** : `Literal['maxval', 'norm']`, default = `'maxval'`
  <details><summary>Description</summary>
  Reserved for selecting the pARTn force-convergence norm ('maxval': MAXVAL(ABS(force)); 'norm': NORM2(force)); currently unused, so pARTn's default remains in effect.
  </details>
- <a id="partn-nevalf_max"></a>**`nevalf_max`** : `int`, default = `9999`
  <details><summary>Description</summary>
  Stop an artn search before end when the number of force evaluations by the force engine is greater than `nevalf_max`.
  </details>
- <a id="partn-push_over"></a>**`push_over`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Factor that scales the displacement vector used to push the system from the saddle point towards a local energy minimum. 
  $$ \text{displacement} = \text{push_factor} \times v_0 \times \text{eigen_step_size} \times \text{push_over} \times 0.8 $$
  </details>
- <a id="partn-dmax"></a>**`dmax`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  dmax parameter used in the `fix ID all artn dmax value` LAMMPS command. Should be greater than `push_step_size`.
  </details>
- <a id="partn-r_nevalf_max"></a>**`r_nevalf_max`** : `int`, default = `300`
  <details><summary>Description</summary>
  Stop an artn refinement before end when the number of force evaluations by the force engine is greater than `r_nevalf_max`.
  </details>
- <a id="partn-r_max_attempts"></a>**`r_max_attempts`** : `int`, default = `5`
  <details><summary>Description</summary>
  When adjusting the saddle energy and positions, in some rare cases pARTn has trouble finding the saddle point and goes back to the minimum. In that case, we do another attempt with a different seed.
  </details>
- <a id="partn-r_delr_sad_thr"></a>**`r_delr_sad_thr`** : `float`, default = `0.4`
  <details><summary>Description</summary>
  When a saddle point is found by pARTn, we compare artn delr_sad to this threshold to check if the system went back to the minimum. If yes, new attempt.
  </details>
- <a id="partn-r_push_mode"></a>**`r_push_mode`** : `Literal['list', 'rad']`, default = `'list'`
  <details><summary>Description</summary>
  Determines how the refinement's initial atomic displacement (push) is generated around the central atom of the currently explored environment:
  - **'list'**: The push is applied *only* to the central atom.
  - **'rad'**: The push is applied to *all atoms* within a specified radial distance (`r_push_dist_thr`) from the central atom.
  </details>
- <a id="partn-r_push_dist_thr"></a>**`r_push_dist_thr`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  When `r_push_mode` is **'rad'**, this defines the radial cutoff (in Angstrom) from the central atom within which atoms receive the refinement's initial displacement.
  </details>
- <a id="partn-r_push_step_size"></a>**`r_push_step_size`** : `float`, default = `0.0001`
  <details><summary>Description</summary>
  Maximum size of a component in the initial displacement vector.
  </details>
- <a id="partn-r_ninit"></a>**`r_ninit`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Specify the minimal number of pushes with the initial push vector.
  </details>
- <a id="partn-r_lanczos_min_size"></a>**`r_lanczos_min_size`** : `int`, default = `20`
  <details><summary>Description</summary>
  Refinement: Enforce Lanczos to always do at least this number of iterations.
  </details>
- <a id="partn-r_lanczos_max_size"></a>**`r_lanczos_max_size`** : `int`, default = `50`
  <details><summary>Description</summary>
  Refinement: Maximum number of Lanczos iterations.
  </details>
- <a id="partn-r_lanczos_disp"></a>**`r_lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Refinement: Scaling factor for displacement during the Lanczos algorithm
  </details>
- <a id="partn-r_lanczos_eval_conv_thr"></a>**`r_lanczos_eval_conv_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Threshold for convergence of eigenvalue in Lanczos. Once convergence is reached, the Lanczos scheme exits.
  </details>
- <a id="partn-r_eigval_thr"></a>**`r_eigval_thr`** : `float`, default = `-0.01`
  <details><summary>Description</summary>
  Refinement: threshold for eigenvalue, which determines when to start following the eigenvector
  </details>
- <a id="partn-r_eigen_step_size"></a>**`r_eigen_step_size`** : `float`, default = `0.005`
  <details><summary>Description</summary>
  Refinement: The limit to the maximum size of the displacement with eigenvector.
  </details>
- <a id="partn-r_nsmooth"></a>**`r_nsmooth`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of smoothing steps from initial displacement to eigenvector.
  </details>
- <a id="partn-r_neigen"></a>**`r_neigen`** : `int`, default = `1`
  <details><summary>Description</summary>
  Refinement: Number of pushes along the eigenvector before starting a perpendicular relax.
  </details>
- <a id="partn-r_alpha_mix_cr"></a>**`r_alpha_mix_cr`** : `float`, default = `0.2`
  <details><summary>Description</summary>
  Refinement: This is the mixing coefficient used to create the push vector when the system enters into a convex region, i.e. when the negative curvature is lost. 
  </details>
- <a id="partn-r_nnewchance"></a>**`r_nnewchance`** : `int`, default = `0`
  <details><summary>Description</summary>
  Refinement: Number of times a search is allowed to cross a convex region (without counting the starting convex region).
  </details>
- <a id="partn-r_nperp"></a>**`r_nperp`** : `int`, default = `3`
  <details><summary>Description</summary>
  Refinement: Control the perpendicular relaxation.
  </details>
- <a id="partn-r_nperp_limitation"></a>**`r_nperp_limitation`** : `list[int]`, default = `[100]`
  <details><summary>Description</summary>
  Refinement: Limit of perpendicular relaxation steps for each ARTn step. More ARTn goes far from the basin more perpendicular relaxation are needed. This option allows the user to customize the number of perp relax. The value -1 means no limitation and -2 represent NULL.
  </details>
- <a id="partn-r_forc_thr"></a>**`r_forc_thr`** : `float`, default = `0.001`
  <details><summary>Description</summary>
  Refinement: The configuration has converged to either a saddle point, or a minimum, when the sum of the parallel and perpendicular components of the atomic forces is lower than this value.
  </details>
- <a id="partn-r_dmax"></a>**`r_dmax`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  Refinement: dmax parameter used in the `fix ID all artn dmax value` LAMMPS command. Should be greater than `r_push_step_size`.
  </details>

---

<a id="section-ira"></a>

## `IRA` Section (conditionally required — required when `[PSR]` `style = ira`)

<details><summary>Section Overview</summary>
  IRA parameters.
</details>

- <a id="ira-kmax_factor"></a>**`kmax_factor`** : `float`, default = `1.8`
  <details><summary>Description</summary>
  Multiplicative factor that needs to be larger than 1.0. Larger value increases the search space of the rotations.
  </details>
- <a id="ira-sym_thr"></a>**`sym_thr`** : `float`, default = `0.01`
  <details><summary>Description</summary>
  Threshold in terms of the Hausdorff distance. If an operation returns a distance value beyond sym_thr, then SOFI will not consider that operation as a symmetry operation.
  </details>

---

<a id="section-basin"></a>

## `Basin` Section (conditionally required — required when `[Control]` `basin = True`)

<details><summary>Section Overview</summary>
  Basin parameters
</details>

- <a id="basin-style"></a>**`style`** : `Literal['global', 'global/reconstruction']`, default = `'global'`
  <details><summary>Description</summary>
  Basin style used.
  </details>
- <a id="basin-energy_thr"></a>**`energy_thr`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Energy threshold
  </details>

---

<a id="section-reconstruction"></a>

## `Reconstruction` Section (optional)

<details><summary>Section Overview</summary>
  Reconstruction parameters
  
  Reconstruction parameters.
</details>

- <a id="reconstruction-push_fraction"></a>**`push_fraction`** : `float`, default = `0.15`
  <details><summary>Description</summary>
  Fraction used to push the system from the saddle point toward each minimum during reconstruction.
  </details>

---

<a id="section-activevolume"></a>

## `ActiveVolume` Section (conditionally required — required when `[Control]` `active_volume = True`)

<details><summary>Section Overview</summary>
  Active volume parameters
  
  Active Volume Parameters
</details>

- <a id="activevolume-ract"></a>**`ract`** : `float`, default = `6.0`
  <details><summary>Description</summary>
  Radius of entire active volume, spherical
  </details>
- <a id="activevolume-rmov"></a>**`rmov`** : `float`, default = `4.0`
  <details><summary>Description</summary>
  Radius of movable atoms in active volume, spherical
  </details>
- <a id="activevolume-AV_debug"></a>**`AV_debug`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Debug flag for active volume size checks
  </details>

---

<a id="section-eventrecycling"></a>

## `EventRecycling` Section (conditionally required — required when `[Control]` `recycle = True`)

<details><summary>Section Overview</summary>
  Event recycling parameters. Required when control.recycle = True.
</details>

- <a id="eventrecycling-style"></a>**`style`** : `Literal['displacement']`, mandatory
  <details><summary>Description</summary>
  Method used to decide which events can be recycled. 'displacement' = central atom moved less than movement_thr AND is farther than distance_thr from the executed event.
  </details>
- <a id="eventrecycling-movement_thr"></a>**`movement_thr`** : `float`, default = `0.02`
  <details><summary>Description</summary>
  Angstroms. Central atoms whose displacement from pre- to post-execution is below this are considered 'unmoved'.
  </details>
- <a id="eventrecycling-distance_thr"></a>**`distance_thr`** : `float`, default = `10.0`
  <details><summary>Description</summary>
  Angstroms. Candidate events whose central atom is farther than this (PBC-aware minimum-image) from the executed event's central atom pass the distance check.
  </details>

---

<a id="section-inactive_atoms"></a>

## `Inactive_Atoms` Section (optional)

<details><summary>Section Overview</summary>
  Atoms on which no event search can be centered. Applies both at search time (central atom selection) and at result time (events where the most-displaced atom is inactive are discarded).
  
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/environments/region.py``.
</details>

- <a id="inactive_atoms-region_type"></a>**`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- <a id="inactive_atoms-center"></a>**`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- <a id="inactive_atoms-radius"></a>**`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- <a id="inactive_atoms-inner_radius"></a>**`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- <a id="inactive_atoms-lo"></a>**`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- <a id="inactive_atoms-hi"></a>**`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- <a id="inactive_atoms-normal"></a>**`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- <a id="inactive_atoms-threshold"></a>**`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- <a id="inactive_atoms-side"></a>**`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- <a id="inactive_atoms-types"></a>**`types`** : `list[str]`, default = `[]`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- <a id="inactive_atoms-indices"></a>**`indices`** : `list[int]`, default = `[]`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

<a id="section-frozen_atoms"></a>

## `Frozen_Atoms` Section (optional)

<details><summary>Section Overview</summary>
  Atoms that cannot move during event search or refinement. Implemented via 'fix setforce 0.0 0.0 0.0' in LAMMPS wrapping fix artn.
  
  Selects atoms by type, index, or geometric region (union semantics).
  
  Used for ``inactive_atoms`` and ``frozen_atoms`` config sections.
  Runtime geometric queries (e.g. ``contains(positions)``) live in
  ``pykmc/environments/region.py``.
</details>

- <a id="frozen_atoms-region_type"></a>**`region_type`** : `Literal['sphere', 'shell', 'box', 'plane']`, optional
  <details><summary>Description</summary>
  Shape of the geometric region.
  </details>
- <a id="frozen_atoms-center"></a>**`center`** : `list[float]`, optional
  <details><summary>Description</summary>
  Center [x, y, z] for sphere or shell regions.
  </details>
- <a id="frozen_atoms-radius"></a>**`radius`** : `float`, optional
  <details><summary>Description</summary>
  Outer radius for sphere or shell regions.
  </details>
- <a id="frozen_atoms-inner_radius"></a>**`inner_radius`** : `float`, optional
  <details><summary>Description</summary>
  Inner (hollow) radius for shell regions.
  </details>
- <a id="frozen_atoms-lo"></a>**`lo`** : `list[float]`, optional
  <details><summary>Description</summary>
  Lower corner [xlo, ylo, zlo] for box regions.
  </details>
- <a id="frozen_atoms-hi"></a>**`hi`** : `list[float]`, optional
  <details><summary>Description</summary>
  Upper corner [xhi, yhi, zhi] for box regions.
  </details>
- <a id="frozen_atoms-normal"></a>**`normal`** : `Literal['x', 'y', 'z']`, optional
  <details><summary>Description</summary>
  Axis normal to the cutting plane.
  </details>
- <a id="frozen_atoms-threshold"></a>**`threshold`** : `float`, optional
  <details><summary>Description</summary>
  Position along the normal axis defining the plane.
  </details>
- <a id="frozen_atoms-side"></a>**`side`** : `Literal['inside', 'outside', 'above', 'below']`, default = `'inside'`
  <details><summary>Description</summary>
  Membership side: 'inside'/'outside' for sphere/shell/box, 'above'/'below' for plane.
  </details>
- <a id="frozen_atoms-types"></a>**`types`** : `list[str]`, default = `[]`
  <details><summary>Description</summary>
  Chemical symbols of atom types to select (e.g. ['Fe', 'O']).
  </details>
- <a id="frozen_atoms-indices"></a>**`indices`** : `list[int]`, default = `[]`
  <details><summary>Description</summary>
  0-based atom indices to select.
  </details>

---

<a id="section-bias"></a>

## `Bias` Section (conditionally required — required when `[Control]` `bias = True`)

<details><summary>Section Overview</summary>
  Event selection bias parameters.
</details>

- <a id="bias-style"></a>**`style`** : `Literal['direction', 'point', 'topo']`, mandatory
  <details><summary>Description</summary>
  Bias style: 'direction' (DirectionBias), 'point' (PointBias), or 'topo' (TopoBias).
  </details>
- <a id="bias-mode"></a>**`mode`** : `Literal['filter', 'boost']`, default = `'filter'`
  <details><summary>Description</summary>
  Selection mode. 'filter': rejection-loop removes non-accepted events. 'boost': multiplies desired event rates by a dynamic factor so they fire with probability bias_weight, without blocking other events.
  </details>
- <a id="bias-bias_weight"></a>**`bias_weight`** : `float`, default = `0.5`
  <details><summary>Description</summary>
  Target probability in (0, 1) that a desired event is selected at each step. Only used in boost mode.
  </details>
- <a id="bias-pass_unlisted"></a>**`pass_unlisted`** : `bool`, default = `False`
  <details><summary>Description</summary>
  Whether atoms not in atom_indices pass through the bias predicate unchanged. False (default): non-listed atoms are rejected/undesired. True: non-listed atoms always pass; only valid in filter mode.
  </details>
- <a id="bias-direction"></a>**`direction`** : `list[float]`, optional
  <details><summary>Description</summary>
  Direction vector [x, y, z] for 'direction' bias.
  </details>
- <a id="bias-target_point"></a>**`target_point`** : `list[float]`, optional
  <details><summary>Description</summary>
  Target point [x, y, z] for 'point' bias.
  </details>
- <a id="bias-atom_indices"></a>**`atom_indices`** : `list[int]`, optional
  <details><summary>Description</summary>
  Global atom indices to bias. None means all atoms.
  </details>
- <a id="bias-threshold"></a>**`threshold`** : `float`, default = `0.0`
  <details><summary>Description</summary>
  Minimum projection onto the bias direction for acceptance.
  </details>
- <a id="bias-topo_source"></a>**`topo_source`** : `str`, optional
  <details><summary>Description</summary>
  Source topology ID for 'topo' bias (e.g. vacancy).
  </details>
- <a id="bias-topo_target"></a>**`topo_target`** : `str`, optional
  <details><summary>Description</summary>
  Target topology ID for 'topo' bias (e.g. interstitial).
  </details>

---
