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
  Path to a reference table generated from a previous simulation.
  </details>
- **`visited_environments`** : `str`, optional
  <details><summary>Description</summary>
  Path to a list of visited environment generated from a previous simulation.
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
- **`engine_use_rank_0`** : `bool`, default = `True`
  <details><summary>Description</summary>
  If use mpi rank 0 or not.
  </details>
- **`verbosity`** : `int`, default = `1`
  <details><summary>Description</summary>
  Controls the level of detail in the simulation output.
  </details>
- **`refine_thr`** : `int`, default = `0.99`
  <details><summary>Description</summary>
  Event constributing to this percent of ktot are refined.
  </details>

---

## `Atomicenvironment` Section (mandatory)

<details><summary>Section Overview</summary>
  Atomic environments parameters.
</details>

- **`style`** : `Literal['cna', 'graph', 'cna/graph']`, mandatory
  <details><summary>Description</summary>
  Method used to characterize and assign an ID to an atom's local atomic environment
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
- **`backward_emin_event`** : `float`, default = `0.05`
  <details><summary>Description</summary>
  To be used with `energy_assymetry`.
  </details>
- **`energy_asymmetry`** : `int`, default = `5`
  <details><summary>Description</summary>
  Prevent highly asymmetric event to be added to the reference table.The con
  </details>
- **`refined_minimum_delr_thr`** : `float`, default = `0.1`
  <details><summary>Description</summary>
  Refinement is accepted only if the central atom moves less than this distance between the current position and the refined minimum.
  </details>
- **`refined_energy_thr`** : `float`, default = `0.2`
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
</details>

- **`style`** : `Literal['constant']`, mandatory
  <details><summary>Description</summary>
  Method used to compute the prefactor of the rate constant. 
  </details>
- **`k0`** : `float`, default = `1.0`
  <details><summary>Description</summary>
  When `style` is set to **'constant'**, this value is used directly as the pre-exponential factor ($k_0$) 
  $$ k = k_{0} \exp\left(-\frac{\Delta E}{k_{b}T}\right) $$
  </details>
- **`T`** : `float`, default = `300`
  <details><summary>Description</summary>
  Temperature (in Kelvin) used for computing rate constants.
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
- **`lanczos_min_size`** : `float`, default = `10`
  <details><summary>Description</summary>
  Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`lanczos_max_size`** : `float`, default = `20`
  <details><summary>Description</summary>
  Maximum number of Lanczos iterations.
  </details>
- **`lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`lanczos_eval_conv_thr`** : `float`, default = `0.01`
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
  Specify how to test convergence of the forces. 'maxval': the convergence will be tested by MAXVAL( ABS( force ) ); 'norm' the convergence will be tested by NORM2( force ).
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
- **`path_artnso`** : `str`, mandatory
  <details><summary>Description</summary>
  Path to use to load the plugin with lammps command 'plugin load /path/to/artn-plugin/libartn.so'
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
- **`r_lanczos_min_size`** : `float`, default = `20`
  <details><summary>Description</summary>
  Refinement: Enforce Lanczos to always do at least this number of iterations.
  </details>
- **`r_lanczos_max_size`** : `float`, default = `50`
  <details><summary>Description</summary>
  Refinement: Maximum number of Lanczos iterations.
  </details>
- **`r_lanczos_disp`** : `float`, default = `0.0005`
  <details><summary>Description</summary>
  Refinement: Scaling factor for displacement during the Lanczos algorithm
  </details>
- **`r_lanczos_eval_conv_thr`** : `float`, default = `0.01`
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

## `EventRecycling` Section (optional)

<details><summary>Section Overview</summary>
  Event recycling parameters. After the selected event is executed at a step, candidate events whose central atom (a) didn't move and (b) is far from the executed event are carried over to the next step instead of being re-searched. If <code>enabled = False</code> (default), behavior is identical to prior versions.
</details>

- **`enabled`** : `bool`, default = `False`
  <details><summary>Description</summary>
  If True, reuse non-perturbed events from the previous step instead of re-searching them.
  </details>
- **`movement_thr`** : `float`, default = `0.02`
  <details><summary>Description</summary>
  Angstroms. Central atoms whose displacement from pre- to post-execution of the selected event is below this are considered "unmoved".
  </details>
- **`distance_thr`** : `float`, default = `10.0`
  <details><summary>Description</summary>
  Angstroms. Candidate events whose central atom is farther than this (PBC-aware minimum-image distance) from the executed event's central atom pass the distance check.
  </details>

---
