# Test data provenance

Binary fixtures tracked under `tests/data/`. Load `.npz` files with
`np.load(path, allow_pickle=False)`.

## `htst_si_vacancy_hop.npz` (SW-Si monovacancy hop, HTST physical oracle)

| Item | Value |
| --- | --- |
| System | 6x6x6 conventional diamond Si cell, one vacancy: 1727 atoms, a = 5.431 Å, cubic cell 32.5857 Å, fully periodic (the file carries no `pbc`; use `(True, True, True)`) |
| Potential | `examples/Si_vac/Si.sw` (Stillinger-Weber Si, `pair_style sw`, `pair_coeff * * Si.sw Si`); `sw` sets no masses, so pyKMC's species map (`Si`, 28.085 amu) is what LAMMPS holds |
| Event | pARTn-found symmetric vacancy migration; moving atom `central_atom_idx = 1033`; `dE_forward = 0.50889 eV`, `dE_backward = 0.50889 eV` (NEB-validated when the fixture was generated); all positions inside `[0, L)` |
| Keys | `idx_ref_forward` (4), `idx_ref_backward` (-1), `central_atom_idx`, `min1_positions`, `saddle_positions`, `min2_positions` (each `(1727, 3)` float64 Å), `types` (`(1727,)` `<U2`, all `Si`), `cell` (`(3, 3)` Å), `dE_forward`, `dE_backward` (eV) |
| sha256 | `7d081092a46e60547bb0adebcb3d368b304da221c5a0a5d9daebdc918638194b` (108103 bytes) |
| Origin | Full-system `min1`/`saddle`/`min2` of one reference event of a pyKMC HTST run on this cell, in pyKMC's catalogue-centred frame (the mover's `min1` position is the cell centre), tracked unchanged; on this tree `PrefactorService.build_request` receives the same geometry up to that rigid periodic translation. The `event_geometry_output` key named here before does not exist on this tree, and the run that wrote the file cannot be re-derived from it. In the file the mover's `min1` position is the cell centre to 4e-15 Å, so the three geometries carry the rigid periodic translation of pyKMC's catalogue centring; on this tree `build_request` receives the uncentred search coordinates, and energies, forces and Hessians are invariant under that translation |

Expected HTST results with `LammpsHTSTExtension` (`dynamical_matrix ... eskm`,
`fd_step = 0.01 Å`, `premin = False`, frozen-boundary partial Hessian,
`n_zero_modes = 0`), measured on LAMMPS 22 Jul 2025. This table is the
**min1-centred** free region (`free_region_center = "min1"`, the original
model); under the default saddle centring the same fixture gives 37 free
atoms and 21.1588 / 21.1844 THz (symmetric to 0.12 percent), as
`tests/engine/test_htst_lammps.py` asserts for both centrings:

| Setting | Free atoms | Forward nu0 | Backward nu0 |
| --- | --- | --- | --- |
| `free_radius = 6`, full system | 43 | 23.614 THz | 19.617 THz |
| `free_radius = 6`, `zone_radius = 10` | 43 | identical to full to 1e-12 relative | identical |
| `free_radius = 6`, `premin = True` | 43 | 23.628 THz | 19.629 THz |
| `free_radius = 4`, full system | 13 | 17.350 THz | 12.593 THz |

The forward value reproduces the 23.6 THz / 43 free atoms of an earlier,
independent HTST implementation. The FD kernel (`pykmc.htst.fd_hessian_fn`)
on the same engine
agrees with eskm to 1e-11 relative at `free_radius = 4`. Tests assert the
forward/backward values to 2 percent (the tolerance covers a LAMMPS or
compiler change; the eskm/FD and zone/full comparisons are the strict
oracles). The backward prefactor differs from the forward one because one
common free set, centred on the mover's `min1` position, feeds all three
Hessians (contract: one common subset and ordering).

## `htst_ni_fcc_vacancy_hop.npz` (Ni fcc monovacancy 1NN hop, HTST free-radius oracle)

| Item | Value |
| --- | --- |
| System | 8x8x8 conventional fcc Ni cell, one vacancy: 2047 atoms, a = 3.52 Å (the potential's P = 0 lattice parameter), cubic cell 28.16 Å, fully periodic (the file carries no `pbc`; use `(True, True, True)`) |
| Potential | `tests/data/Ni_v6_2.0_LKBeland2016.eam` (byte-identical to the copy in `examples/Ni_fcc_2047at_monovacancy/`; `pair_style eam/alloy`, `pair_coeff * * Ni_v6_2.0_LKBeland2016.eam Ni`, cutoff 6.5 Å); the file sets the mass, 58.71 amu, which is what the engine map carries |
| Event | pARTn-found symmetric first-neighbour vacancy hop; moving atom `central_atom_idx = 1423`, in the vacancy's first-neighbour shell in `min1` (2.459 Å from the vacancy site; the 12-atom shell spans 2.459-2.460 Å, the next shell starts at 3.518 Å); in `min2` it sits 0.031 Å from the vacancy site (displacement 2.428 Å) and every other atom is within 0.031 Å of its `min1` position (0.175 Å at the saddle); `dE_forward = 1.188043 eV`, `dE_backward = 1.188043 eV` (they differ by 3e-8 eV; LAMMPS recomputes both from the stored geometry to 2e-10 eV); largest atomic force at most 1.05e-3 eV/Å in each of the three geometries; all positions inside `[0, L)` |
| Keys | `idx_ref_forward` (0), `idx_ref_backward` (1), `central_atom_idx`, `min1_positions`, `saddle_positions`, `min2_positions` (each `(2047, 3)` float64 Å), `types` (`(2047,)` `<U2`, all `Ni`), `cell` (`(3, 3)` Å), `dE_forward`, `dE_backward` (eV); deflate-compressed like the SW-Si file |
| sha256 | `aba822c40edd9fd7b746c552ce1ee51fb8276f6e9273e07340453fbe5edfe4ce` (121063 bytes) |
| Origin | The cell is `examples/Ni_fcc_2047at_monovacancy/initial_config.xyz`, which is not a minimum (largest atomic force 0.284 eV/Å); it was relaxed at fixed box with LAMMPS CG (`min_modify line quadratic`, `minimize 0.0 1e-10 100000 1000000`) to a largest force of 5e-12 eV/Å (largest displacement 0.030 Å). The event comes from one in-process KMC step with `[RateConstant] style = htst` (`free_radius = 6`, `interaction_range = 13`, `k0 = 10`, `T = 300`), `[Control] seed = 12345`, `n_sessions = 1`, `[pARTn] zseed = 12345`, `[EventSearch] nsearch = 10`, `emax_event = 5.0`, `[AtomicEnvironment] style = cna/graph`, `rnei = 3.0`, `rcut = 6.5`, `[Lammps] minimize = 1e-10 1e-12 10000 10000`, IRA registration (`matching_score_thr = 0.4`, `kmax_factor = 2.0`). It ran through the in-process manager of `tests/regression/si_vac_constant_harness.py` (two serial `Engine.create(..., comm=MPI.COMM_SELF)` engines, with `LammpsHTSTExtension` attached to both as `run.py` does), and `PrefactorService.build_request` was wrapped on the class to copy its inputs. The vacancy's 12 first neighbours formed the one new environment. Its 10 searches gave one new event (search atom 1423, catalogued as reference events 0 and 1, both directions retained) and two duplicates. The file holds that request's `min1`/`saddle`/`min2` after one rigid translation t = (L/2 + a/4)(1, 1, 1) - vacancy site = (-0.88, 0.88, 2.64) Å and wrapping into `[0, L)`. In the captured coordinates, boundary atoms sit up to 3 mÅ outside the cell, on different sides of a face in different geometries. After the shift every lattice plane lies a/4 from the cell faces, and raw differences between the geometries equal their minimum-image differences. ν0 computed on the untranslated capture agrees with the file to 4e-12 relative |

Expected HTST results with `LammpsHTSTExtension` (`dynamical_matrix ... eskm`,
`fd_step = 0.01 Å`, `premin = False`, frozen-boundary partial Hessian,
free region centred on the saddle (the default) unless stated). The engine is
serial and holds the fixture `min1`, and requests are built as in the
`hop_request` fixture of `tests/engine/test_htst_lammps.py`. Measured on
LAMMPS 22 Jul 2025 with one core of an Apple M1 Max. Every direction is `ok`
with exactly one negative saddle mode and `3 * n_free` stable modes at each
minimum:

| Setting | Free atoms | Forward nu0 | Backward nu0 | Wall |
| --- | --- | --- | --- | --- |
| `free_radius = 6`, full system | 83 | 22.0389 THz | 22.0388 THz | 4.3 s |
| `free_radius = 6`, `zone_radius = 19` | 83 | identical to full (bit-for-bit here: the 13 Å shell is twice the 6.5 Å cutoff) | identical | 4.1 s |
| `free_radius = 4`, full system | 23 | 15.3830 THz | 15.3830 THz | 1.3 s |
| `free_radius = 6`, `free_region_center = "min1"`, full system | 78 | 23.0156 THz | 21.0840 THz | 4.0 s |

The in-run reference estimate of event 0 (the same request through the live
prefactor path) was 22.0389 / 22.0388 THz with 83 free atoms.

## `htst_fe_bcc_vacancy_hop.npz` (Fe bcc monovacancy, first half of the 1NN hop, HTST free-radius oracle)

| Item | Value |
| --- | --- |
| System | 10x10x10 conventional bcc Fe cell, one vacancy: 1999 atoms, a = 2.855325 Å (the potential's P = 0 lattice parameter), cubic cell 28.55325 Å, fully periodic (the file carries no `pbc`; use `(True, True, True)`) |
| Potential | `tests/data/Fe_mm.eam.fs` (byte-identical to the copy in the LAMMPS distribution's `potentials/` directory; Mendelev et al., Philos. Mag. 83, 3977 (2003), potential 2; see its entry below; `pair_style eam/fs`, `pair_coeff * * Fe_mm.eam.fs Fe`, cutoff 5.3 Å); the file sets the mass, 55.845 amu, which is what the engine map carries |
| Event | pARTn-found **first half** of the first-neighbour vacancy migration. This potential's migration path is double-humped, with a local minimum at the hop midpoint (the split vacancy; the same artefact is reported for the Ackland 2004 derivative of this potential by Brommer et al., Phys. Rev. B 90, 134109 (2014)). `min2` is that split-vacancy minimum, not the vacancy site. Moving atom `central_atom_idx = 1091`, in the vacancy's first-neighbour shell in `min1` (2.400 Å from the vacancy site; the 8-atom shell spans 2.400-2.401 Å, the next shell starts at 2.893 Å). In `min2` it sits 0.037 Å from the midpoint between its `min1` position and the vacancy site, and 1.237 Å from the vacancy site (displacement 1.163 Å); every other atom is within 0.125 Å of its `min1` position (0.159 Å at the saddle). `dE_forward = 0.628734 eV` (the 1NN migration barrier of this potential), `dE_backward = 0.092396 eV`: **not symmetric**, E(`min2`) - E(`min1`) = 0.5363 eV. A full-system CG relaxation from `min2`, and from `min2` with the mover displaced 0.1 or 0.3 Å either way along the hop, returns to the same split-vacancy minimum. Largest atomic force 1.7e-3 / 1.5e-3 / 1.0e-3 eV/Å in `min1` / saddle / `min2`; LAMMPS recomputes both barriers from the stored geometry to 1e-11 eV; all positions inside `[0, L)` |
| Keys | `idx_ref_forward` (0), `idx_ref_backward` (1), `central_atom_idx`, `min1_positions`, `saddle_positions`, `min2_positions` (each `(1999, 3)` float64 Å), `types` (`(1999,)` `<U2`, all `Fe`), `cell` (`(3, 3)` Å), `dE_forward`, `dE_backward` (eV); deflate-compressed like the SW-Si file |
| sha256 | `13b5a39b35414d3507c6b20cb07db232049883650f9a8d12f169daab8388fed8` (121853 bytes) |
| Origin | The cell was built with the workspace's periodic structure generator (`single-crystal`, bcc, 10x10x10, explicit composition Fe 1.0, `add_vacancy`, which removes the site nearest the box centre, (5a, 5a, 5a)). The lattice parameter was determined for this potential on the perfect 2000-atom cell: E(a) quartic fit 2.8553246 Å, P(a) = 0 at 2.8553246 Å, isotropic `box/relax` 2.8553249 Å. The cell was relaxed at fixed box with LAMMPS CG (`min_modify line quadratic`, `minimize 0.0 1e-10 100000 1000000`) to a largest force of 4e-12 eV/Å (largest displacement 0.072 Å; vacancy formation energy 1.7125 eV). The event comes from one in-process KMC step set up as for the Ni fixture (`style = htst`, `free_radius = 6`, `interaction_range = 10.6`, `seed = 12345`, `zseed = 12345`, `nsearch = 10`, `emax_event = 5.0`, `cna/graph`, `rcut = 6.5`, same minimiser and IRA settings, same in-process manager and `build_request` wrapper), except `rnei = 3.4` so the CNA sees the 14 first and second neighbours of bcc. The vacancy's 8 first and 6 second neighbours formed the two new environments. Their 20 searches gave one new event (search atom 1091, catalogued as reference events 0 and 1), one duplicate and two events above `emax_event`; a second run with `nsearch = 20` (40 searches) found the same single event and no hop ending on the vacancy site. The file holds that request's geometry after the same rigid translation as the Ni fixture, t = (a/4)(1, 1, 1) = 0.714 Å per axis (the vacancy site moves from 5a to 5a + a/4), and wrapping into `[0, L)`. ν0 computed on the untranslated capture agrees with the file to 4e-12 relative |

Expected HTST results with the same settings and harness as the Ni fixture.
The backward direction and the `free_radius = 4` forward direction are `ok`
under the default acceptance window. At `free_radius = 6` the forward nu0 is
above the default `nu0_max_hz = 1e14`, so with default settings that
direction is `rejected` (`out_of_window`) and carries no value (the live run
logged the same rejection for reference event 0). Its values below were
measured with `nu0_max_hz = 1e16`; the other entries are identical under
both windows. Every computed direction has exactly one negative saddle mode
and `3 * n_free` stable modes at each minimum:

| Setting | Free atoms | Forward nu0 | Backward nu0 | Wall |
| --- | --- | --- | --- | --- |
| `free_radius = 6`, full system | 73 | 141.452 THz (`rejected`, `out_of_window`, under the default window) | 14.5869 THz | 1.9 s |
| `free_radius = 6`, `zone_radius = 19` | 73 | identical to full (bit-for-bit here: the 13 Å shell exceeds twice the 5.3 Å cutoff) | identical | 2.0 s |
| `free_radius = 4`, full system | 18 | 68.008 THz | 12.0739 THz | 0.6 s |
| `free_radius = 6`, `free_region_center = "min1"`, full system | 64 | 118.919 THz (`rejected` under the default window) | 12.9828 THz | 1.7 s |

Between `free_radius = 4` and `6` the forward value changes by a factor of 2.1
and the backward value by 21 percent.

## `Fe_mm.eam.fs` (Mendelev 2003 bcc Fe EAM potential)

| Item | Value |
| --- | --- |
| Potential | Mendelev, Han, Srolovitz, Ackland, Sun, Asta, Phil. Mag. 83 (2003) 3977, as shipped in the LAMMPS `potentials/` directory (copied unchanged from the LAMMPS 22 Jul 2025 distribution; the file header dates it 2007-06-11 and names only the authors; the NIST Interatomic Potentials Repository entry for this file identifies it as the paper's potential 2) |
| Format | Finnis-Sinclair EAM, one element: `pair_style eam/fs`, `pair_coeff * * Fe_mm.eam.fs Fe`; cutoff 5.3 Å; the file sets the mass, 55.845 amu |
| sha256 | `d7b15f435a3f5192e8ed264d0dcbc052f1d5641ba5db610eab88199d77d20f7c` (756468 bytes) |
| Consumers | `htst_fe_bcc_vacancy_hop.npz` was generated with it; `tests/validation/test_free_radius_convergence.py` (opt-in, `PYKMC_RUN_VALIDATION=1`) computes the Fe hop's prefactors with it |

## `constant_endpoint_develop_c6f2824.json` (constant-mode develop oracle for `tests/regression`)

| Item | Value |
| --- | --- |
| System | `examples/Si_vac`: 511-atom SW-Si cell with one vacancy (`Si.sw`, `initial_config.xyz`, sha256 in `meta.example_sha256`) |
| Run | 10 constant-mode steps, `n_sessions = 1`, `nsearch = 2`, `rcut = 6.3`, `k0 = 10`, `T = 300`; `random`/NumPy seeded 12345 immediately before `KMC` construction (develop has no `[Control] seed`), `[pARTn] zseed = 12345`; the full INI is in `meta.ini` |
| Recorded | once, on develop `c6f28245f93893d4f997cd39525e1fb1c9a24e18` (tree `28bff77a`), with `tests/regression/si_vac_constant_harness.py` in a subprocess launched with `PYTHONHASHSEED=20260919` (develop orders new environments by string hash; this tree sorts them), cwd a scratch directory, `PYTHONPATH` the develop tree; the adapter run is bit-identical to a real `mpirun -n 2` develop run of the same input (trajectory, `pykmc.out`, reference table, restart time) |
| Keys | `k_tot` (ps⁻¹) and `T` (s, the run's `total_time`) after step 10; `selected` = executed `(atom_index, energy_barrier)` per step; `reference_pairs` = sorted `(dE_forward, dE_backward)` multiset of the final catalogue built from `idx_backward` links; `steps` (per-step draws, searched atoms, new environments, attempts) for failure messages; `checks` (cross-checks against `pykmc.out` and `restart_10.npz`); `meta` (versions, command shape, placeholders for local paths) |
| sha256 | `8d001c8266f30df2ce5d2596a171cef87521cc599b61afeca2b8b45e6d6fee48` (12741 bytes) |
| Consumer | `tests/regression/test_constant_endpoint.py`: atom indices equal, barriers within 1e-9 eV, `k_tot` and `T` within 1e-6 relative |
