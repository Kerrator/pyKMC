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
| Origin | `event_geometry_output` dump of a pyKMC HTST run on this cell, tracked unchanged |

Expected HTST results with `LammpsHTSTExtension` (`dynamical_matrix ... eskm`,
`fd_step = 0.01 Å`, `premin = False`, frozen-boundary partial Hessian,
`n_zero_modes = 0`), measured on LAMMPS 22 Jul 2025:

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
