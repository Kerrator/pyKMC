# recycle_diagnostics

Standalone scripts that drive a **real** `Reconstruction.reconstruct` through the MPI manager
on a Ni FCC vacancy-hop event, used to investigate the recycle reconstruction non-viability
bug. See [`../HANDOFF_recycle_reconstruction_findings.md`](../HANDOFF_recycle_reconstruction_findings.md)
for the findings these produced.

> **✅ Resolved (2026-06-24).** Root cause = **neighbour-ORDERING scatter** (stored saddle in
> refinement-neighbour order vs live re-derived order → `push_towards` pairs wrong atoms → overlap →
> `Lost atoms` in min1), fixed by `ba0841d`+`582060b` on this branch and validated 32k (16/16 configs,
> 0 Lost atoms). The decisive probe is **`recon_order.py`**. `freeze_outer_op.patch` is a **dead end**
> (wrong diagnosis). The `recon_sweep/pf/compare` scripts reproduce only the *mis-land* side-mode and
> are kept as mechanism probes, not a faithful 32k repro.

## Prerequisites

- A built LAMMPS with the EAM Ni potential (`tests/data/Ni_v6_2.0_LKBeland2016.eam`) and `mpirun`.
- **Run from the repository root** (scripts reference `./tests/data/input.in`), and make sure
  `import pykmc` resolves to *this* checkout:

  ```bash
  cd <this checkout>
  python -c "import pykmc; print(pykmc.__file__)"   # must point INTO this checkout
  export PYTHONPATH=$PWD                              # belt-and-braces if it doesn't
  ```

  > A script run from outside the repo puts its own dir on `sys.path[0]`, so `import pykmc`
  > silently picks the editable-installed checkout (often `develop`, which has **no
  > `Config.reconstruction`**). Symptom: `AttributeError: 'Config' object has no attribute
  > 'reconstruction'`. Run from the repo root or set `PYTHONPATH`.

All scripts use `ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True)` → `mpirun -n 2`
(rank 0 driver, rank 1 engine). Do **not** use `use_rank_0=True` — it deadlocks.

## Scripts

| script | what it shows | example |
|---|---|---|
| `recon_order.py` | **decisive** — neighbour-ORDER scatter is the root cause: identity order `Ok`, every permutation `FAIL`, at all sizes incl. 32000 atoms (clean self-consistent event, no σ) | `mpirun -n 2 python recycle_diagnostics/recon_order.py 6` |
| `recon_smoke.py` | one real reconstruction succeeds (`Ok`) at small scale | `mpirun -n 2 python recycle_diagnostics/recon_smoke.py 4` |
| `recon_sweep.py` | reconstruct a shell perturbed by σ, swept over size — reproduces the **mis-land** failure and its (inverted) size scaling | `mpirun -n 2 python recycle_diagnostics/recon_sweep.py 8 0.0,0.3,0.5,0.7,0.9` |
| `recon_pf.py` | `push_fraction` sweep at a red point — larger push (≥0.55) recovers `min1` | `mpirun -n 2 python recycle_diagnostics/recon_pf.py 8 0.7 0.05,0.15,0.35,0.55,0.85` |
| `recon_compare.py` | full-cell **vs** freeze-outer reconstruct — shows freeze-outer does **not** fix the mis-land. **Requires `freeze_outer_op.patch` applied** | `git apply recycle_diagnostics/freeze_outer_op.patch && mpirun -n 2 python recycle_diagnostics/recon_compare.py 8 0.7` |

## `freeze_outer_op.patch` — ⚠️ DEAD END (kept for the record)

This patch chased the **wrong** (full-cell-fragility) diagnosis and does **not** fix the bug —
freezing the *outer* field cannot prevent a *within-shell* neighbour-order scatter. The real fix is
neighbour-order persistence (`ba0841d`+`582060b`); use `recon_order.py` to see the mechanism. Original
description follows.

Prototype only — adds a `minimize_freeze_outer_with_results` engine op + session method so the
manager exposes `global_minimize_freeze_outer_with_results` (local reconstruction minimize:
relax only atoms within `rmov` of the central atom, freeze beyond). Needed by `recon_compare.py`
and as the starting point if the 32k diagnosis points at the **Lost atoms** mode (see the
handoff §3–4). Apply with `git apply`, revert with `git apply -R`.

> ⚠️ The artificial shell perturbation in `recon_sweep`/`recon_compare`/`recon_pf` reproduces a
> `RECONSTRUCTION_INVALID_MIN1` **mis-land**, which is **not** the 32k `Lost atoms` mode and
> scales the wrong way with size. Treat these as mechanism probes, not a faithful 32k repro.
> The faithful repro needs the cluster — see the handoff.
