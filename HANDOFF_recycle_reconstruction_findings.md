# Handoff: recycle reconstruction non-viability — investigation findings (2026-06-22)

> ## ✅ RESOLVED (2026-06-24) — corrected diagnosis; this doc's framing is superseded
>
> This file's mechanism (a full-cell minimize seam instability) and its two candidate fixes
> (freeze-outer / adaptive `push_fraction`) were **not** the root cause. The actual cause is a
> **neighbour-ORDERING scatter**: the stored saddle shell is presented in refinement-neighbour
> order against a *live* re-derived order, so `push_towards` pairs row *i* with the wrong atom →
> overlap → `Lost atoms` in the min1 minimize. Fixed by `ba0841d` (persist neighbour ordering) +
> `582060b` (acceptance redesign) on this branch — see the RESOLVED banner in
> [`HANDOFF_recycle_reconstruction_nonviable.md`](HANDOFF_recycle_reconstruction_nonviable.md)
> for the full write-up, the 16/16-configs-@-0-Lost-atoms validation, and why
> `freeze_outer_op.patch` is a dead end.
>
> The empirical observations below remain accurate **as data** — the gentle σ-perturbation does
> surface a mis-land (not Lost atoms), and freeze-outer does *not* fix that mis-land. Only their
> **interpretation** as "full-cell fragility" was wrong: both are explained by the ordering
> scatter (the σ probe is just too gentle to overlap hard, exactly like a vacancy hop). New
> decisive probe: [`recycle_diagnostics/recon_order.py`](recycle_diagnostics/recon_order.py).

**Companion to** [`HANDOFF_recycle_reconstruction_nonviable.md`](HANDOFF_recycle_reconstruction_nonviable.md)
(the original problem statement). This file records a verified root-cause re-read of the
code **plus an empirical local investigation** that (a) confirms the bug needs 32k scale,
and (b) **partly refutes the "freeze-outer local minimize" fix recommendation** — that fix
was prototyped and did *not* resolve the failure mode reproducible locally.

**Read this on the 32k machine before re-attempting a fix.** The actionable next step is a
*diagnosis-first* run (which failure mode is it?), not a blind fix.

---

## 1. Verified root cause (re-read against `bug_recycle_reconstruction_nonviable`)

The reconstruct minimize at [`reconstruction.py`](pykmc/reconstruction.py) L66 and L94 is an
**unconstrained, full-cell** relaxation:

- `manager.global_minimize_with_results(...)` resolves via the `global_` `__getattr__`
  shim ([`pool/manager.py`](pykmc/enginemanager/lmpi/pool/manager.py) ~L231) to
  `global_session.minimize_with_results` → [`lammps_operations.py`](pykmc/enginemanager/lmpi/lammps_operations.py)
  `minimize_with_results` (L482), which minimizes **all atoms** (the `_make_frozen_group`
  step there only freezes a *static user* `config.frozen_atoms` region — `None` for a bulk
  cell, so nothing is frozen).
- In [`kmc.py`](pykmc/kmc.py) `_reconstruction_active_event` (L699–718): `neighbors` is the
  `rcut` shell of the central atom (L701); the stored generic **saddle** shell is transplanted
  onto **only that shell** (L710); `supposed_min1` is read **live** from the current system
  (L704); everything beyond the shell stays at its **live minimized** positions.

**Mechanism:** a transplanted (IRA-mapped, generic) saddle shell meets the surrounding live
lattice, then the *whole cell* is relaxed unconstrained. The shell↔lattice seam is the
instability; at 32k the unconstrained relaxation over thousands of soft collective DOFs lets
atoms cross → LAMMPS `Lost atoms` → `RuntimeError` (caught) → `Err(RECONSTRUCTION_MINIMIZE_FAILED)`.

**Correction to the earlier triage synthesis:** the far field is **not** "left at the saddle"
— it's at the live minimum. Only the `rcut` shell is at the (pushed) saddle. The bug is the
seam-under-unconstrained-global-minimize, not a whole-cell saddle.

**The codebase already knows this:** [`config.py`](pykmc/config.py) `BasinConfig.style` documents
that the plain `global` minimize *"can lose atoms in the minimize on some systems,"* and basin
reconstruction was hardened with `global/reconstruction` + `_minimize_freeze_outer_sphere`
([`lammps_operations.py`](pykmc/enginemanager/lmpi/lammps_operations.py) L142, freezes atoms
**beyond** `rmov`). That primitive is the basis of the candidate fix — see §3.

---

## 2. Empirical findings (local: macOS, `mpirun -n 2`, Ni `Ni_v6_2.0_LKBeland2016.eam`)

Reproduction scripts are in [`recycle_diagnostics/`](recycle_diagnostics/) (see its README). All run
a **real** `Reconstruction.reconstruct` through the manager on a Ni FCC vacancy-hop event.

1. **The harness works**, and there was a coverage gap: `tests/test_reconstruction.py` only
   *mocks* the manager, so the real full-cell minimize had **zero** test coverage.
   - Use `ManagerFactory(use_rank_0=False)`. `use_rank_0=True` (the input's default, and the
     deprecated path) **deadlocks** — both ranks busy-spin at ~97% CPU on a 256-atom cell.

2. **Clean reconstruction is robust** — a self-consistent Ni vacancy hop reconstructs `Ok`
   from 255 (4³) up to 3999 (10³) atoms. So the bug is **not** pure scale for consistent events.

3. **Perturbing the transplanted saddle shell reproduces a reconstruction *failure* — but the
   WRONG one.** Gaussian shell perturbation σ (Å), `recon_sweep.py`:

   | cell | atoms | first failing σ | delr1 |
   |---|---|---|---|
   | 6³ | 863 | 0.30 | 2.43 |
   | 8³ | 2047 | 0.50 | 2.43 |
   | 10³ | 3999 | 0.70 | 3.52 |

   - Mode is **`RECONSTRUCTION_INVALID_MIN1`** (mis-land; delr1 ≈ Ni NN 2.49 Å → the vacancy
     hops into the **wrong basin**), **not `Lost atoms`** (the 32k symptom).
   - The failing σ **increases** with cell size — i.e. **bigger systems are more tolerant**.
     That is **opposite** to the real bug (worse with scale) → this is a finite-size / PBC
     artifact, **not the 32k mechanism**.

4. **The freeze-outer "fix" does NOT fix this mode.** A prototype `minimize_freeze_outer_with_results`
   op (patch in `recycle_diagnostics/freeze_outer_op.patch`) wired into reconstruct with
   `rmov = rcut`, at (8³, σ=0.7): **full-cell and freeze-outer both fail** with delr1≈2.43.
   The perturbation lives *inside* the relaxed shell, so freezing the *outer* field cannot
   prevent a within-shell hop (`recon_compare.py`).

5. **`push_fraction` does fix this mode.** At (8³, σ=0.7), reconstruction is `FAIL` for
   `push_fraction ≤ 0.35` and `Ok` for `≥ 0.55` (`recon_pf.py`). A *larger* push (closer to the
   target minimum) lands in the right basin. (Original handoff guessed *smaller* push — for the
   mis-land mode, **larger** helps; "smaller" may still matter for the Lost-atoms mode.)

---

## 3. What this means for the fix

- The local perturbation reproduction is a **mis-land** that is **fixed by push_fraction, not by
  freeze-outer**. The real 32k failure is **Lost atoms** (numerical divergence of the full-cell
  minimize), which freeze-outer (DOF cropping) is designed to address — but **I could not
  reproduce Lost atoms locally**, so freeze-outer remains **unvalidated for this bug.**
- Therefore the fix is **not settled**. There are (at least) two distinct failure modes with
  different remedies:
  - `RECONSTRUCTION_MINIMIZE_FAILED` (Lost atoms) → likely **freeze-outer / local minimize**.
  - `RECONSTRUCTION_INVALID_MIN1/2` (mis-land) → **adaptive/larger push_fraction**, and/or
    **IRA match-score gating** (reject poorly-mapped events before reconstruction).

---

## 4. Next steps on the 32k machine (diagnosis first)

1. **Classify the failures.** Re-run the `recycle` / `av+recycle` benchmark on
   `Ni_fcc_32000at_4vac+4sia` and count, among the dropped reconstructions, how many are
   `RECONSTRUCTION_MINIMIZE_FAILED` (Lost atoms) vs `RECONSTRUCTION_INVALID_MIN1/2` (mis-land).
   The original handoff reports the log says `Lost atoms`, so the **minimize-failed** mode is
   expected to dominate — confirm the count. This bifurcation decides the fix.
2. **If Lost atoms dominates** → apply `recycle_diagnostics/freeze_outer_op.patch`, point
   reconstruct's two minimizes at `global_minimize_freeze_outer_with_results`
   (`central_atom = active_table...atom_index`, `rmov = config.atomicenvironment.rcut`), rerun,
   and check whether the `Lost atoms` failures vanish and the run reaches ≥5 steps.
3. **If mis-land dominates** → freeze-outer won't help; test adaptive `push_fraction` (and/or
   gate recycled events on IRA match score).
4. **Scale the diagnostics.** `recon_sweep.py` / `recon_compare.py` accept a `repeat` arg — push
   them toward 32k (replicate the cell, or load the real config) to confirm at real size which
   mode appears and which remedy works **with a clean self-consistent event** (no artificial σ).

---

## 5. Gotchas that cost time here (save yourself the trouble)

- **Import path.** A script run from *outside* the repo (e.g. `/tmp/foo.py`) puts its own dir on
  `sys.path[0]`, so `import pykmc` resolves to the **editable-installed** checkout (here:
  `pyKMC-develop` on `develop`), **not** this branch. Symptom: `Config` has **no
  `.reconstruction`** (that field only exists on this branch). Fix: run scripts from the repo
  root, or `export PYTHONPATH=<this checkout>`. On the 32k box, confirm
  `python -c "import pykmc; print(pykmc.__file__)"` points at the bug-branch checkout.
- **`use_rank_0`.** Drive the diagnostic manager with `use_rank_0=False` (rank 0 = pure driver,
  rank 1 = engine). `use_rank_0=True` is the deprecated rank-0-double-duty path and **hangs**.
- **Benign init noise.** Driving both `initialize_sessions` and `global_initialize_*` on a
  single-session pool prints `Units command after simulation box is defined` / `Reuse of region
  ID box` — harmless (same physical engine, box already built); the minimize energies are correct
  (e.g. 255-atom Ni cell → −4.44 eV/atom).

---

## 6. Key anchors (verified on this branch)

| what | location |
|---|---|
| reconstruct push + full-cell minimize | [`reconstruction.py`](pykmc/reconstruction.py) L58/L66 (min1), L90/L94 (min2) |
| recycle reconstruction call site | [`kmc.py`](pykmc/kmc.py) `_reconstruction_active_event` L699–718 |
| global-minimize shim | [`pool/manager.py`](pykmc/enginemanager/lmpi/pool/manager.py) `__getattr__` ~L231 |
| full-cell minimize op | [`lammps_operations.py`](pykmc/enginemanager/lmpi/lammps_operations.py) `minimize_with_results` L482 |
| freeze-outer primitive (fix basis) | [`lammps_operations.py`](pykmc/enginemanager/lmpi/lammps_operations.py) `_minimize_freeze_outer_sphere` L142 |
| "global minimize can lose atoms" note | [`config.py`](pykmc/config.py) `BasinConfig.style` |
| push_fraction knob | [`config.py`](pykmc/config.py) `ReconstructionConfig.push_fraction` (default 0.15) |
| diagnostics + freeze-outer patch | [`recycle_diagnostics/`](recycle_diagnostics/) |
