"""Derive the effective bond energy E_b for the dissolution model from the EAM.

Method (bond-counting calibration against our own potential, so E_b is
self-consistent with the pARTn/HTST hop barriers computed from the same EAM):

1. Load a production slab (extxyz with Lattice), pick surface atoms of the
   target species (top layer, first-shell coordination 8 within RNEI = the
   pyKMC [AtomicEnvironment] rnei).
2. For each target, progressively delete its neighbours (in-plane first, then
   subsurface) to leave it with n = 8, 7, ..., N_MIN remaining first-shell
   neighbours -- the adatom/kink/terrace ladder a dealloying surface exposes.
3. At each n, compute the UNRELAXED removal energy
       dE(n) = E(config without target) - E(config with target)
   with single-point LAMMPS evaluations (eam/alloy, alphabetical type order,
   same convention as pyKMC's engine setup).
4. Least-squares fit dE(n) = E_b * n + c pooled over targets: the slope is the
   effective per-bond energy of the bond-counting rate
       k_diss(n) = nu_d * exp((phi - n*E_b) / (kb*T)).

Unrelaxed by design: bond counting is a rigid-lattice concept; relaxation
effects belong to the driving-force term phi, not to E_b. The intercept c
absorbs the many-body (embedding) offset of the EAM and is reported but unused.

Usage:
    python derive_Eb_eam.py            # runs NiCr (Cr, Ni) and NiFe (Fe, Ni)
"""

from __future__ import annotations

import os
import sys
import tempfile

# Isolate this script's MPI session state BEFORE any lammps import: each
# lammps() call here is an Open MPI singleton, and singletons share the
# per-user session-directory tree under $TMPDIR with any concurrently running
# mpirun job. A finalizing singleton can tear down state a live production run
# depends on (observed: a 5-rank pykmc run died silently, no traceback, the
# minute this script exited). A private TMPDIR removes the collision.
_tmp = tempfile.mkdtemp(prefix="derive_eb_ompi_")
os.environ["TMPDIR"] = _tmp
os.environ.setdefault("OMPI_MCA_orte_tmpdir_base", _tmp)

import numpy as np  # noqa: E402  (must come after the TMPDIR isolation above)
from ase import Atoms  # noqa: E402
from ase.io import read as ase_read  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

RNEI = 3.0  # first-shell cutoff, must match [AtomicEnvironment] rnei
N_MIN = 3  # smallest coordination probed (eligibility window is n <= 6)
N_TARGETS = 4  # surface atoms of the target species to pool
# Workspace meta-root (Linux box layout); override with PYKMC_ROOT if needed.
ROOT = os.environ.get("PYKMC_ROOT", os.path.expanduser("~/pykmc"))
POTENTIAL = ROOT + "/Clusters/Research/NiFeCr_LKB2017.eam"
SLABS = {
    "NiCr": ROOT + "/structures/NiCr_Ni95_Cr05/slab_NiCr_Ni95_Cr05_1vac.xyz",
    "NiFe": ROOT + "/structures/NiFe_Ni95_Fe05/slab_NiFe_Ni95_Fe05_1vac.xyz",
}


def single_point_energy(atoms: Atoms, elements: list[str]) -> float:
    """Return the eam/alloy potential energy of ``atoms`` (one LAMMPS run 0)."""
    from lammps import lammps

    lmp = lammps(cmdargs=["-log", "none", "-screen", "none", "-nocite"])
    cell = atoms.cell.lengths()
    lmp.command("units metal")
    lmp.command("atom_style atomic")
    lmp.command("boundary p p p")
    lmp.command("region box block 0 {} 0 {} 0 {}".format(cell[0], cell[1], cell[2]))
    lmp.command("create_box {} box".format(len(elements)))
    type_of = {el: i + 1 for i, el in enumerate(elements)}  # alphabetical order
    types = [type_of[s] for s in atoms.get_chemical_symbols()]
    lmp.create_atoms(
        len(atoms),
        None,
        np.array(types, dtype=np.int32),
        atoms.get_positions().ravel(),
    )
    for el in elements:
        lmp.command("mass {} 1.0".format(type_of[el]))  # irrelevant for run 0
    lmp.command("pair_style eam/alloy")
    lmp.command("pair_coeff * * {} {}".format(POTENTIAL, " ".join(elements)))
    lmp.command("run 0")
    pe = float(lmp.get_thermo("pe"))
    lmp.close()
    return pe


def removal_ladder(
    atoms: Atoms, target: int, neighbors: np.ndarray, elements: list[str]
) -> list[tuple[int, float]]:
    """Return [(n, dE_removal)] as the target's neighbours are stripped.

    Neighbours are deleted in-plane-first (ascending |z - z_target|), the
    ladder a receding (100) surface exposes: 8 (terrace) -> 4 (adatom) keeps
    the subsurface square, then digs into it down to N_MIN.
    """
    z_t = atoms.positions[target, 2]
    order = neighbors[np.argsort(np.abs(atoms.positions[neighbors, 2] - z_t))]
    out = []
    for n in range(len(neighbors), N_MIN - 1, -1):
        removed = set(order[n:].tolist())
        keep_with = [i for i in range(len(atoms)) if i not in removed]
        keep_without = [i for i in keep_with if i != target]
        e_with = single_point_energy(atoms[keep_with], elements)
        e_without = single_point_energy(atoms[keep_without], elements)
        out.append((n, e_without - e_with))
    return out


def surface_targets(atoms: Atoms, species: str, tree: cKDTree) -> list[int]:
    """Pick N_TARGETS well-separated top-surface atoms of ``species`` (n=8)."""
    z = atoms.positions[:, 2]
    z_top = z.max()
    picks: list[int] = []
    for i in np.argsort(-z):
        if atoms[i].symbol != species or z_top - z[i] > 1.0:
            continue
        nn = tree.query_ball_point(atoms.positions[i], RNEI)
        if len(nn) - 1 != 8:  # clean terrace site only
            continue
        if all(
            np.linalg.norm(atoms.positions[i] - atoms.positions[j]) > 12.0
            for j in picks
        ):
            picks.append(int(i))
        if len(picks) == N_TARGETS:
            break
    return picks


def main() -> int:
    """Run the ladder + fit for each system and species; print a report."""
    for system, path in SLABS.items():
        atoms = ase_read(path)
        elements = sorted(set(atoms.get_chemical_symbols()))
        tree = cKDTree(atoms.get_positions())
        solute = [e for e in elements if e != "Ni"][0]
        print(
            "== {} ({}): {} atoms, elements {}".format(
                system, path.rsplit("/", 1)[-1], len(atoms), elements
            )
        )
        for species in (solute, "Ni"):
            targets = surface_targets(atoms, species, tree)
            if not targets:
                print("  {}: no clean terrace target found".format(species))
                continue
            pts: list[tuple[int, float]] = []
            for t in targets:
                nn = [
                    j for j in tree.query_ball_point(atoms.positions[t], RNEI) if j != t
                ]
                pts += removal_ladder(atoms, t, np.array(nn), elements)
            ns = np.array([p[0] for p in pts], dtype=float)
            des = np.array([p[1] for p in pts])
            slope, intercept = np.polyfit(ns, des, 1)
            pred = slope * ns + intercept
            ss_res = float(np.sum((des - pred) ** 2))
            ss_tot = float(np.sum((des - des.mean()) ** 2))
            by_n = {int(n): float(np.mean(des[ns == n])) for n in np.unique(ns)}
            print(
                "  {}-host: E_b = {:.3f} eV/bond (intercept {:+.2f} eV, R^2 = {:.4f}, "
                "{} targets x n = {}..8)".format(
                    species, slope, intercept, 1 - ss_res / ss_tot, len(targets), N_MIN
                )
            )
            print(
                "    mean dE_removal(n): {}".format(
                    {n: round(v, 2) for n, v in sorted(by_n.items())}
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
