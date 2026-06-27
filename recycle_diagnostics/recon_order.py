"""Decisive repro: does neighbor-ORDER scrambling reproduce the 32k 'Lost atoms'
(RECONSTRUCTION_MINIMIZE_FAILED) mode, with an otherwise CLEAN self-consistent event?

Background. The real 32k recycle failures are 99.6% RECONSTRUCTION_MINIMIZE_FAILED
(LAMMPS 'Lost atoms', always in the min1 minimize), NOT mis-lands. The competing
diagnoses are:
  (A) full-cell minimize fragility at scale  -> freeze-outer fix
  (B) neighbor-ORDER scatter: stored saddle is in refinement-neighbor order, but
      min1 is read in live-neighbor order; cKDTree.query_ball_point returns the same
      atoms in a different traversal order between steps, so push() pairs row i of the
      saddle (atom a) with row i of min1 (atom b != a) -> overlaps -> Lost atoms.

This script holds the event perfectly self-consistent (no sigma) and only permutes the
order in which the saddle shell is presented relative to min1. If a permutation turns a
green reconstruction RED with MINIMIZE_FAILED (Lost atoms), diagnosis (B) is the cause
and freeze-outer (which relaxes the shell where the scatter lives) cannot be the fix.

Run: mpirun -n 2 python recycle_diagnostics/recon_order.py <repeat>
"""
import sys
import numpy as np

from pykmc import System, Config
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc.reconstruction import Reconstruction

RCUT = 6.5


def build(repeat, a=3.52):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    pos = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                for at in basis:
                    pos.append(at + np.array([i, j, k]) * a)
    pos = np.array(pos)
    L = repeat * a
    vac = int(np.argmin(np.linalg.norm(pos - L / 2, axis=1)))
    vp = pos[vac].copy()
    keep = np.ones(len(pos), bool)
    keep[vac] = False
    pos = pos[keep]
    s = System()
    s.positions = pos
    s.types = ["Ni"] * len(pos)
    s.cell = np.diag([L, L, L])
    s.pbc = np.array([True, True, True])
    s.index = np.arange(len(pos))
    return s, vp, L


def main():
    repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    config = Config.from_ini_file("./tests/data/input.in")
    thr = config.psr.matching_score_thr
    s, vp, L = build(repeat)

    mgr = ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True).launch()
    if mgr is None:
        return
    try:
        mgr.initialize_sessions(config, s)
        mgr.global_initialize_parameters()
        mgr.global_initialize_system(s)
        mgr.global_initialize_potential(config)

        m1f, _ = mgr.global_minimize_with_results(config, positions=s.positions, types=s.types)
        m1f = np.asarray(m1f)
        d = m1f - vp; d -= L * np.round(d / L)
        A = int(np.argmin(np.linalg.norm(d, axis=1)))
        Ap = m1f[A].copy()
        delta = vp - Ap; delta -= L * np.round(delta / L)
        m2s = m1f.copy(); m2s[A] = Ap + delta
        m2f, _ = mgr.global_minimize_with_results(config, positions=m2s, types=s.types)
        m2f = np.asarray(m2f)
        saddle = m1f.copy(); saddle[A] = Ap + 0.5 * delta
        dd = m1f - Ap; dd -= L * np.round(dd / L)
        nbrs = np.where(np.linalg.norm(dd, axis=1) <= RCUT)[0]
        m1 = m1f[nbrs]; m2 = m2f[nbrs]
        n = len(nbrs)
        # locate hop atom A within the shell rows
        a_row = int(np.where(nbrs == A)[0][0])

        print(f"# repeat={repeat} natoms={len(s.positions)} nbrs={n} thr={thr} A_row={a_row}", flush=True)

        rng = np.random.default_rng(0)

        def make_perm(kind):
            p = np.arange(n)
            if kind == "identity":
                return p
            if kind == "reverse":
                return p[::-1].copy()
            if kind == "adjacent-swap":  # swap two neighbouring shell rows
                p = p.copy(); p[1], p[2] = p[2], p[1]; return p
            if kind == "swap-hop":  # swap the hop atom row with a far shell row
                p = p.copy(); far = int(np.argmax(np.linalg.norm(m1 - m1[a_row], axis=1)))
                p[a_row], p[far] = p[far], p[a_row]; return p
            if kind == "random":
                p = p.copy(); rng.shuffle(p); return p
            raise ValueError(kind)

        for kind in ["identity", "adjacent-swap", "swap-hop", "reverse", "random"]:
            perm = make_perm(kind)
            # Emulate kmc: the shell SADDLE values are presented in a permuted (stored)
            # order while min1 is read in the live order. saddle_scr[nbrs[i]] holds the
            # clean saddle of atom nbrs[perm[i]].
            saddle_scr = saddle.copy()
            saddle_scr[nbrs] = saddle[nbrs][perm]
            # max displacement that push() will demand for min1 stage (overlap proxy)
            pushed = saddle_scr[nbrs] + config.reconstruction.push_fraction * (m1 - saddle_scr[nbrs])
            # min pairwise distance among pushed shell atoms (overlap detector)
            from scipy.spatial.distance import pdist
            dmin = float(pdist(pushed).min()) if n > 1 else np.nan
            r = Reconstruction(config, mgr, types=s.types).reconstruct(m1, m2, saddle_scr, s.cell, nbrs)
            if r.is_ok():
                tag = "OK"
            else:
                ev = r.err_value()
                tag = f"FAIL {ev.type.name} {ev.variables}"
            print(f"  perm={kind:14s} min_pushed_dist={dmin:5.2f}A  {tag}", flush=True)
    finally:
        mgr.close_all()


if __name__ == "__main__":
    main()
