"""Compare full-cell vs freeze-outer reconstruction at the red points.

For each (repeat, sigma): build the Ni vacancy-hop event, perturb the transplanted
saddle shell by sigma, then reconstruct two ways:
  - full-cell  : manager.global_minimize_with_results            (current code)
  - freeze-out : manager.global_minimize_freeze_outer_with_results (prototype fix)
Reports OK / failure for each, so we can confirm the fix turns red -> green.

Run: mpirun -n 2 python /tmp/recon_compare.py <repeat> <sigma1,sigma2,...>
"""
import sys
import copy
import numpy as np
import ase.geometry

from pykmc import System, Config
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc.reconstruction import Reconstruction
from pykmc.utils.geometry import push_towards, compute_delr

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
    vac_pos = pos[vac].copy()
    keep = np.ones(len(pos), bool); keep[vac] = False
    pos = pos[keep]
    s = System()
    s.positions = pos; s.types = ["Ni"] * len(pos)
    s.cell = np.diag([L, L, L]); s.pbc = np.array([True, True, True]); s.index = np.arange(len(pos))
    return s, vac_pos, L


def reconstruct_freeze_outer(manager, config, pf, m1, m2, saddle, cell, thr, nbrs, A, rmov, types):
    tmp = copy.deepcopy(saddle)
    tmp[nbrs] = push_towards(saddle[nbrs], m1, fraction=pf, cell=cell)
    try:
        p1, _ = manager.global_minimize_freeze_outer_with_results(config, positions=tmp, types=types, central_atom=A, rmov=rmov)
    except RuntimeError as e:
        return ("MINIMIZE_FAILED", str(e)[:60])
    t1 = ase.geometry.wrap_positions(np.asarray(p1), cell, pbc=True)
    d1 = compute_delr(m1, t1[nbrs], cell)
    if d1 > thr:
        return ("INVALID_MIN1", round(float(d1), 3))
    tmp[nbrs] = push_towards(saddle[nbrs], m2, fraction=pf, cell=cell)
    try:
        p2, _ = manager.global_minimize_freeze_outer_with_results(config, positions=tmp, types=types, central_atom=A, rmov=rmov)
    except RuntimeError as e:
        return ("MINIMIZE_FAILED2", str(e)[:60])
    t2 = ase.geometry.wrap_positions(np.asarray(p2), cell, pbc=True)
    d2 = compute_delr(m2, t2[nbrs], cell)
    if d2 > thr:
        return ("INVALID_MIN2", round(float(d2), 3))
    return ("OK", (round(float(d1), 3), round(float(d2), 3)))


def main():
    repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    sigmas = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.5]
    config = Config.from_ini_file("./tests/data/input.in")
    thr = config.psr.matching_score_thr
    pf = float(config.reconstruction.push_fraction)
    s, vac_pos, L = build(repeat)

    manager = ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True).launch()
    if manager is None:
        return
    try:
        manager.initialize_sessions(config, s)
        manager.global_initialize_parameters()
        manager.global_initialize_system(s)
        manager.global_initialize_potential(config)
        m1f, _ = manager.global_minimize_with_results(config, positions=s.positions, types=s.types)
        m1f = np.asarray(m1f)
        d = m1f - vac_pos; d -= L * np.round(d / L)
        A = int(np.argmin(np.linalg.norm(d, axis=1)))
        Apos = m1f[A].copy()
        delta = vac_pos - Apos; delta -= L * np.round(delta / L)
        m2s = m1f.copy(); m2s[A] = Apos + delta
        m2f, _ = manager.global_minimize_with_results(config, positions=m2s, types=s.types)
        m2f = np.asarray(m2f)
        saddle = m1f.copy(); saddle[A] = Apos + 0.5 * delta
        dd = m1f - Apos; dd -= L * np.round(dd / L)
        nbrs = np.where(np.linalg.norm(dd, axis=1) <= RCUT)[0]
        m1 = m1f[nbrs]; m2 = m2f[nbrs]

        print(f"# repeat={repeat} natoms={len(s.positions)} nbrs={len(nbrs)} thr={thr} rmov={RCUT}", flush=True)
        for sigma in sigmas:
            rng = np.random.default_rng(0)
            sad = saddle.copy()
            if sigma > 0:
                sad[nbrs] = sad[nbrs] + rng.normal(0.0, sigma, (len(nbrs), 3))
            # full-cell (current)
            r_full = Reconstruction(config, manager, types=s.types).reconstruct(m1, m2, sad, s.cell, thr, nbrs)
            if r_full.is_ok():
                full = "OK"
            else:
                ev = r_full.err_value()
                full = f"{ev.type.name}({ev.variables})"
            # freeze-outer (prototype fix)
            fo = reconstruct_freeze_outer(manager, config, pf, m1, m2, sad, s.cell, thr, nbrs, A, RCUT, s.types)
            print(f"  sigma={sigma:.2f}  full-cell={full:40s}  freeze-outer={fo}", flush=True)
    finally:
        manager.close_all()


if __name__ == "__main__":
    main()
