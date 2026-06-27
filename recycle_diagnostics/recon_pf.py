"""push_fraction sensitivity at a red point: does changing push_fraction recover min1?

Run: PYTHONPATH=<worktree> mpirun -n 2 python /tmp/recon_pf.py <repeat> <sigma> <pf1,pf2,...>
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
    pos = np.array(pos); L = repeat * a
    vac = int(np.argmin(np.linalg.norm(pos - L / 2, axis=1)))
    vp = pos[vac].copy(); keep = np.ones(len(pos), bool); keep[vac] = False
    pos = pos[keep]
    s = System(); s.positions = pos; s.types = ["Ni"] * len(pos)
    s.cell = np.diag([L, L, L]); s.pbc = np.array([True, True, True]); s.index = np.arange(len(pos))
    return s, vp, L


def main():
    repeat = int(sys.argv[1]); sigma = float(sys.argv[2])
    pfs = [float(x) for x in sys.argv[3].split(",")]
    config = Config.from_ini_file("./tests/data/input.in")
    thr = config.psr.matching_score_thr
    s, vp, L = build(repeat)
    mgr = ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True).launch()
    if mgr is None:
        return
    try:
        mgr.initialize_sessions(config, s); mgr.global_initialize_parameters()
        mgr.global_initialize_system(s); mgr.global_initialize_potential(config)
        m1f, _ = mgr.global_minimize_with_results(config, positions=s.positions, types=s.types)
        m1f = np.asarray(m1f)
        d = m1f - vp; d -= L * np.round(d / L); A = int(np.argmin(np.linalg.norm(d, axis=1)))
        Ap = m1f[A].copy(); delta = vp - Ap; delta -= L * np.round(delta / L)
        m2s = m1f.copy(); m2s[A] = Ap + delta
        m2f, _ = mgr.global_minimize_with_results(config, positions=m2s, types=s.types); m2f = np.asarray(m2f)
        saddle = m1f.copy(); saddle[A] = Ap + 0.5 * delta
        dd = m1f - Ap; dd -= L * np.round(dd / L); nbrs = np.where(np.linalg.norm(dd, axis=1) <= RCUT)[0]
        m1 = m1f[nbrs]; m2 = m2f[nbrs]
        rng = np.random.default_rng(0); sad = saddle.copy()
        sad[nbrs] = sad[nbrs] + rng.normal(0.0, sigma, (len(nbrs), 3))
        print(f"# repeat={repeat} natoms={len(s.positions)} sigma={sigma} thr={thr}", flush=True)
        for pf in pfs:
            try:
                config.reconstruction.push_fraction = pf
            except Exception:
                config = config.model_copy(update={"reconstruction": config.reconstruction.model_copy(update={"push_fraction": pf})})
            r = Reconstruction(config, mgr, types=s.types).reconstruct(m1, m2, sad, s.cell, nbrs)
            if r.is_ok():
                print(f"  push_fraction={pf:.2f}  OK", flush=True)
            else:
                ev = r.err_value()
                print(f"  push_fraction={pf:.2f}  FAIL {ev.type.name} {ev.variables}", flush=True)
    finally:
        mgr.close_all()


if __name__ == "__main__":
    main()
