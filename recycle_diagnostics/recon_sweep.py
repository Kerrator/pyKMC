"""Sweep: reconstruct a Ni vacancy-hop event recycled onto a PERTURBED shell.

Models the real recycle condition: a generic event's saddle shell is mapped (with
registration error) onto a site whose real minima are min1/min2. We perturb the
transplanted saddle shell by sigma (Angstrom, gaussian) and reconstruct through the
REAL manager (full-cell minimize), reporting ok / error-type per sigma.

Run: mpirun -n 2 python /tmp/recon_sweep.py <repeat> <sigma1,sigma2,...>
"""
import sys
import numpy as np

from pykmc import System, Config
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc.reconstruction import Reconstruction


def build_ni_fcc_vacancy(repeat, a=3.52):
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]) * a
    pos = []
    for i in range(repeat):
        for j in range(repeat):
            for k in range(repeat):
                shift = np.array([i, j, k]) * a
                for at in basis:
                    pos.append(at + shift)
    pos = np.array(pos)
    L = repeat * a
    centre = np.array([L / 2, L / 2, L / 2])
    vac = int(np.argmin(np.linalg.norm(pos - centre, axis=1)))
    vac_pos = pos[vac].copy()
    keep = np.ones(len(pos), bool)
    keep[vac] = False
    pos = pos[keep]
    system = System()
    system.positions = pos
    system.types = ["Ni"] * len(pos)
    system.cell = np.diag([L, L, L])
    system.pbc = np.array([True, True, True])
    system.index = np.arange(len(pos))
    return system, vac_pos, L


def main():
    repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    sigmas = [float(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else [0.0, 0.2, 0.4, 0.6]
    rcut = 6.5
    config = Config.from_ini_file("./tests/data/input.in")
    thr = config.psr.matching_score_thr

    system, vac_pos, L = build_ni_fcc_vacancy(repeat)

    factory = ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True)
    manager = factory.launch()
    if manager is None:
        return
    try:
        manager.initialize_sessions(config, system)
        manager.global_initialize_parameters()
        manager.global_initialize_system(system)
        manager.global_initialize_potential(config)

        min1_full, e1 = manager.global_minimize_with_results(config, positions=system.positions, types=system.types)
        min1_full = np.asarray(min1_full)
        d = min1_full - vac_pos
        d -= L * np.round(d / L)
        A = int(np.argmin(np.linalg.norm(d, axis=1)))
        A_pos = min1_full[A].copy()
        delta = vac_pos - A_pos
        delta -= L * np.round(delta / L)
        min2_start = min1_full.copy()
        min2_start[A] = A_pos + delta
        min2_full, e2 = manager.global_minimize_with_results(config, positions=min2_start, types=system.types)
        min2_full = np.asarray(min2_full)
        saddle_full = min1_full.copy()
        saddle_full[A] = A_pos + 0.5 * delta
        dd = min1_full - A_pos
        dd -= L * np.round(dd / L)
        neighbors = np.where(np.linalg.norm(dd, axis=1) <= rcut)[0]
        supposed_min1 = min1_full[neighbors]
        supposed_min2 = min2_full[neighbors]

        print(f"# repeat={repeat} natoms={len(system.positions)} nbrs={len(neighbors)} "
              f"e1={e1:.3f} e2={e2:.3f} thr={thr}", flush=True)
        for sigma in sigmas:
            rng = np.random.default_rng(0)
            sad = saddle_full.copy()
            if sigma > 0:
                sad[neighbors] = sad[neighbors] + rng.normal(0.0, sigma, (len(neighbors), 3))
            recon = Reconstruction(config, manager, types=system.types)
            res = recon.reconstruct(supposed_min1, supposed_min2, sad, system.cell, thr, neighbors)
            if res.is_ok():
                print(f"  sigma={sigma:.2f}  OK", flush=True)
            else:
                ev = res.err_value()
                extra = ev.variables if getattr(ev, "variables", None) else ""
                print(f"  sigma={sigma:.2f}  FAIL {ev.type}  {extra}", flush=True)
    finally:
        manager.close_all()


if __name__ == "__main__":
    main()
