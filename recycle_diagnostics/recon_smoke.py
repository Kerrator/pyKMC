"""Smoke: drive a REAL Reconstruction through the manager for a Ni vacancy hop.

Run: mpirun -n 2 python /tmp/recon_smoke.py <repeat>
Builds an R x R x R Ni FCC cell, removes one atom (vacancy), relaxes it, constructs
a nearest-neighbour vacancy-hop event (min1 / saddle=midpoint / min2), then calls
Reconstruction.reconstruct() and reports Ok / delr.
"""
import sys
import numpy as np

from pykmc import System, Config
from pykmc.enginemanager.lmpi.pool import ManagerFactory
from pykmc.reconstruction import Reconstruction


def build_ni_fcc_vacancy(repeat: float, a: float = 3.52):
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
    # remove a central-ish atom as the vacancy
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
    repeat = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    rcut = 6.5
    config = Config.from_ini_file("./tests/data/input.in")
    thr = config.psr.matching_score_thr

    system, vac_pos, L = build_ni_fcc_vacancy(repeat)

    factory = ManagerFactory(n_sessions=1, use_rank_0=False, has_global=True)
    manager = factory.launch()
    if manager is None:
        return  # engine ranks exit here
    try:
        print("[driver] sessions launched; initializing...", flush=True)
        manager.initialize_sessions(config, system)
        manager.global_initialize_parameters()
        manager.global_initialize_system(system)
        manager.global_initialize_potential(config)
        print("[driver] initialized; minimizing min1...", flush=True)

        # min1: relax the vacancy cell
        min1_full, e1 = manager.global_minimize_with_results(
            config, positions=system.positions, types=system.types
        )
        min1_full = np.asarray(min1_full)

        # hopping atom A = nearest atom to the vacancy
        d = min1_full - vac_pos
        d -= L * np.round(d / L)
        A = int(np.argmin(np.linalg.norm(d, axis=1)))
        A_pos = min1_full[A].copy()

        # min2: move A into the vacancy site, relax
        min2_start = min1_full.copy()
        # unwrap vac relative to A then place A there
        delta = vac_pos - A_pos
        delta -= L * np.round(delta / L)
        min2_start[A] = A_pos + delta
        min2_full, e2 = manager.global_minimize_with_results(
            config, positions=min2_start, types=system.types
        )
        min2_full = np.asarray(min2_full)

        # saddle: A at the midpoint A_pos -> vac
        saddle_full = min1_full.copy()
        saddle_full[A] = A_pos + 0.5 * delta

        # neighbours = rcut ball around A (in min1 geometry)
        dd = min1_full - A_pos
        dd -= L * np.round(dd / L)
        neighbors = np.where(np.linalg.norm(dd, axis=1) <= rcut)[0]

        supposed_min1 = min1_full[neighbors]
        supposed_min2 = min2_full[neighbors]

        recon = Reconstruction(config, manager, types=system.types)
        result = recon.reconstruct(
            supposed_min1, supposed_min2, saddle_full, system.cell, thr, neighbors
        )
        print(f"[repeat={repeat} natoms={len(system.positions)} nbrs={len(neighbors)} "
              f"e1={e1:.4f} e2={e2:.4f}] reconstruct ok={result.is_ok()}", flush=True)
        if not result.is_ok():
            print("   ERR:", result.err_value().type, result.err_value().message, flush=True)
    finally:
        manager.close_all()


if __name__ == "__main__":
    main()
