"""Validate the ``coordination`` classifier against ``cna`` on a Ni95Cr5 slab.

This script cross-checks the coordination-based atomic-environment classifier
against Common Neighbour Analysis (CNA) on the Ni95Cr5 surface-vacancy slab. For
each atom it computes both the CNA label (``"crystal"`` / ``"noncrystal"``) and a
coordination label derived from a first-neighbour count threshold, then reports
how the two sets of ``noncrystal`` atoms overlap. At the FCC-bulk coordination
(12) the coordination classifier reproduces CNA exactly; at a stricter
production threshold (e.g. 8) it flags a strict subset.

The classifiers use only the ``rnei`` neighbour list, so no LAMMPS, pARTn, or
MPI is required.

Run requirement (IMPORTANT)
---------------------------
A plain ``python scripts/validate_coordination_vs_cna.py ...`` puts the script's
directory on ``sys.path``, which makes ``import pykmc`` resolve to the shared
venv's *editable install* (a different checkout that may lack ``coordination``).
That surfaces as ``unexpected keyword argument 'coordination_threshold'``. Always
force this worktree onto the path (root = the worktree directory containing
this ``scripts/`` folder)::

    PYTHONPATH=<worktree-root> python scripts/validate_coordination_vs_cna.py <xyz> [threshold]

Confirm the resolved checkout first::

    PYTHONPATH=. python -c "import pykmc; print(pykmc.__file__)"
    # must contain 'gifted-bardeen-8ba8f3'

Parameters
----------
The ``rnei`` / ``rcut`` / ``threshold`` defaults should match the target
system's ``input.in`` ``[AtomicEnvironment]`` section. For the Ni95Cr5 grey slab
(``Data/Research/Ni_Slab_Alloys/NiCr_Ni95_Cr05_T300_1vac_grey``) that is
``rnei = 3.0``, ``rcut = 8.5``, ``coordination_threshold = 8``. Only ``rnei``
affects the cna/coordination result here; ``rcut`` is set to match the system
but does not influence these labels.

Examples
--------
FCC-bulk parity (threshold 12)::

    PYTHONPATH=. python scripts/validate_coordination_vs_cna.py <xyz> 12

Production threshold (strict subset of CNA)::

    PYTHONPATH=. python scripts/validate_coordination_vs_cna.py <xyz> 8

"""

import sys
from collections import Counter

from pykmc import AtomicEnvironment, NeighborsList, System


def main(
    xyz_path: str,
    threshold: int = 12,
    rnei: float = 3.0,
    rcut: float = 8.5,
) -> None:
    """Compare coordination and CNA non-crystal classifications on a structure.

    Parameters
    ----------
    xyz_path : str
        Path to the extended-XYZ structure to classify.
    threshold : int, optional
        Coordination cut-off; an atom with fewer than ``threshold`` first
        neighbours is labelled ``"noncrystal"``. Defaults to ``12`` (FCC bulk).
    rnei : float, optional
        First-neighbour cut-off radius (Angstrom). Defaults to ``3.0``.
    rcut : float, optional
        Environment cut-off radius (Angstrom). Set to match the target system's
        ``input.in``; it does not affect the cna/coordination labels. Defaults
        to ``8.5``.

    Returns
    -------
    None
        Results are printed to stdout.

    """
    system = System.create_from_file(xyz_path)
    nl = NeighborsList(system, rnei, rcut)

    cna_ids = AtomicEnvironment(
        "cna", nl.neighbors_list["rnei"]
    ).atomic_environment_list
    coord_ids = AtomicEnvironment(
        "coordination",
        nl.neighbors_list["rnei"],
        coordination_threshold=threshold,
    ).atomic_environment_list

    cna_noncrystal = {i for i, e in enumerate(cna_ids) if e == "noncrystal"}
    coord_noncrystal = {i for i, e in enumerate(coord_ids) if e == "noncrystal"}

    print(f"atoms:                   {len(cna_ids)}")
    print(f"CNA noncrystal:          {len(cna_noncrystal)}")
    print(f"coordination noncrystal: {len(coord_noncrystal)}  (threshold={threshold})")
    print(f"in both:                 {len(cna_noncrystal & coord_noncrystal)}")
    print(f"CNA-only:                {len(cna_noncrystal - coord_noncrystal)}")
    print(f"coordination-only:       {len(coord_noncrystal - cna_noncrystal)}")
    print(f"coordination breakdown:  {Counter(coord_ids)}")


if __name__ == "__main__":
    xyz = sys.argv[1]
    thr = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    main(xyz, thr)
