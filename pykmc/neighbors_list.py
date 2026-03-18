"""Manage atomic neighbor lists for an `System` using radial cutoffs."""

from scipy.spatial import cKDTree
from .system import System
import numpy as np


class NeighborsList:
    """Store and manage neighbor lists for atoms in a system.

    Builds neighbor lists and environment lists based on two cutoff radii (`rnei` and `rcut`)

    Attributes
    ----------
    system : System
        The atomic system.
    rnei : float
        First neighbor radial cutoff distance.
    rcut : float
        Environment radial cutoff distance.
    neighbors_list : dict[list[int]]
        Pre-calculated neighbor lists: `{'rnei': [...], 'rcut': [...]}`.

    """

    def __init__(
        self,
        system: System,
        rnei: float,
        rcut: float = None,
        graph_cutoff: dict[str, float] | None = None,
    ) -> None:
        self.system = system
        self.rnei = rnei
        self.rcut = rcut
        self._graph_cutoff = graph_cutoff
        if rcut is not None :
            self.neighbors_list = {"rnei": [], "rcut": []}
        else :
            self.neighbors_list = {"rnei" : []}
        self._build_neighbors_list()

    def _get_pair_cutoff(self, type_i: str, type_j: str) -> float:
        """Look up pair-specific cutoff, falling back to rnei."""
        if self._graph_cutoff is None:
            return self.rnei
        key = "-".join(sorted([type_i, type_j]))
        return self._graph_cutoff.get(key, self.rnei)

    def _build_neighbors_list(self) -> None:
        """Build and populate the neighbor lists.

        Contract:
        - `rnei` excludes the central atom.
        - `rcut` includes the central atom exactly once.
        """
        positions = self.system.positions
        cell_diag = np.array([self.system.cell[0][0], self.system.cell[1][1], self.system.cell[2][2]])
        pbc = self.system.pbc if self.system.pbc is not None else np.array([True, True, True])

        # Determine query radius for rnei — use max of all cutoffs when pair-specific
        if self._graph_cutoff is not None:
            query_rnei = max(self.rnei, max(self._graph_cutoff.values()))
        else:
            query_rnei = self.rnei
        needs_filtering = self._graph_cutoff is not None and self.system.types is not None
        types = self.system.types

        if np.all(pbc):
            # Fully periodic: use boxsize (existing fast path)
            # Wrap positions into [0, box) — cKDTree requires non-negative coords
            wrapped = np.mod(positions, cell_diag)
            tree = cKDTree(wrapped, boxsize=cell_diag.tolist())
            for i in range(len(wrapped)):
                candidates = tree.query_ball_point(wrapped[i], query_rnei)
                if i in candidates:
                    candidates.remove(i)  # don't have self as neighbor
                if needs_filtering:
                    neighbors = []
                    for j in candidates:
                        pair_cutoff = self._get_pair_cutoff(types[i], types[j])
                        # Minimum image distance for PBC
                        delta = wrapped[i] - wrapped[j]
                        delta -= cell_diag * np.round(delta / cell_diag)
                        dist = np.linalg.norm(delta)
                        if dist <= pair_cutoff:
                            neighbors.append(j)
                    self.neighbors_list["rnei"].append(neighbors)
                else:
                    self.neighbors_list["rnei"].append(candidates)
                if self.rcut is not None:
                    neighbors = tree.query_ball_point(wrapped[i], self.rcut)
                    self.neighbors_list["rcut"].append(neighbors)
        else:
            # Mixed PBC: create ghost images in periodic directions
            shifts = [[-1, 0, 1] if pbc[d] else [0] for d in range(3)]
            all_positions = []
            index_map = []
            for sx in shifts[0]:
                for sy in shifts[1]:
                    for sz in shifts[2]:
                        shift_vec = np.array([sx * cell_diag[0], sy * cell_diag[1], sz * cell_diag[2]])
                        all_positions.append(positions + shift_vec)
                        index_map.extend(range(len(positions)))
            all_positions = np.vstack(all_positions)
            index_map = np.array(index_map)
            tree = cKDTree(all_positions)

            n_real = len(positions)
            for i in range(n_real):
                raw = tree.query_ball_point(positions[i], query_rnei)
                if needs_filtering:
                    # Filter by pair-specific cutoff using ghost distances
                    neighbors = []
                    seen = set()
                    for j in raw:
                        real_j = index_map[j]
                        if real_j == i or real_j in seen:
                            continue
                        pair_cutoff = self._get_pair_cutoff(types[i], types[real_j])
                        dist = np.linalg.norm(positions[i] - all_positions[j])
                        if dist <= pair_cutoff:
                            seen.add(real_j)
                            neighbors.append(real_j)
                    self.neighbors_list["rnei"].append(sorted(neighbors))
                else:
                    mapped = sorted(set(index_map[j] for j in raw) - {i})
                    self.neighbors_list["rnei"].append(mapped)
                if self.rcut is not None:
                    raw = tree.query_ball_point(positions[i], self.rcut)
                    mapped = sorted(set(index_map[j] for j in raw))
                    self.neighbors_list["rcut"].append(mapped)

    def get_neighbors(self, cutoff_type: float, idx: int) -> list[int]:
        """Retrieve the neighbor list for a specific atom and cutoff.

        Parameters
        ----------
        cutoff_type : str
            The cutoff type ('rnei' or 'rcut').
        idx : int
            The index of the atom.

        Returns
        -------
        list of int
            Indices of neighboring atoms.

        """
        return self.neighbors_list[cutoff_type][idx]

    def update_neighbors(self, list_atoms: np.ndarray) -> None:
        """Update placeholder for future implementation.

        Parameters
        ----------
        list_atoms : np.ndarray
            list of atoms

        """
        pass
