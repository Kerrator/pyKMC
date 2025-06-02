"""Defines the `AtomicEnvironment` class for characterizing and computing local atomic environments."""

import numpy as np
from .environments import cna, graph
from .config import Config


class AtomicEnvironment:
    """Computes and stores atomic environment ID based on a specified style.

    Attributes
    ----------
    config : Config
        configuration object
    style : str
        The atomic environment style (e.g., 'cna', 'graph', 'cna/graph').
    neighbors_list : list[list[int]]
       first neighbors lists
    environment_list : list[list[int]] or None
        Optional. lists of atoms in environments (used for 'graph' or 'cna/graph' styles).
    atomic_environment_list : list
        Computed atomic environment ID for each atom. **Populated during initialization**
        based on the chosen `style`.

    Raises
    ------
    Exception
        If the specified 'AtomicEnvironment' style in `config` is unknown.

    """

    def __init__(
        self,
        config: Config,
        neighbors_list: list[list[int]],
        environment_list: list[list[int]] | None = None,
    ) -> None:
        self.config = config
        self.style = config.atomicenvironment.style
        self.neighbors_list = neighbors_list
        self.environment_list = environment_list

        # Compute the atomic environment ID and store it in self.atomic_environment_list
        match self.style:
            case "cna":
                self.atomic_environment_list = self.compute_cna()
            case "graph":
                self.atomic_environment_list = self.compute_graph(
                    neighbors_list, environment_list
                )
            case "cna/graph":
                self.atomic_environment_list = self.compute_cnagraph(
                    neighbors_list, environment_list
                )
            case _:
                raise Exception("Atomic environment style unknown")

    def compute_cna(self) -> list[str]:
        """See :py:func:`.environments.cna` for details on CNA computation."""
        return cna(self.neighbors_list)

    def compute_graph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[bytes]:
        """See :py:func:`.environment.graph` for detail on Graph Topology computation."""
        return graph(neighbors_list, environment_list)

    def compute_cnagraph(
        self, neighbors_list: list[list[int]], environment_list: list[list[int]]
    ) -> list[str | bytes]:
        """Compute CNA and then Graph Topology for all atoms that have a non cristalline environment.

        Parameters
        ----------
        neighbors_list : list[list[int]]
            first neighbors lists
        environment_list : list[list[int]]
            Optional. lists of atoms in environments (used for 'graph' or 'cna/graph' styles).

        Returns
        -------
        list[str | bytes]
            atomic environment ID for each atom

        """
        # Compute CNA ID
        list_hash = cna(neighbors_list)

        non_crystal_idx = (
            np.where(np.array(list_hash) == "noncrystal")[0].astype(int).tolist()
        )

        # If radd_cna != None add neighbors of non crystal from cna
        n_neighbors = self.config.atomicenvironment.neighbors_add
        if n_neighbors is not None:
            tmp = []
            for _i in range(n_neighbors):  # Do it recursively
                for idx in non_crystal_idx:
                    tmp += neighbors_list[idx]
            non_crystal_idx += tmp
            non_crystal_idx = list(set(non_crystal_idx))
        # Compute graph topo for all non cristalline atoms

        list_graphs_hash = graph(neighbors_list, environment_list, non_crystal_idx)
        for i, idx in enumerate(non_crystal_idx):
            list_hash[idx] = list_graphs_hash[i]

        return list_hash
