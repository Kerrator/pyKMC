"""Provide functions for different Kinetic Monte Carlo (KMC) algorithms."""

import random
import numpy as np
import math as m


def rejection_free(l_k: list[float] | np.ndarray) -> tuple[int, float]:
    """Select an event index and calculates time step using the rejection-free KMC algorithm.

    Parameters
    ----------
    l_k : list of float or np.ndarray of float
        List or array of individual event rate constants

    Returns
    -------
    tuple[int, float]
        - idx_selected_event : int
            The index of of l_k of the selected event.
        - delta_t : float
            The time step update

    """
    # compute cumulative rate constant
    k_cumulative = [np.sum(l_k[:i]) for i in range(1, len(l_k) + 1)]

    rand = random.random()
    # find event index satisfy ki-1<rand1ktot<ki
    idx_selected_event = np.searchsorted(
        k_cumulative, rand * k_cumulative[-1], side="left"
    )

    # compute associated update time:
    delta_t = -m.log(random.random()) / k_cumulative[-1]
    return idx_selected_event, delta_t, k_cumulative[-1]
