"""Snapshots appended to a ``.traj`` trajectory_output read back as frames.

``ase.io.write(path, atoms, append=True)`` writes a new ULM segment per call
for a ``.traj`` path, and the header's frame count lands at the end of the
file, so ``ase.io.read(path, ":")`` returned no frames.
"""

from typing import Any

import numpy as np
from ase.io import read

from pykmc.kmc import KMC


def test_traj_output_reads_back_every_snapshot(
    config_Cu: Any, system_Cu: Any, tmp_path: Any
) -> None:
    """Two appended snapshots read back as two frames, positions intact."""
    path = tmp_path / "trajkmc.traj"
    config_Cu.control.trajectory_output = str(path)
    sim = KMC(config_Cu)
    sim.system = system_Cu

    first = system_Cu.positions.copy()
    sim._append_snapshot_to_trajectory()
    system_Cu.positions[0] += np.array([0.123456789012, 0.0, 0.0])
    sim._append_snapshot_to_trajectory()

    frames = read(path, ":")
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0].get_positions(), first)
    np.testing.assert_array_equal(frames[1].get_positions(), system_Cu.positions)
