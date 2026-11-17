"""Replay recorded pARTn searches through the shared search rules.

Each file in ``tests/data/partn_replay`` holds one event search recorded through
``LammpsEngine.partn_search``: what pARTn returned, the other inputs the rules
read, and the result pyKMC built (see the README there). The replay feeds the
recorded pARTn values through the same steps ``partn_search`` runs after pARTn
and checks the result is the recorded one exactly. Neither LAMMPS nor pARTn is
needed.

The recordings were made on ``develop`` 1f633fc, before PR #150 moved ARTn's
move-atom pick to the minimum image. In the four boundary cases the recorded
move atom is the old pick (a bystander); the replay expects the atom with the
largest minimum-image displacement, which the manifest lists, and everything
else as recorded.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from pykmc.result import Err, Ok
from pykmc.search_rules import (
    SearchEndpoints,
    artn_move_atom,
    map_to_full_system,
    search_result,
)

DATA = Path(__file__).parent / "data" / "partn_replay"
MANIFEST = json.loads((DATA / "manifest.json").read_text())
RECORDS = {r["file"]: r for r in MANIFEST["records"]}


class RecordedPartn:
    """Stand-in for a pARTn handle that returns one search's recorded values.

    It logs the order values are read in, to compare with the recorded reads.
    """

    def __init__(self, archive: np.lib.npyio.NpzFile, meta: dict) -> None:
        self.archive = archive
        self.meta = meta
        self.calls: list[str] = []

    def get_error(self) -> tuple[int, str]:
        """Return the recorded error code and message."""
        self.calls.append("get_error")
        code, message = self.meta["err"]
        return code, message

    def extract(self, name: str) -> np.ndarray | np.float64:
        """Return a recorded value, whether or not pyKMC read it when recorded."""
        self.calls.append(name)
        for prefix in ("raw_", "extra_"):
            if prefix + name in self.archive.files:
                value = self.archive[prefix + name]
                return value[()] if value.ndim == 0 else value
        raise KeyError(name)


def replay(archive: np.lib.npyio.NpzFile, meta: dict, artn: RecordedPartn) -> object:
    """Run the steps ``LammpsEngine.partn_search`` runs once pARTn has finished."""
    config = meta["config"]
    positions = archive["in_positions"]
    types = archive["in_types"]

    def read_distances() -> tuple[float, float]:
        return artn.extract("delr_min1"), artn.extract("delr_min2")

    def read_endpoints() -> SearchEndpoints:
        e_saddle = artn.extract("etot_sad")
        e_min1 = artn.extract("etot_min1")
        e_min2 = artn.extract("etot_min2")
        min1 = artn.extract("tau_min1")
        min2 = artn.extract("tau_min2")
        saddle = artn.extract("tau_sad")
        if config["active_volume"]:
            atom_map = archive["atom_map"]
            move_atom = atom_map[artn_move_atom(min1, saddle, archive["in_cell"], True)]
            min1 = map_to_full_system(min1, atom_map, positions)
            min2 = map_to_full_system(min2, atom_map, positions)
            saddle = map_to_full_system(saddle, atom_map, positions)
        else:
            if not meta["in"]["is_orthorhombic"]:
                # LammpsEngine._positions_from_lammps
                Q = archive["Q"]
                min1, min2, saddle = min1 @ Q, min2 @ Q, saddle @ Q
            # LammpsEngine.partn_search: the LAMMPS box as cell vectors
            lx, ly, lz = np.subtract(archive["box_hi"], archive["box_lo"])
            xy, yz, xz = meta["in"]["box_tilt"]
            box = np.array([[lx, 0, 0], [xy, ly, 0], [xz, yz, lz]])
            if not meta["in"]["is_orthorhombic"]:
                box = box @ archive["Q"]
            move_atom = artn_move_atom(min1, saddle, box, meta["in"]["periodicity"])
        return SearchEndpoints(
            e_min1=e_min1,
            e_saddle=e_saddle,
            e_min2=e_min2,
            min1_positions=min1,
            saddle_positions=saddle,
            min2_positions=min2,
            move_atom_index=move_atom,
        )

    return search_result(
        central_atom_index=meta["in"]["central_atom_idx"],
        error=artn.get_error(),
        read_distances=read_distances,
        read_endpoints=read_endpoints,
        delr_threshold=config["eventsearch.delr_thr"],
        types=types,
    )


def test_recordings_cover_every_rule_branch() -> None:
    """Both active-volume modes have each case; the boundary cases were recorded
    with the old pick, which differs from the minimum-image mover."""
    cases = {(r["active_volume"], r["case"]) for r in RECORDS.values()}
    for active_volume in (True, False):
        for case in (
            "partn_failure",
            "rejected",
            "plain",
            "swapped",
            "plain_boundary_bystander",
            "swapped_boundary_bystander",
        ):
            assert (active_volume, case) in cases
    for record in RECORDS.values():
        if record["case"].endswith("boundary_bystander"):
            assert record["result"]["move_atom_index"] == record["raw_pick_atom"]
            assert record["raw_pick_atom"] != record["min_image_largest_mover"]


@pytest.mark.parametrize("name", sorted(RECORDS))
def test_replay_gives_recorded_result(name: str) -> None:
    """The shared rules rebuild the recorded result bit for bit."""
    archive = np.load(DATA / name, allow_pickle=False)
    meta = json.loads(str(archive["meta"]))
    want = meta["result"]
    artn = RecordedPartn(archive, meta)

    got = replay(archive, meta, artn)

    assert artn.calls == meta["calls"]
    if want["kind"] == "err":
        assert isinstance(got, Err)
        info = got.err_value()
        assert info.type.name == want["type"]
        assert info.message == want["message"]
        if want["details"] is None:
            assert info.details is None
        else:
            assert list(info.details) == want["details"]
        assert (info.variables or {}) == want["variables"]
    else:
        assert isinstance(got, Ok)
        out = got.ok_value()
        assert out.central_atom_index == want["central_atom_index"]
        assert out.dE_forward == want["dE_forward"]
        assert out.dE_backward == want["dE_backward"]
        if RECORDS[name]["case"].endswith("boundary_bystander"):
            assert out.move_atom_index == RECORDS[name]["min_image_largest_mover"]
        else:
            assert out.move_atom_index == want["move_atom_index"]
        assert np.array_equal(out.min1_positions, archive["res_min1"])
        assert np.array_equal(out.saddle_positions, archive["res_saddle"])
        assert np.array_equal(out.min2_positions, archive["res_min2"])
        assert np.array_equal(np.asarray(out.types), archive["res_types"])
        assert out.cell is None


def test_rules_import_without_lammps_or_partn() -> None:
    """``pykmc.search_rules`` imports with ``lammps`` and ``pypARTn`` unavailable."""
    code = (
        "import sys; sys.modules['lammps'] = None; sys.modules['pypARTn'] = None; "
        "import pykmc.search_rules"
    )
    subprocess.run(
        [sys.executable, "-c", code], check=True, cwd=Path(__file__).parents[1]
    )
