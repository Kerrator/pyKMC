"""Pure AV command-stream oracle: both entries preserve the live full map."""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.activevolume import active_volume as av


POSITIONS = np.array(
    [[5.0, 5.0, 5.0], [7.4, 5.0, 5.0], [5.5, 7.7, 5.0], [14.0, 14.0, 14.0]]
)
CELL = np.eye(3) * 40.0
TYPES = ("Ni", "Ni", "Ni", "Cu")


class RecordingLAMMPS:
    """Records atom creation; simulates pair_coeff overriding emitted masses."""

    def __init__(self):
        self.natoms = 0
        self.types = ()
        self.positions = None

    def create_atoms(self, n, ids, types, x):
        self.natoms = int(n)
        self.types = tuple(int(v) for v in types)
        self.positions = np.asarray(x, dtype=float).reshape(n, 3).copy()

    def get_natoms(self):
        return self.natoms

    def scatter_atoms(self, name, dtype, count, data):
        self.positions = np.array(list(data), dtype=float).reshape(-1, 3)

    def extract_compute(self, *args):
        return -1.0  # Protocol-only: not a force/energy result.


class RecordingEngine:
    def __init__(self, full):
        self.full_system = full
        self.commands = []
        self.mass_slots = {}
        self.ntypes = 0
        self.rank = 0
        self.lmp = RecordingLAMMPS()

    def command(self, command):
        self.commands.append(command)
        words = command.split()
        if words[0] == "clear":
            self.mass_slots = {}
            self.ntypes = 0
        elif words[0] == "create_box":
            self.ntypes = int(words[1])
        elif words[0] == "mass":
            self.mass_slots[int(words[1])] = float(words[2])
        elif words[0] == "pair_coeff":
            # Represents file-based potentials that reset masses. Values are
            # deliberately different from the authoritative source descriptor.
            self.mass_slots = {i: 100.0 + i for i in range(1, self.ntypes + 1)}


def configuration(species=("Ni", "Cu", "Fe")):
    return SimpleNamespace(
        lammps=SimpleNamespace(
            pair_style="eam/alloy",
            pair_coeff="* * unused.eam " + " ".join(species),
            min_style="cg",
            frz_min="1e-8 1e-8 1 10",
        ),
        activevolume=SimpleNamespace(ract=6.0, rmov=6.0, AV_debug=False),
    )


def enter(operation, engine, config):
    if operation == "search":
        return av.partn_search_AV(engine, config, 0, POSITIONS.copy(), CELL, TYPES)
    _, atom_map, central = av.partn_refine_AV(
        engine,
        config,
        0,
        POSITIONS.copy(),
        CELL,
        TYPES,
        np.array([0, 1, 2]),
        POSITIONS[:3].copy(),
    )
    return atom_map, central


@pytest.mark.parametrize("operation", ["search", "refine"])
@pytest.mark.parametrize(
    "species,masses,expected_atom_type",
    [
        (("Ni", "Cu", "Fe"), (61.0, 65.0, 57.0), 1),
        (("Cu", "Ni"), (63.546, 58.6934), 2),
    ],
)
def test_crop_uses_authoritative_order_absent_slots_and_post_potential_masses(
    operation, species, masses, expected_atom_type
):
    full = SimpleNamespace(
        types=TYPES,
        species=species,
        masses=masses,
        cell=CELL.copy(),
        pbc=(True, True, True),
        physics=None,
    )
    engine = RecordingEngine(full)
    atom_map, central = enter(operation, engine, configuration(species))
    assert tuple(atom_map) == (0, 1, 2)
    assert tuple(central) == (1,)
    assert engine.full_system is full
    assert engine.ntypes == len(species)
    assert engine.lmp.types == (expected_atom_type,) * 3
    assert engine.mass_slots == dict(enumerate(masses, start=1))
    assert np.array_equal(engine.lmp.positions, POSITIONS[:3])
    potential_index = next(
        i for i, cmd in enumerate(engine.commands) if cmd.startswith("pair_coeff ")
    )
    final_mass_index = max(
        i for i, cmd in enumerate(engine.commands) if cmd.startswith("mass ")
    )
    assert final_mass_index > potential_index, (
        "pair_coeff must not undo authoritative masses"
    )
    assert "unfix 1" in engine.commands
    if operation == "refine":
        assert (
            "unfix f_core" in engine.commands and "group core delete" in engine.commands
        )


@pytest.mark.parametrize("operation", ["search", "refine"])
@pytest.mark.parametrize(
    "species,masses",
    [
        (("Ni", "Ni"), (61.0, 62.0)),
        (("Ni", "Cu"), (61.0,)),
        (("Ni", "Cu"), (61.0, -1.0)),
        (("Ni", "Fe"), (61.0, 57.0)),
    ],
)
def test_bad_authoritative_map_rejects_before_clear(operation, species, masses):
    full = SimpleNamespace(
        types=TYPES,
        species=species,
        masses=masses,
        cell=CELL.copy(),
        pbc=(True, True, True),
        physics=None,
    )
    engine = RecordingEngine(full)
    with pytest.raises((ValueError, TypeError)):
        enter(operation, engine, configuration())
    assert engine.commands == []
    assert engine.lmp.natoms == 0
