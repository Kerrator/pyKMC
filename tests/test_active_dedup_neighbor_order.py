"""Active-table dedup must align rows by their stored crop identities.

``ActiveEventTable.remove_duplicates`` compares two rows' ``saddle_positions``
to decide whether they encode the same physical event. Each row's positions are
ordered by the crop captured at event time, so a recycled row keeps its
event-time ordering while a fresh row follows the current
:class:`NeighborsList`: two rows describing the same event can carry the same
atoms in a **different order**, and after the system moved even a different
atom **set**. Comparing the arrays element-wise therefore compares
non-corresponding atoms -- or raises a numpy broadcast error outright when the
two crops have different sizes.

Each row's stable atom identities are recorded at refinement. The htst/rpa
styles keep them in the row (``crop_atom_ids``); the constant style keeps the
legacy table schema and holds them beside it, so dedup reaches both through
``ActiveEventTable.row_crop_ids``. This suite covers the constant style, where
the identities are the ones that used to be ignored: permuted-but-identical
events dedup, distinct events that would alias under the wrong ordering stay
distinct, identical-ordering rows behave exactly as before, and rows whose
crops have different sizes or disjoint members keep both rows.
"""

from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest

import pykmc
from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import ActiveEventTable
from pykmc.result import EventRefinementOutput

_CELL = np.diag([20.0, 20.0, 20.0])

# Stable atom identities are not array positions: id = 100 + position, so a row
# that confused the two would not line up.
_ID_OFFSET = 100

# A canonical 4-atom saddle geometry, ordered by atoms [0, 1, 2, 3].
_SADDLE = np.array(
    [
        [10.0, 10.0, 10.0],
        [12.5, 10.0, 10.0],
        [10.0, 12.0, 10.0],
        [10.0, 10.0, 12.0],
    ]
)


@pytest.fixture
def config() -> Config:
    """Real constant-style config; only the PSR tolerance matters here."""
    root = Path(pykmc.__file__).resolve().parent.parent
    cfg = Config.from_ini_file(str(root / "tests/data/input.in"))
    cfg.rateconstant = RateConstantConfig(style="constant", k0=7.0)
    cfg.psr.matching_score_thr = 0.1
    return cfg


def _event(
    atom_index: int,
    saddle_positions: np.ndarray,
    crop_positions: "list[int] | None",
    energy_barrier: float = 0.5,
    num_reference_event: int = 0,
) -> EventRefinementOutput:
    """One refined event whose crop is given by array positions."""
    saddle = np.asarray(saddle_positions, dtype=float)
    return EventRefinementOutput(
        central_atom_index=atom_index,
        saddle_positions=saddle,
        E_saddle=0.1,
        min2_positions=saddle.copy(),
        dE_forward=energy_barrier,
        num_reference_event=num_reference_event,
        refined="T",
        crop_atom_ids=(
            None
            if crop_positions is None
            else tuple(_ID_OFFSET + int(i) for i in crop_positions)
        ),
    )


def _table(config: Config, events: "list[EventRefinementOutput]") -> ActiveEventTable:
    table = ActiveEventTable(config)
    for event in events:
        table.add_events(event)
    assert not table.uses_prefactors
    return table


def _neighbors_list(current: "dict[int, list[int]]", n_atoms: int = 8) -> Mock:
    """Stand-in carrying the identity map and the *current* rcut orderings.

    ``current`` is what ``get_neighbors('rcut', atom)`` returns now, which is
    not the order the rows' crops were stored in.
    """
    neighbors_list = Mock()
    neighbors_list.system.index = np.arange(n_atoms) + _ID_OFFSET
    neighbors_list.system.pbc = True
    neighbors_list.get_neighbors.side_effect = lambda style, atom: list(
        current[int(atom)]
    )
    return neighbors_list


def test_size_mismatched_crops_no_crash_both_kept(config: Config) -> None:
    """A recycled row whose stored crop size differs must not crash.

    This is the loud recycling failure mode: a recycled row keeps the crop
    captured at event time while a fresh row on the same central atom reflects
    the current neighbour list, and after membership drift the two
    ``saddle_positions`` arrays have different lengths. Feeding both straight
    into ``compute_delr`` raises a numpy broadcast ``ValueError``. Differing
    crops on the same central atom mean a different local environment, so both
    rows must simply be kept.
    """
    table = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, _SADDLE[:3], [0, 1, 2]),
        ],
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_permuted_ordering_same_event_is_deduplicated(config: Config) -> None:
    """Same physical event with permuted crop ordering -> one duplicate.

    Row A stores atoms [0, 1, 2, 3]; row B stores the SAME atoms in order
    [3, 2, 1, 0] with its ``saddle_positions`` permuted to match. Physically
    identical, so exactly one row must survive.

    Comparing the arrays element-wise instead differences ``saddle[0]`` of A
    (atom 0) against ``saddle[0]`` of B (atom 3). That yields a large ``delr``
    and the true duplicate is KEPT, its rate double-counted in the selection.
    """
    perm = [3, 2, 1, 0]
    table = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, _SADDLE[perm], perm),
        ],
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 1


def test_permuted_ordering_distinct_events_stay_distinct(config: Config) -> None:
    """Two distinct events that would alias under a raw permuted compare.

    Row A (atoms [0, 1, 2, 3]) moves atom 1. Row B stores its crop as
    [1, 0, 2, 3] but is a genuinely different saddle (a different atom moved).
    A naive element-wise compare of the permuted arrays could coincidentally
    line up the moved coordinates; the identity-aligned compare keeps both.
    """
    saddle_b = _SADDLE.copy()
    saddle_b[0] = [10.0, 10.0, 10.0]  # atom 0 at rest
    saddle_b[1] = [12.0, 10.0, 10.0]  # atom 1 at rest (no +0.5 A hop)
    saddle_b[2] = [10.0, 13.0, 10.0]  # atom 2 moved instead
    perm = [1, 0, 2, 3]
    table = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, saddle_b[perm], perm),
        ],
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_identical_ordering_matches_positional_behaviour(config: Config) -> None:
    """Identical-ordering duplicate/non-duplicate outcomes are unchanged.

    With identical crop orderings the aligned compare reduces to the original
    element-wise ``compute_delr``, so the accept/reject verdict must match the
    positional code exactly.
    """
    # (a) identical geometry, identical ordering -> duplicate removed.
    duplicate = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, _SADDLE, [0, 1, 2, 3]),
        ],
    )
    duplicate.remove_duplicates(_CELL)
    assert len(duplicate.table) == 1

    # (b) different geometry, identical ordering -> both kept.
    saddle_far = _SADDLE.copy()
    saddle_far[1] = [15.0, 10.0, 10.0]  # well beyond matching_score_thr
    distinct = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, saddle_far, [0, 1, 2, 3]),
        ],
    )
    distinct.remove_duplicates(_CELL)
    assert len(distinct.table) == 2


def test_legacy_rows_without_identities_stay_positional(config: Config) -> None:
    """Rows that stored no identities keep the legacy positional compare.

    A constant row refined before identities were recorded (an old restart
    pickle) carries none, and the equal-size positional compare is all dedup
    can do for it. Identity-aligned dedup is therefore only as good as the
    identities on the rows: recycled legacy rows keep the ordering assumption.
    """
    perm = [3, 2, 1, 0]
    table = _table(
        config,
        [
            _event(1, _SADDLE, None),
            _event(1, _SADDLE[perm], None),
        ],
    )
    table.remove_duplicates(_CELL)
    # Positional compare of the permuted arrays: a large delr, so the true
    # duplicate survives.
    assert len(table.table) == 2


def test_disjoint_crops_same_central_atom_kept(config: Config) -> None:
    """Same central atom but disjoint crops -> different environment.

    A recycled row whose membership drifted after the system moved must not be
    treated as a duplicate of a fresh row on the same central atom.
    """
    table = _table(
        config,
        [
            _event(1, _SADDLE, [0, 1, 2, 3]),
            _event(1, _SADDLE, [4, 5, 6, 7]),
        ],
    )
    table.remove_duplicates(_CELL)
    assert len(table.table) == 2


def test_symmetric_events_permuted_ordering_deduplicated(config: Config) -> None:
    """Part 2: symmetric events on different central atoms, permuted ordering.

    The symmetric pass only fires for a ``num_reference_event`` that appears
    more than once on some single central atom, so this scenario has two rows
    on atom 1 (with distinct barriers, keeping part 1 from touching them) plus
    a cross-atom duplicate on atom 2. Over their shared crop the atom-2 row
    matches the first atom-1 row; the stored orderings differ, so the atom-2
    row is only detected as a duplicate once alignment uses the stored
    identities rather than the current ``get_neighbors('rcut', ...)``
    ordering, which scatters coordinates onto the wrong atoms for a recycled
    row: under the orderings below that compare misses the duplicate.
    """
    # Shared atoms {1, 2, 3} carry identical positions in the matching rows; the
    # non-shared atom differs. central_atom1 (=1) is a member of the atom-2 crop.
    shared_pos = {
        1: [12.0, 10.0, 10.0],
        2: [10.0, 12.0, 10.0],
        3: [10.0, 10.0, 12.0],
    }
    # Row A0 on atom 1: crop [1, 2, 3, 0], extra atom 0.
    crop_a0 = [1, 2, 3, 0]
    pos_a0 = np.array([shared_pos[1], shared_pos[2], shared_pos[3], [8.0, 8.0, 8.0]])
    # Row A1 on atom 1: a genuinely different saddle + different barrier so part
    # 1 leaves it alone; only present to make num_ref=7 fire the symmetric pass.
    pos_a1 = pos_a0.copy()
    pos_a1[0] = [15.0, 10.0, 10.0]
    # Row B on atom 2: crop [3, 2, 1, 4] (permuted), extra atom 4; matches A0
    # over the shared atoms {1, 2, 3}.
    crop_b = [3, 2, 1, 4]
    pos_b = np.array([shared_pos[3], shared_pos[2], shared_pos[1], [15.0, 15.0, 15.0]])

    table = _table(
        config,
        [
            _event(1, pos_a0, crop_a0, num_reference_event=7),
            _event(1, pos_a1, crop_a0, energy_barrier=0.9, num_reference_event=7),
            _event(2, pos_b, crop_b, num_reference_event=7),
        ],
    )
    # The current rcut orderings differ from both stored crops.
    current = {1: [0, 1, 2, 3], 2: [1, 2, 3, 4]}
    table.remove_duplicates(_CELL, neighbors_list=_neighbors_list(current))
    # A0 and A1 survive part 1 (distinct barriers, distinct geometry); B is the
    # symmetric duplicate of A0 removed by part 2.
    assert len(table.table) == 2
    assert set(table.table["atom_index"]) == {1}


def test_symmetric_pass_skips_a_row_without_identities(config: Config) -> None:
    """Part 2 never compares a row with identities against one without.

    The two crops follow different orders -- one its stored identities, the
    other the current neighbour list -- so there is nothing to align on. Row 0
    has no identities; row 2 is geometrically the same event as row 0 under the
    current ordering, so a positional compare of the pair would remove the
    healthy row 2 as a duplicate. Both are kept, whichever row comes first.
    """
    positions = np.array(
        [
            [10.0, 10.0, 10.0],
            [12.0, 10.0, 10.0],
            [10.0, 12.0, 10.0],
            [10.0, 10.0, 12.0],
        ]
    )
    current = {0: [0, 1, 2, 3], 1: [1, 0, 2, 3]}
    far = positions.copy()
    far[1] = [15.0, 10.0, 10.0]
    table = _table(
        config,
        [
            _event(0, positions, None, num_reference_event=7),
            _event(0, far, [0, 1, 2, 3], energy_barrier=0.9, num_reference_event=7),
            _event(1, positions[[1, 0, 2, 3]], [1, 0, 2, 3], num_reference_event=7),
        ],
    )
    table.remove_duplicates(_CELL, neighbors_list=_neighbors_list(current))
    assert len(table.table) == 3
