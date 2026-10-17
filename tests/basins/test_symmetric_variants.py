"""Symmetry variants of a reference event must move only the event atoms.

Reference events are stored in box coordinates; their SOFI symmetry operations act
about the event centre. The basin used to rotate the absolute positions (about the
box origin), so every ``sym != 0`` connectivity row reconstructed a shell displaced
by tens of Angstrom (fcc(100) pit example, 2026-08-27).
``geometry.symmetric_event_positions`` is the shared displacement-form transform now
used by basin.py (system_from_state, refine_absorbing, _map_reference_saddle) and
the engine-side _basin_reconstruct_impl.

Permutation convention (SOFI/IRA, as returned by
``pykmc.symmetries.unique_symmetries``): ``sym_perm[k]`` is the atom whose rotated
image lands on site ``k``, i.e. ``(x @ R.T)[sym_perm] == x`` about the shell
centre. The (R, perm) pairs below are the ones ``ira_mod.SOFI`` returns for this
shell in this atom order, so these tests need no IRA.
"""

import logging
from types import SimpleNamespace

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest

import pykmc.basins.basin as basin_module
from pykmc import System
from pykmc.result import Ok, PSROutput
from pykmc.utils import geometry

A = 2.49  # fcc(100) nearest-neighbour distance, Angstrom
C4 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # +90 deg, z
SQUARE_OPERATIONS = {
    "E": (np.eye(3), np.array([0, 1, 2, 3, 4])),
    "C4": (C4, np.array([3, 0, 1, 2, 4])),
    "C2": (C4 @ C4, np.array([2, 3, 0, 1, 4])),
    "C4^3": (C4 @ C4 @ C4, np.array([1, 2, 3, 0, 4])),
    "S4": (C4 @ np.diag([1.0, 1.0, -1.0]), np.array([3, 0, 1, 2, 4])),
}


def _square_sites(offset: float) -> np.ndarray:
    """Return a square of 4 neighbours (+x, +y, -x, -y) around a central mover.

    Parameters
    ----------
    offset : float
        Shift of the whole shell from the box origin, Angstrom.

    Returns
    -------
    np.ndarray
        Positions, shape (5, 3); the mover is index 4.

    """
    sites = [[A, 0, 0], [0, A, 0], [-A, 0, 0], [0, -A, 0], [0, 0, 0]]
    return np.array(sites, dtype=float) + offset


def _square_shell_event(
    offset: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build a mover at the centre of a square of 4 neighbours, far from the origin.

    The event moves the mover by (+2.49, 0, 0). The +90 deg rotation about z carries
    neighbour i onto the site of neighbour i + 1 (mod 4), so in the SOFI convention
    ``perm[k] = k - 1 (mod 4)``; the mover (index 4) is fixed.

    Parameters
    ----------
    offset : float, optional
        Shift of the whole shell from the box origin, by default 30.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(initial, saddle, final, c4, perm)``.

    """
    initial = _square_sites(offset)
    disp_final = np.zeros_like(initial)
    disp_final[4] = [A, 0, 0]
    disp_saddle = 0.5 * disp_final
    final = initial + disp_final
    saddle = initial + disp_saddle
    c4, perm = SQUARE_OPERATIONS["C4"]
    # site k receives the rotated image of atom perm[k] (SOFI/IRA convention):
    # rotated neighbour 3 lands on site 0, 0 on 1, 1 on 2, 2 on 3; the mover is fixed
    return initial, saddle, final, c4, perm


def _pushed_neighbour_event(
    offset: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an event whose variants tell the two permutation conventions apart.

    Same square shell, but the mover (index 4) hops +1.2 A along x, so it does not
    land on a neighbour site, and only the neighbour ahead of it (index 0, on +x) is
    pushed: +0.04 A along x at the saddle, +0.10 A at the final state. The other
    neighbours stay fixed.

    Parameters
    ----------
    offset : float, optional
        Shift of the whole shell from the box origin, by default 30.0.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(initial, saddle, final)``, each of shape (5, 3).

    """
    initial = _square_sites(offset)
    saddle = initial.copy()
    final = initial.copy()
    saddle[4] += [0.6, 0.0, 0.0]
    final[4] += [1.2, 0.0, 0.0]
    saddle[0] += [0.04, 0.0, 0.0]
    final[0] += [0.10, 0.0, 0.0]
    return initial, saddle, final


def _rotated_event(
    initial: np.ndarray, saddle: np.ndarray, final: np.ndarray, rot: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a whole event about its shell centre and relabel it by site.

    Uses no permutation array: site ``k`` takes the rotated saddle and final
    positions of the atom whose rotated initial position lands on site ``k``.

    Parameters
    ----------
    initial, saddle, final : np.ndarray
        Event positions, shape (N, 3); ``initial`` is symmetric under ``rot``.
    rot : np.ndarray
        3x3 point operation about the centre of ``initial``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(saddle, final)`` of the rotated event, in site order.

    """
    centre = initial.mean(axis=0)
    rot_initial = (initial - centre) @ rot.T + centre
    rot_saddle = (saddle - centre) @ rot.T + centre
    rot_final = (final - centre) @ rot.T + centre
    gap = np.linalg.norm(initial[:, None, :] - rot_initial[None, :, :], axis=2)
    atom_on_site = gap.argmin(axis=1)
    npt.assert_allclose(gap[np.arange(len(initial)), atom_on_site], 0.0, atol=1e-9)
    assert len(set(atom_on_site.tolist())) == len(initial)
    return rot_saddle[atom_on_site], rot_final[atom_on_site]


class TestSymmetricEventPositions:
    """geometry.symmetric_event_positions."""

    def test_only_the_mover_moves_and_along_the_rotated_direction(
        self, test_logger: logging.Logger
    ) -> None:
        """The variant's final state = initial + rotated, relabelled displacement."""
        initial, saddle, final, c4, perm = _square_shell_event()
        ini_v, sad_v, fin_v = geometry.symmetric_event_positions(
            initial, saddle, final, c4, perm
        )
        npt.assert_allclose(ini_v, initial)  # PSR anchor: unchanged
        d_fin = fin_v - initial
        d_sad = sad_v - initial
        # neighbours do not move
        npt.assert_allclose(d_fin[:4], 0.0, atol=1e-12)
        npt.assert_allclose(d_sad[:4], 0.0, atol=1e-12)
        # mover displacement rotated by +90 deg: (+2.49, 0, 0) -> (0, +2.49, 0)
        npt.assert_allclose(d_fin[4], [0.0, 2.49, 0.0], atol=1e-12)
        npt.assert_allclose(d_sad[4], [0.0, 1.245, 0.0], atol=1e-12)

    def test_identity_variant_is_the_reference_event(
        self, test_logger: logging.Logger
    ) -> None:
        """Variant 0 (identity, identity permutation) reproduces the stored event."""
        initial, saddle, final, _, _ = _square_shell_event()
        ini_v, sad_v, fin_v = geometry.symmetric_event_positions(
            initial, saddle, final, np.eye(3), np.arange(5)
        )
        npt.assert_allclose(ini_v, initial)
        npt.assert_allclose(sad_v, saddle)
        npt.assert_allclose(fin_v, final)

    def test_rotating_absolute_positions_is_the_old_bug(
        self, test_logger: logging.Logger
    ) -> None:
        """Guard: the position-rotation form displaces every atom by ~the offset."""
        initial, saddle, final, c4, perm = _square_shell_event(offset=30.0)
        old_final = geometry.transform_positions(final, c4, 0, perm)
        moved = np.linalg.norm(old_final - initial, axis=1)
        assert np.all(moved > 10.0)  # whole shell thrown across the box
        _, _, fin_v = geometry.symmetric_event_positions(
            initial, saddle, final, c4, perm
        )
        moved_new = np.linalg.norm(fin_v - initial, axis=1)
        assert np.count_nonzero(moved_new > 1e-9) == 1

    def test_pushed_neighbour_follows_the_rotation(
        self, test_logger: logging.Logger
    ) -> None:
        """Under +90 deg about z the push moves to the +y neighbour, along +y.

        The +x neighbour (index 0) is pushed in the reference event; its rotated
        image sits on the +y site (index 1). With the inverse permutation the push
        would land on the -y site (index 3), and with the identity permutation it
        would stay on the +x site, so this pins the SOFI convention.
        """
        initial, saddle, final = _pushed_neighbour_event()
        rot, perm = SQUARE_OPERATIONS["C4"]
        _, sad_v, fin_v = geometry.symmetric_event_positions(
            initial, saddle, final, rot, perm
        )
        expected_fin = np.zeros((5, 3))
        expected_fin[4] = [0.0, 1.2, 0.0]  # mover: +x hop -> +y hop
        expected_fin[1] = [0.0, 0.10, 0.0]  # push: +x neighbour -> +y neighbour
        expected_sad = np.zeros((5, 3))
        expected_sad[4] = [0.0, 0.6, 0.0]
        expected_sad[1] = [0.0, 0.04, 0.0]
        npt.assert_allclose(fin_v - initial, expected_fin, atol=1e-12)
        npt.assert_allclose(sad_v - initial, expected_sad, atol=1e-12)

    @pytest.mark.parametrize("name", list(SQUARE_OPERATIONS))
    def test_variant_is_the_event_rotated_about_its_centre(
        self, test_logger: logging.Logger, name: str
    ) -> None:
        """Each variant equals the whole event rotated about the shell centre.

        The expected positions come from ``_rotated_event``, which never reads
        ``perm``; C4, C4^3 and S4 have permutations that are not their own inverse,
        so an inverted convention fails here as well.
        """
        initial, saddle, final = _pushed_neighbour_event()
        rot, perm = SQUARE_OPERATIONS[name]
        ini_v, sad_v, fin_v = geometry.symmetric_event_positions(
            initial, saddle, final, rot, perm
        )
        exp_sad, exp_fin = _rotated_event(initial, saddle, final, rot)
        npt.assert_array_equal(ini_v, initial)
        npt.assert_allclose(sad_v, exp_sad, atol=1e-12)
        npt.assert_allclose(fin_v, exp_fin, atol=1e-12)


class TestMapReferenceSaddle:
    """A basin call site applies the variant in reference order, before the PSR."""

    def test_variant_saddle_lands_on_the_matched_shell(
        self, test_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``_map_reference_saddle`` moves only the event atoms of the state's shell.

        The state's shell is the reference shell translated by ``shift`` and listed
        in another order (state atom k is reference atom ``order[k]``); the stubbed
        PSR returns exactly that transform. The C4 variant must push the state atoms
        that stand for the reference mover (state 2) and the +y neighbour (state 3).
        Rotating absolute positions, an inverted permutation, or applying the
        variant after the PSR all break this.
        """
        initial, saddle, final = _pushed_neighbour_event()
        rot, perm = SQUARE_OPERATIONS["C4"]
        shift = np.array([-12.0, 5.0, 3.0])
        order = np.array([2, 0, 4, 1, 3])
        state_positions = (initial + shift)[order]
        table = pd.DataFrame(
            [
                {
                    "idx_ref": 7,
                    "initial_positions": initial,
                    "saddle_positions": saddle,
                    "final_positions": final,
                    "sym_matrix": np.array([np.eye(3), rot]),
                    "sym_perm": np.array([np.arange(5), perm]),
                }
            ]
        )
        psr = PSROutput(
            rotation_matrix=np.eye(3),
            translation_matrix=shift,
            permutation_matrix=order,
            matching_score=0.0,
        )
        monkeypatch.setattr(
            basin_module,
            "PointSetRegistration",
            lambda *_args: SimpleNamespace(match=lambda: Ok(psr)),
        )
        basin = basin_module.BasinsGenericEvents.__new__(
            basin_module.BasinsGenericEvents
        )
        basin.reference_table = SimpleNamespace(table=table)
        basin.config = SimpleNamespace(psr=SimpleNamespace(matching_score_thr=0.1))
        state_system = System(
            types=np.array(["Ni"] * 5),
            positions=state_positions.copy(),
            cell=np.diag([60.0, 60.0, 60.0]),
            pbc=np.array([True, True, True]),
            index=np.arange(5),
        )
        neighbors = SimpleNamespace(get_neighbors=lambda *_args: np.arange(5))
        basin.states = {
            0: SimpleNamespace(system=state_system, neighbors_list=neighbors)
        }
        mapped = basin._map_reference_saddle(0, 2, 7, 1)
        expected = state_positions.copy()
        expected[2] += [0.0, 0.6, 0.0]  # reference mover: +x hop -> +y hop
        expected[3] += [0.0, 0.04, 0.0]  # reference +y neighbour takes the push
        npt.assert_allclose(mapped, expected, atol=1e-12)
        npt.assert_array_equal(state_system.positions, state_positions)  # untouched
