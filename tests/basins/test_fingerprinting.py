"""Invariance and discrimination tests for the structural fingerprint module.

These exercise pure numpy functions (no MPI, no engine), so they run in the fast
pure-python subset. They pin the two properties the dedup pre-filter relies on:
periodic re-imaging must not change a fingerprint, and the atoms-of-interest
fingerprint must still encode the defect's absolute position.
"""

import numpy as np

from pykmc.basins import fingerprinting as fp


class TestFingerprinting:
    def test_com_fingerprint_boundary_crossing_invariance(self):
        """COM fingerprint must be invariant when translation causes atoms to wrap."""
        positions = np.array(
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]],
            dtype=float,
        )
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = fp.com_fingerprint(positions, cell, pbc)
        shifted = positions + np.array(
            [9.7, 9.7, 9.7]
        )  # 0.5 + 9.7 = 10.2 -> wraps to 0.2
        fp2 = fp.com_fingerprint(shifted, cell, pbc)
        assert np.allclose(fp1, fp2, atol=1e-10)

    def test_aoi_fingerprint_boundary_crossing_invariance(self):
        """Atoms-of-interest fingerprint must be invariant when a shift causes wrapping."""
        positions = np.array(
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]],
            dtype=float,
        )
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        fp1 = fp.atoms_of_interest_fingerprint(
            positions, cell, pbc, rnei=1.5, coord_thr=10
        )
        shifted = positions + np.array([9.7, 0.0, 0.0])  # straddle the x boundary
        fp2 = fp.atoms_of_interest_fingerprint(
            shifted, cell, pbc, rnei=1.5, coord_thr=10
        )
        assert np.allclose(fp1, fp2, atol=1e-10)

    def test_scalar_pbc_matches_array_pbc(self):
        """A scalar pbc (True), as the basin passes via System(pbc=True), must work.

        Reproduces the IndexError in atoms_of_interest_fingerprint / com_fingerprint:
        ``np.asarray(True, bool)`` is 0-dimensional, so ``pbc_array[dim]`` raised
        "too many indices for array: array is 0-dimensional". The basin builds its
        reconstructed states with System(..., pbc=True), so the fingerprint must
        accept a scalar pbc and treat it as all-periodic (== np.array([True]*3)).
        """
        positions = np.array(
            [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]],
            dtype=float,
        )
        cell = np.diag([10.0, 10.0, 10.0])
        pbc_array = np.array([True, True, True])

        # com_fingerprint: scalar True == [True, True, True]
        assert np.allclose(
            fp.com_fingerprint(positions, cell, True),
            fp.com_fingerprint(positions, cell, pbc_array),
        )
        # atoms_of_interest_fingerprint: scalar True == [True, True, True]
        assert np.allclose(
            fp.atoms_of_interest_fingerprint(
                positions, cell, True, rnei=1.5, coord_thr=10
            ),
            fp.atoms_of_interest_fingerprint(
                positions, cell, pbc_array, rnei=1.5, coord_thr=10
            ),
        )

    def test_circular_mean_localized_cluster(self):
        """Circular mean preserves COM-to-atom distances across boundary wrapping."""
        box = np.array([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        cluster = np.array(
            [[0.5, 0.8, 1.0], [1.5, 1.2, 0.9], [1.0, 0.5, 1.1], [1.0, 1.5, 1.0]],
            dtype=float,
        )
        com1, r1 = fp.circular_mean_position(cluster, box, pbc)
        assert np.all(r1 > 0.9), "Localized cluster should have high resultant"

        shifted = (cluster + np.array([9.5, 0.0, 0.0])) % box
        com2, _ = fp.circular_mean_position(shifted, box, pbc)

        def _mic_dists(positions, com):
            diffs = positions - com
            for dim in range(3):
                diffs[:, dim] -= np.round(diffs[:, dim] / box[dim]) * box[dim]
            return np.sort(np.linalg.norm(diffs, axis=1))

        assert np.allclose(
            _mic_dists(cluster, com1), _mic_dists(shifted, com2), atol=1e-10
        )

    def test_circular_mean_fallback_triggers(self):
        """A uniform atom distribution should produce a low resultant (ill-conditioned)."""
        box = np.array([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        rng = np.random.default_rng(42)
        uniform_pos = rng.uniform(0, 10, size=(200, 3))
        _, resultant = fp.circular_mean_position(uniform_pos, box, pbc)
        assert np.any(resultant < 0.2), (
            f"Expected low resultant for uniform dist, got {resultant}"
        )

    def test_two_component_discriminates_position(self):
        """K defect atoms give a K+1 vector whose last element encodes absolute position."""
        cell = np.diag([30.0, 30.0, 30.0])
        pbc = np.array([True, True, True])
        rng = np.random.default_rng(123)
        bulk = rng.uniform(-0.2, 0.2, size=(10, 3)) + np.array([15.0, 15.0, 15.0])
        defect = np.array(
            [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]], dtype=float
        )
        pos1 = np.vstack([bulk, defect + np.array([3.0, 3.0, 3.0])])
        pos2 = np.vstack([bulk, defect + np.array([25.0, 25.0, 25.0])])
        fp1 = fp.atoms_of_interest_fingerprint(pos1, cell, pbc, rnei=1.5, coord_thr=4)
        fp2 = fp.atoms_of_interest_fingerprint(pos2, cell, pbc, rnei=1.5, coord_thr=4)
        assert len(fp1) == 4
        assert len(fp2) == 4
        assert np.allclose(fp1[:3], fp2[:3], atol=1e-10)  # same defect shape
        assert not np.isclose(fp1[-1], fp2[-1], atol=0.1)  # different absolute position

    def test_no_undercoordinated_atoms_returns_empty(self):
        """A fully coordinated cell yields an empty atoms-of-interest fingerprint."""
        cell = np.diag([10.0, 10.0, 10.0])
        pbc = np.array([True, True, True])
        rng = np.random.default_rng(7)
        dense = rng.uniform(
            0, 10, size=(200, 3)
        )  # every atom has many neighbours within rnei
        out = fp.atoms_of_interest_fingerprint(dense, cell, pbc, rnei=2.0, coord_thr=1)
        assert out.size == 0


class TestFingerprintModeDispatch:
    """compute_fingerprint must honour [BASIN] fingerprint_mode."""

    @staticmethod
    def _config(mode, *, style="graph", coord_thr=None, fp_thr=None):
        from types import SimpleNamespace

        return SimpleNamespace(
            basin=SimpleNamespace(
                fingerprint_mode=mode,
                fingerprint_coordination_thr=fp_thr,
                fingerprint_tolerance=None,
            ),
            atomicenvironment=SimpleNamespace(
                style=style, coordination_threshold=coord_thr, rnei=1.5
            ),
        )

    _positions = np.array(
        [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5], [0.5, 1.5, 0.5], [0.5, 0.5, 1.5]]
    )
    _cell = np.diag([10.0, 10.0, 10.0])
    _pbc = np.array([True, True, True])

    def test_off_returns_none(self):
        cfg = self._config("off", style="coordination/graph", coord_thr=8)
        assert (
            fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc) is None
        )

    def test_com_forced_even_with_coordination_style(self):
        cfg = self._config("com", style="coordination/graph", coord_thr=8)
        out = fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc)
        expected = fp.com_fingerprint(self._positions, self._cell, self._pbc)
        assert np.allclose(out, expected)

    def test_atoms_of_interest_forced_with_explicit_thr(self):
        cfg = self._config("atoms_of_interest", fp_thr=10)
        out = fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc)
        expected = fp.atoms_of_interest_fingerprint(
            self._positions, self._cell, self._pbc, rnei=1.5, coord_thr=10
        )
        assert np.allclose(out, expected)

    def test_atoms_of_interest_without_thr_raises(self):
        import pytest

        cfg = self._config("atoms_of_interest")
        with pytest.raises(ValueError, match="fingerprint_coordination_thr"):
            fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc)

    def test_auto_derives_from_coordination_style(self):
        cfg = self._config("auto", style="coordination/graph", coord_thr=9)
        out = fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc)
        expected = fp.atoms_of_interest_fingerprint(
            self._positions, self._cell, self._pbc, rnei=1.5, coord_thr=10
        )
        assert np.allclose(out, expected)

    def test_auto_falls_back_to_com(self):
        cfg = self._config("auto")  # graph style, no thresholds
        out = fp.compute_fingerprint(cfg, self._positions, self._cell, self._pbc)
        expected = fp.com_fingerprint(self._positions, self._cell, self._pbc)
        assert np.allclose(out, expected)
