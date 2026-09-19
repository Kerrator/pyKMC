"""Reference-table persistence (architecture rule 6).

HTST tables carry versioned unit metadata inside the pickle (pandas
``DataFrame.attrs`` round-trip through ``to_pickle``/``read_pickle``);
legacy tables are loaded deliberately with a ``legacy`` status; a temperature
change on reload recomputes the rates; constant-mode pickles are byte for
byte the base's.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pykmc.config import Config, RateConstantConfig
from pykmc.event_table import (
    REFERENCE_BASE_COLUMNS,
    REFERENCE_HTST_COLUMNS,
    TABLE_SCHEMA_VERSION,
    ReferenceEventTable,
)
from pykmc.rate_constant import rate_from_prefactor
from tests.lifecycle.conftest import DATA_INPUT, accepted, rejected

LEGACY_PICKLE = Path("tests/data/reference_table_Cu_fake.pickle")


def _config(style: str, path: str | None = None, **rate: Any) -> Config:
    """Config with the given rate style and an optional reference table path."""
    config = Config.from_ini_file(DATA_INPUT)
    rate_cfg = RateConstantConfig(style=style, **rate)
    control = config.control.model_copy(update={"reference_table": path})
    return config.model_copy(update={"rateconstant": rate_cfg, "control": control})


def _populate(table: ReferenceEventTable, system: Any, ids: list[int]) -> None:
    """Insert one trivial row per id (sparse, non-contiguous) with pending status."""
    pos = system.positions
    for n, idx in enumerate(ids):
        fwd, _ = table._build_event_series(
            min1_positions=pos,
            saddle_positions=pos,
            min2_positions=pos,
            index_move=0,
            dE_forward=0.4 + 0.3 * n,
            dE_backward=0.4 + 0.3 * n,
            cell=system.cell,
            types=list(system.types),
        )
        fwd["idx_ref"] = idx
        fwd["idx_backward"] = idx
        table.table = pd.concat([table.table, fwd.to_frame().T], ignore_index=True)


class TestHtstRoundTrip:
    """Finite, rejected and pending rows with sparse ids survive save/load."""

    def test_round_trip_preserves_rows_and_metadata(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """Ids, statuses, Hz nu0, ps^-1 prefactors and metadata come back."""
        config = _config("htst", k0=2.0, T=400.0)
        table = ReferenceEventTable(config)
        _populate(table, system_single_type_fcc, [12, 3, 7])
        table._patch_row(12, accepted(5.0e12))
        table._patch_row(3, rejected("unstable"))
        # id 7 stays pending (never resolved)
        out = tmp_path / "reference_table.pickle"
        table.save(str(out))

        raw = pd.read_pickle(out)
        assert raw.attrs == table.table_metadata()
        assert raw.attrs["schema_version"] == TABLE_SCHEMA_VERSION
        assert raw.attrs["nu0_units"] == "Hz"
        assert raw.attrs["k_prefactor_units"] == "ps^-1"
        assert raw.attrs["T"] == 400.0 and raw.attrs["k0"] == 2.0
        assert raw.attrs["settings"]["nu0_min_hz"] == 1.0e12
        assert raw.attrs["settings"]["nu0_max_hz"] == 1.0e14

        loaded = ReferenceEventTable(_config("htst", str(out), k0=2.0, T=400.0))
        assert list(loaded.table.columns) == list(REFERENCE_BASE_COLUMNS) + list(
            REFERENCE_HTST_COLUMNS
        )
        assert list(loaded.table["idx_ref"]) == [12, 3, 7]
        assert list(loaded.table["nu0_status"]) == ["ok", "rejected", "pending"]
        assert loaded.metadata["T"] == 400.0
        ok, rej, pend = (loaded.table.iloc[i] for i in range(3))
        assert ok["nu0"] == 5.0e12 and ok["k_prefactor"] == 5.0
        assert ok["k"] == rate_from_prefactor(5.0, float(ok["energy_barrier"]), 400.0)
        assert math.isnan(rej["nu0"]) and rej["k_prefactor"] == 2.0
        assert rej["nu0_reason"] == "out_of_window: unstable"
        assert rej["k"] == rate_from_prefactor(2.0, float(rej["energy_barrier"]), 400.0)
        assert math.isnan(pend["nu0"]) and pend["k_prefactor"] == 2.0
        # same temperature: the recomputation is bit-identical to what was saved
        pd.testing.assert_series_equal(
            loaded.table["k"].astype(float), table.table["k"].astype(float)
        )

    def test_temperature_change_recomputes_rates(
        self, system_single_type_fcc: Any, tmp_path: Path, caplog: Any
    ) -> None:
        """Reloading at another T recomputes k from k_prefactor and the barrier."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0, 1])
        table._patch_row(0, accepted(5.0e12))
        table._patch_row(1, rejected("x"))
        out = tmp_path / "ref.pickle"
        table.save(str(out))

        with caplog.at_level(logging.INFO, logger="log"):
            loaded = ReferenceEventTable(_config("htst", str(out), k0=1.0, T=600.0))
        r0, r1 = loaded.table.iloc[0], loaded.table.iloc[1]
        assert r0["k"] == rate_from_prefactor(5.0, float(r0["energy_barrier"]), 600.0)
        assert r1["k"] == rate_from_prefactor(1.0, float(r1["energy_barrier"]), 600.0)
        assert r0["k"] != table.table.iloc[0]["k"]
        assert loaded.metadata["T"] == 300.0
        assert any("saved at T = 300.0" in r.getMessage() for r in caplog.records)

    def test_k0_change_rebases_fallback_rows(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """Rejected rows take the current k0; accepted rows keep their nu0."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0, 1])
        table._patch_row(0, accepted(5.0e12))
        table._patch_row(1, rejected("x"))
        out = tmp_path / "ref.pickle"
        table.save(str(out))

        loaded = ReferenceEventTable(_config("htst", str(out), k0=3.0, T=300.0))
        assert loaded.table.iloc[0]["k_prefactor"] == 5.0
        assert loaded.table.iloc[1]["k_prefactor"] == 3.0
        assert loaded.table.iloc[1]["k"] == rate_from_prefactor(
            3.0, float(loaded.table.iloc[1]["energy_barrier"]), 300.0
        )

    def _saved_ok_table(
        self, system: Any, tmp_path: Path, nu0_hz: float = 5.0e12
    ) -> Path:
        """Save an htst table with one accepted row and return the pickle path."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system, [0])
        table._patch_row(0, accepted(nu0_hz))
        out = tmp_path / "ref.pickle"
        table.save(str(out))
        return out

    def test_inconsistent_k_prefactor_is_refused(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """An accepted row whose k_prefactor is not hz_to_per_ps(nu0) cannot load."""
        out = self._saved_ok_table(system_single_type_fcc, tmp_path)
        df = pd.read_pickle(out)
        attrs = dict(df.attrs)
        df.loc[0, "k_prefactor"] = 99.0
        df.attrs = attrs
        df.to_pickle(out)
        with pytest.raises(ValueError, match="k_prefactor = 99.0 .* resolves to 5.0"):
            ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))

    def test_consistent_ok_row_loads_through_the_backend(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """The resolution of nu0 reproduces the stored prefactor bit for bit."""
        out = self._saved_ok_table(system_single_type_fcc, tmp_path, nu0_hz=7.3e12)
        loaded = ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))
        row = loaded.table.iloc[0]
        assert row["k_prefactor"] == 7.3e12 * 1.0e-12
        assert row["k"] == rate_from_prefactor(
            row["k_prefactor"], float(row["energy_barrier"]), 300.0
        )

    def test_ok_row_without_finite_nu0_is_refused(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """Status ok with a NaN frequency is a corrupted row, not a k0 fallback."""
        out = self._saved_ok_table(system_single_type_fcc, tmp_path)
        df = pd.read_pickle(out)
        attrs = dict(df.attrs)
        df.loc[0, "nu0"] = float("nan")
        df.attrs = attrs
        df.to_pickle(out)
        with pytest.raises(ValueError, match="finite positive frequency"):
            ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))

    def test_ok_row_with_none_nu0_is_refused_as_value_error(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """An object ``None`` in nu0 on an accepted row raises ValueError, not TypeError."""
        out = self._saved_ok_table(system_single_type_fcc, tmp_path)
        df = pd.read_pickle(out)
        attrs = dict(df.attrs)
        df["nu0"] = df["nu0"].astype(object)
        df.loc[0, "nu0"] = None
        df.attrs = attrs
        df.to_pickle(out)
        with pytest.raises(ValueError, match="no nu0 value"):
            ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))

    def test_unknown_status_is_refused(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """A status outside the vocabulary is refused, never treated as k0."""
        out = self._saved_ok_table(system_single_type_fcc, tmp_path)
        df = pd.read_pickle(out)
        attrs = dict(df.attrs)
        df.loc[0, "nu0_status"] = "accepted"
        df.attrs = attrs
        df.to_pickle(out)
        with pytest.raises(ValueError, match="nu0_status 'accepted'"):
            ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))

    def test_rpa_and_htst_metadata_style(self, tmp_path: Path) -> None:
        """The persisted style names the backend that produced the table."""
        table = ReferenceEventTable(_config("rpa", k0=1.0, T=300.0))
        assert table.table_metadata()["style"] == "rpa"

    @pytest.mark.parametrize(
        "patch,match",
        [
            ({"schema_version": 2}, "schema_version"),
            ({"nu0_units": "THz"}, "units are never inferred"),
            ({"k_prefactor_units": "Hz"}, "units are never inferred"),
        ],
    )
    def test_bad_metadata_is_refused(
        self,
        patch: dict,
        match: str,
        system_single_type_fcc: Any,
        tmp_path: Path,
    ) -> None:
        """Unknown versions or units are errors, not guesses."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0])
        table._patch_row(0, accepted(5.0e12))
        out = tmp_path / "ref.pickle"
        table.save(str(out))
        df = pd.read_pickle(out)
        df.attrs = {**df.attrs, **patch}
        df.to_pickle(out)
        with pytest.raises(ValueError, match=match):
            ReferenceEventTable(_config("htst", str(out), k0=1.0, T=300.0))


def _donor_era_pickle(
    system: Any, path: Path, nu0_hz: float = 5.0e12, T: float = 300.0
) -> pd.DataFrame:
    """Write a donor-era HTST pickle: the 15 base columns + k_prefactor + nu0.

    No ``nu0_status``/``nu0_reason`` columns and no attrs, with ``k`` rated
    through the Hz frequency (``hz_to_per_ps``), as the pre-S6 HTST branches
    persisted their tables.
    """
    table = ReferenceEventTable(_config("constant", k0=1.0, T=T))
    _populate(table, system, [0, 1])
    df = table.table.copy()
    prefactor = nu0_hz * 1.0e-12
    df["k_prefactor"] = prefactor
    df["nu0"] = nu0_hz
    df["k"] = [
        rate_from_prefactor(prefactor, float(dE), T) for dE in df["energy_barrier"]
    ]
    assert list(df.columns) == list(REFERENCE_BASE_COLUMNS) + ["k_prefactor", "nu0"]
    df.to_pickle(path)
    assert pd.read_pickle(path).attrs == {}
    return df


class TestLegacyLoad:
    """Tables without HTST provenance are loaded as legacy, once warned."""

    def test_donor_era_pickle_in_htst_mode_is_legacy(
        self, system_single_type_fcc: Any, tmp_path: Path, caplog: Any
    ) -> None:
        """k_prefactor + nu0 without status columns or metadata is not an estimate."""
        out = tmp_path / "donor.pickle"
        _donor_era_pickle(system_single_type_fcc, out)
        with caplog.at_level(logging.WARNING, logger="log"):
            loaded = ReferenceEventTable(_config("htst", str(out), k0=2.0, T=300.0))
        assert list(loaded.table.columns) == list(REFERENCE_BASE_COLUMNS) + list(
            REFERENCE_HTST_COLUMNS
        )
        assert set(loaded.table["nu0_status"]) == {"legacy"}
        assert set(loaded.table["k_prefactor"]) == {2.0}
        assert set(loaded.table["nu0"]) == {5.0e12}  # diagnostic only
        for _, row in loaded.table.iterrows():
            assert row["k"] == rate_from_prefactor(
                2.0, float(row["energy_barrier"]), 300.0
            )
            assert "incomplete HTST columns (k_prefactor, nu0)" in row["nu0_reason"]
        assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1

    def test_constant_pickle_in_htst_mode(self, caplog: Any) -> None:
        """No HTST columns: status legacy, k_prefactor k0, k recomputed, one warning."""
        config = _config("htst", str(LEGACY_PICKLE), k0=2.0, T=500.0)
        with caplog.at_level(logging.WARNING, logger="log"):
            table = ReferenceEventTable(config)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "legacy" in warnings[0].getMessage()

        assert all(c in table.table.columns for c in REFERENCE_HTST_COLUMNS)
        assert set(table.table["nu0_status"]) == {"legacy"}
        assert set(table.table["k_prefactor"]) == {2.0}
        assert all(math.isnan(v) for v in table.table["nu0"])
        for _, row in table.table.iterrows():
            assert row["k"] == rate_from_prefactor(
                2.0, float(row["energy_barrier"]), 500.0
            )
        assert table.metadata == {}
        assert table.prefactor_summary()["legacy"] == len(table.table)

    def test_htst_columns_without_metadata_are_legacy(
        self, system_single_type_fcc: Any, tmp_path: Path, caplog: Any
    ) -> None:
        """Hz nu0 without table metadata is not trusted as an estimate."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0])
        table._patch_row(0, accepted(5.0e12))
        out = tmp_path / "ref.pickle"
        table.table.to_pickle(out)  # bypass save(): no attrs
        assert pd.read_pickle(out).attrs == {}

        with caplog.at_level(logging.WARNING, logger="log"):
            loaded = ReferenceEventTable(_config("htst", str(out), k0=1.5, T=300.0))
        row = loaded.table.iloc[0]
        assert row["nu0_status"] == "legacy"
        assert row["k_prefactor"] == 1.5
        assert row["nu0"] == 5.0e12  # kept as a diagnostic, not used
        assert row["k"] == rate_from_prefactor(1.5, float(row["energy_barrier"]), 300.0)
        assert "no HTST metadata" in row["nu0_reason"]
        assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


class TestConstantModeUnchanged:
    """Constant-mode tables load and save exactly as the base."""

    def test_constant_pickle_loads_as_the_base_loader(self) -> None:
        """A constant run reads a constant pickle byte for byte like pd.read_pickle."""
        table = ReferenceEventTable(_config("constant", str(LEGACY_PICKLE), k0=1.0))
        base = pd.read_pickle(LEGACY_PICKLE)
        pd.testing.assert_frame_equal(table.table, base)
        assert table.table.attrs == {}
        assert not any(c in table.table.columns for c in REFERENCE_HTST_COLUMNS)

    def test_constant_save_writes_no_attrs(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """The saved bytes equal a plain to_pickle of the same frame."""
        table = ReferenceEventTable(_config("constant", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0, 1])
        out = tmp_path / "ref.pickle"
        table.save(str(out))
        ref = tmp_path / "plain.pickle"
        table.table.to_pickle(ref)
        assert out.read_bytes() == ref.read_bytes()
        assert pd.read_pickle(out).attrs == {}
        assert list(pd.read_pickle(out).columns) == list(REFERENCE_BASE_COLUMNS)

    def test_htst_pickle_in_constant_mode_is_stripped(
        self, system_single_type_fcc: Any, tmp_path: Path, caplog: Any
    ) -> None:
        """A constant run never reuses per-event prefactors from an HTST pickle."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0])
        table._patch_row(0, accepted(5.0e12))
        out = tmp_path / "ref.pickle"
        table.save(str(out))

        with caplog.at_level(logging.WARNING, logger="log"):
            loaded = ReferenceEventTable(_config("constant", str(out), k0=4.0, T=300.0))
        assert list(loaded.table.columns) == list(REFERENCE_BASE_COLUMNS)
        assert loaded.table.attrs == {}
        row = loaded.table.iloc[0]
        assert row["k"] == rate_from_prefactor(4.0, float(row["energy_barrier"]), 300.0)
        assert any("constant style" in r.getMessage() for r in caplog.records)

    def test_donor_era_htst_pickle_in_constant_mode_is_stripped(
        self, system_single_type_fcc: Any, tmp_path: Path, caplog: Any
    ) -> None:
        """A partial HTST column set (k_prefactor + nu0, no attrs) is stripped too.

        Regression: the strip used to fire only for the complete S6 column
        set, so a donor-era pickle was loaded unchanged with its HTST-derived
        rates in a constant run.
        """
        out = tmp_path / "donor.pickle"
        donor = _donor_era_pickle(system_single_type_fcc, out, nu0_hz=5.0e12, T=300.0)
        with caplog.at_level(logging.WARNING, logger="log"):
            loaded = ReferenceEventTable(_config("constant", str(out), k0=1.0, T=300.0))
        assert list(loaded.table.columns) == list(REFERENCE_BASE_COLUMNS)
        assert loaded.table.attrs == {}
        for (_, row), (_, before) in zip(
            loaded.table.iterrows(), donor.iterrows(), strict=True
        ):
            k0_rate = rate_from_prefactor(1.0, float(row["energy_barrier"]), 300.0)
            assert row["k"] == k0_rate
            assert before["k"] != k0_rate  # the HTST rate was not reused
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "constant style" in warnings[0].getMessage()
        assert "k_prefactor, nu0" in warnings[0].getMessage()


class TestSaveIsResilientToConcat:
    """concat drops attrs; save re-attaches the metadata every time."""

    def test_metadata_survives_row_appends(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """Rows appended after a save still produce a pickle with metadata."""
        table = ReferenceEventTable(_config("htst", k0=1.0, T=300.0))
        _populate(table, system_single_type_fcc, [0])
        first = tmp_path / "a.pickle"
        table.save(str(first))
        _populate(table, system_single_type_fcc, [5])
        assert table.table.attrs == {}  # concat dropped them
        second = tmp_path / "b.pickle"
        table.save(str(second))
        assert pd.read_pickle(second).attrs == table.table_metadata()
        assert list(pd.read_pickle(second)["idx_ref"]) == [0, 5]
        assert np.array_equal(
            pd.read_pickle(first)["idx_ref"].to_numpy(), np.array([0], dtype=object)
        )


@pytest.fixture
def log_records() -> Any:
    """Collect every record the catalogue writes to the ``log`` logger."""
    records: list[logging.LogRecord] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    handler = _Collect(level=logging.DEBUG)
    logger = logging.getLogger("log")
    previous = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous)


def _warnings(records: list[logging.LogRecord]) -> list[str]:
    return [r.getMessage() for r in records if r.levelno == logging.WARNING]


class TestReloadCompatibility:
    """Contracts section 7d, F2: changed settings invalidate, the window re-applies."""

    @staticmethod
    def _saved(
        config: Config, system: Any, tmp_path: Path, nu0: float = 5.0e12
    ) -> Path:
        """Save one accepted (id 12) and one rejected (id 3) row under ``config``."""
        table = ReferenceEventTable(config)
        _populate(table, system, [12, 3])
        table._patch_row(12, accepted(nu0))
        table._patch_row(3, rejected("unstable"))
        out = tmp_path / "reference_table.pickle"
        table.save(str(out))
        return out

    @staticmethod
    def _row(table: ReferenceEventTable, idx_ref: int) -> Any:
        return table.table[table.table["idx_ref"] == idx_ref].iloc[0]

    def test_changed_free_region_center_marks_accepted_rows_stale(
        self, system_single_type_fcc: Any, tmp_path: Path, log_records: Any
    ) -> None:
        """Stored saddle-centred, loaded min1: the accepted row is stale on k0."""
        saved = self._saved(
            _config("htst", k0=2.0, T=300.0, free_region_center="saddle"),
            system_single_type_fcc,
            tmp_path,
        )
        loaded = ReferenceEventTable(
            _config("htst", str(saved), k0=2.0, T=300.0, free_region_center="min1")
        )
        stale = self._row(loaded, 12)
        assert stale["nu0_status"] == "stale"
        assert math.isnan(stale["nu0"])
        assert stale["nu0_reason"] == (
            "stale: free_region_center changed on reload (stored saddle, current "
            "min1); estimate discarded"
        )
        assert stale["k_prefactor"] == 2.0
        assert stale["k"] == rate_from_prefactor(
            2.0, float(stale["energy_barrier"]), 300.0
        )
        rej = self._row(loaded, 3)
        assert rej["nu0_status"] == "rejected"
        assert rej["nu0_reason"] == "out_of_window: unstable"
        assert loaded.prefactor_summary() == {
            "ok": 0,
            "rejected": 1,
            "pending": 0,
            "legacy": 0,
            "stale": 1,
        }
        assert loaded.reference_estimate(12) == {
            "nu0_hz": None,
            "nu0_status": "stale",
            "nu0_reason": stale["nu0_reason"],
            "nu0_source": "reference",
        }
        warnings = _warnings(log_records)
        assert len(warnings) == 1
        assert (
            "free_region_center changed on reload (stored saddle, current min1)"
            in (warnings[0])
        )
        assert "1 accepted estimate(s) discarded" in warnings[0]
        # Saving writes the current settings (no retained nu0 was computed
        # under the old ones); a reload under the stored settings does not
        # resurrect the discarded estimate.
        resaved = tmp_path / "resaved.pickle"
        loaded.save(str(resaved))
        raw = pd.read_pickle(resaved)
        assert raw.attrs["settings"]["free_region_center"] == "min1"
        assert list(raw["nu0_status"]) == ["stale", "rejected"]
        assert not pd.to_numeric(raw["nu0"], errors="coerce").notna().any()
        again = ReferenceEventTable(
            _config("htst", str(resaved), k0=2.0, T=300.0, free_region_center="min1")
        )
        assert list(again.table["nu0_status"]) == ["stale", "rejected"]
        assert len(_warnings(log_records)) == 1  # same settings: no new warning

    def test_pre_7c_table_without_the_centring_key_counts_as_min1(
        self, system_single_type_fcc: Any, tmp_path: Path, log_records: Any
    ) -> None:
        """A stored table without ``free_region_center`` was min1-centred."""
        saved = self._saved(
            _config("htst", k0=1.0, T=300.0, free_region_center="min1"),
            system_single_type_fcc,
            tmp_path,
        )
        raw = pd.read_pickle(saved)
        del raw.attrs["settings"]["free_region_center"]
        pre_7c = tmp_path / "pre_7c.pickle"
        raw.to_pickle(pre_7c)
        assert "free_region_center" not in pd.read_pickle(pre_7c).attrs["settings"]

        # Loaded under today's default (saddle): stale.
        under_default = ReferenceEventTable(
            _config("htst", str(pre_7c), k0=1.0, T=300.0)
        )
        assert under_default.config.rateconstant.free_region_center == "saddle"
        stale = self._row(under_default, 12)
        assert stale["nu0_status"] == "stale"
        assert stale["nu0_reason"] == (
            "stale: free_region_center changed on reload (stored min1, current "
            "saddle); estimate discarded"
        )
        assert stale["k_prefactor"] == 1.0
        assert len(_warnings(log_records)) == 1

        # Loaded under min1 (its actual centring): compatible, estimate kept.
        under_min1 = ReferenceEventTable(
            _config("htst", str(pre_7c), k0=1.0, T=300.0, free_region_center="min1")
        )
        kept = self._row(under_min1, 12)
        assert kept["nu0_status"] == "ok" and kept["nu0"] == 5.0e12
        assert kept["k_prefactor"] == 5.0
        assert len(_warnings(log_records)) == 1

    def test_missing_setting_key_counts_as_changed(
        self, system_single_type_fcc: Any, tmp_path: Path, log_records: Any
    ) -> None:
        """Unknown provenance is never compatible: an absent key invalidates."""
        saved = self._saved(
            _config("htst", k0=1.0, T=300.0), system_single_type_fcc, tmp_path
        )
        raw = pd.read_pickle(saved)
        del raw.attrs["settings"]["fd_step"]
        raw.to_pickle(saved)
        loaded = ReferenceEventTable(_config("htst", str(saved), k0=1.0, T=300.0))
        stale = self._row(loaded, 12)
        assert stale["nu0_status"] == "stale"
        assert stale["nu0_reason"] == (
            "stale: fd_step changed on reload (stored absent, current 0.01); "
            "estimate discarded"
        )

    def test_every_changed_setting_is_named(
        self, system_single_type_fcc: Any, tmp_path: Path, log_records: Any
    ) -> None:
        """Several differences produce one reason naming all of them, in order."""
        saved = self._saved(
            _config("htst", k0=1.0, T=300.0, free_radius=6.0, fd_step=0.01),
            system_single_type_fcc,
            tmp_path,
        )
        loaded = ReferenceEventTable(
            _config(
                "htst",
                str(saved),
                k0=1.0,
                T=300.0,
                free_radius=8.0,
                fd_step=0.02,
                zone_radius=12.0,
                premin=True,
            )
        )
        stale = self._row(loaded, 12)
        assert stale["nu0_reason"] == (
            "stale: free_radius changed on reload (stored 6.0, current 8.0); "
            "fd_step changed on reload (stored 0.01, current 0.02); "
            "zone_radius changed on reload (stored None, current 12.0); "
            "premin changed on reload (stored False, current True); "
            "estimate discarded"
        )
        assert len(_warnings(log_records)) == 1

    def test_window_is_reapplied_with_its_own_reason_and_warning(
        self, system_single_type_fcc: Any, tmp_path: Path, log_records: Any
    ) -> None:
        """The window is not a compatibility setting: rejected, not stale."""
        saved = self._saved(
            _config("htst", k0=1.0, T=300.0),
            system_single_type_fcc,
            tmp_path,
            nu0=20.0e12,
        )
        loaded = ReferenceEventTable(
            _config("htst", str(saved), k0=1.0, T=300.0, nu0_max_THz=10.0)
        )
        rej = self._row(loaded, 12)
        assert rej["nu0_status"] == "rejected"
        assert math.isnan(rej["nu0"])
        assert rej["nu0_reason"] == (
            "out_of_window (reload): nu0 = 2.0000e+13 Hz outside "
            "[1.0000e+12, 1.0000e+13] Hz"
        )
        assert rej["k_prefactor"] == 1.0
        assert rej["k"] == rate_from_prefactor(1.0, float(rej["energy_barrier"]), 300.0)
        assert loaded.prefactor_summary()["stale"] == 0
        warnings = _warnings(log_records)
        assert len(warnings) == 1
        assert (
            "1 accepted estimate(s) lie outside the current nu0 window" in warnings[0]
        )
        # Inclusive endpoints: a stored value exactly on the bound stays accepted.
        bound_dir = tmp_path / "bound"
        bound_dir.mkdir()
        on_bound = self._saved(
            _config("htst", k0=1.0, T=300.0),
            system_single_type_fcc,
            bound_dir,
            nu0=10.0e12,
        )
        kept = ReferenceEventTable(
            _config("htst", str(on_bound), k0=1.0, T=300.0, nu0_max_THz=10.0)
        )
        assert self._row(kept, 12)["nu0_status"] == "ok"

    def test_stale_rows_survive_a_second_reload_and_seed_k0(
        self, system_single_type_fcc: Any, tmp_path: Path
    ) -> None:
        """A stale row loads as stale (never recomputed) and inherits as k0."""
        from pykmc.event_table import ActiveEventTable
        from pykmc.result import EventRefinementOutput

        saved = self._saved(
            _config("htst", k0=1.0, T=300.0, fd_step=0.01),
            system_single_type_fcc,
            tmp_path,
        )
        config = _config("htst", str(saved), k0=1.0, T=300.0, fd_step=0.02)
        loaded = ReferenceEventTable(config)
        assert self._row(loaded, 12)["nu0_status"] == "stale"
        active = ActiveEventTable(config)
        active.add_events(
            EventRefinementOutput(
                central_atom_index=0,
                saddle_positions=np.zeros((2, 3)),
                E_saddle=0.5,
                min2_positions=np.zeros((2, 3)),
                dE_forward=0.5,
                num_reference_event=12,
                refined="F",
                **loaded.reference_estimate(12),
            )
        )
        row = active.table.iloc[0]
        assert row["nu0_status"] == "stale" and row["nu0_source"] == "k0"
        assert row["k_prefactor"] == 1.0 and math.isnan(row["nu0"])
        assert row["nu0_reason"].startswith("stale: fd_step changed on reload")
