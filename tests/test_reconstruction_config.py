"""Cluster E: ReconstructionConfig validation of ``containment_margin``.

``containment_margin`` is subtracted from ``atomicenvironment.rcut`` to form the
mover-containment limit. A zero/negative margin silently disables the guard and a
margin >= rcut drives the limit <= 0 so every event is rejected as not contained
(the run silently purges its whole catalogue). These tests pin the field-level
``gt=0`` and the cross-field ``containment_margin < atomicenvironment.rcut`` check
added on the top-level ``Config`` model.
"""

import configparser

import pytest
from pydantic import ValidationError

from pykmc.config import Config


def _config_dict() -> dict:
    """Parse the canonical test INI into the section dict ``from_ini_file`` builds."""
    parser = configparser.ConfigParser()
    parser.optionxform = str
    parser.read("./tests/data/input.in")
    return {
        section.lower(): dict(parser.items(section)) for section in parser.sections()
    }


def _config_with_margin(margin: float) -> dict:
    """Return the test-INI section dict with a ``[Reconstruction]`` containment margin.

    ``rcut`` in the canonical INI is 6.5, so a valid margin is (0, 6.5).
    """
    d = _config_dict()
    d["reconstruction"] = {"containment_margin": str(margin)}
    return d


def test_valid_containment_margin_accepted() -> None:
    """A margin in (0, rcut) parses cleanly."""
    cfg = Config.model_validate(_config_with_margin(1.0))
    assert cfg.reconstruction.containment_margin == 1.0
    assert cfg.reconstruction.containment_margin < cfg.atomicenvironment.rcut


@pytest.mark.parametrize("margin", [0.0, -0.5])
def test_nonpositive_containment_margin_rejected(margin: float) -> None:
    """Zero / negative margin is rejected at parse by the field-level gt=0."""
    with pytest.raises(ValidationError):
        Config.model_validate(_config_with_margin(margin))


def test_margin_at_or_above_rcut_rejected_names_both_fields() -> None:
    """A margin >= rcut is rejected and the message names both fields + values."""
    with pytest.raises(ValidationError) as excinfo:
        Config.model_validate(_config_with_margin(6.5))  # == rcut
    msg = str(excinfo.value)
    assert "containment_margin" in msg
    assert "atomicenvironment.rcut" in msg
    assert "6.5" in msg  # both the margin and rcut value are 6.5 here

    with pytest.raises(ValidationError) as excinfo2:
        Config.model_validate(_config_with_margin(10.0))  # > rcut
    msg2 = str(excinfo2.value)
    assert "containment_margin" in msg2
    assert "atomicenvironment.rcut" in msg2
    assert "10.0" in msg2 and "6.5" in msg2
