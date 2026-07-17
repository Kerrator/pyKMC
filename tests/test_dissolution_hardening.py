"""Hardening tests for dealloying dissolution (post-review defect fixes).

These cover the config-level guards and the LAMMPS type-universe helper added
after the adversarial review: bias/dissolution mutual exclusion, element-symbol
validation, and the fixed type universe (pair_coeff elements are always declared
even when the live atom set is missing a species -- the terminal-state bug).
"""

import types as _types

import pytest
from pydantic import ValidationError

from pykmc.activevolume.active_volume import _type_universe
from pykmc.config import Config, DissolutionConfig
from pykmc.system import elements_from_pair_coeff


def _config_dict(**control_overrides: object) -> dict:
    """Minimal-but-complete nested config dict for Config.model_validate."""
    control = {
        "initial_config": "x.xyz",
        "n_steps": 3,
        "engine": "lammps",
    }
    control.update(control_overrides)
    return {
        "control": control,
        "lammps": {"pair_style": "eam/alloy", "pair_coeff": "* * pot Cr Ni"},
        "atomicenvironment": {"style": "cna/graph", "rnei": 2.8, "rcut": 6.5},
        "eventsearch": {"style": "partn", "nsearch": 4},
        "partn": {},
        "rateconstant": {"style": "constant", "k0": 1.0},
        "psr": {"style": "ira"},
        "ira": {},
        "bias": {"style": "direction"},
        "dissolution": {"elements": "Cr", "nu_d": 10.0, "E_b": 0.1},
    }


# ------------------------------------------------------- bias x dissolution ----


def test_bias_and_dissolution_are_mutually_exclusive() -> None:
    """Enabling both bias and dissolution is rejected at config time."""
    with pytest.raises(ValidationError) as ei:
        Config.model_validate(_config_dict(bias=True, dissolution=True))
    assert "cannot both be True" in str(ei.value)


def test_bias_only_is_allowed() -> None:
    """Bias without dissolution still validates."""
    cfg = Config.model_validate(_config_dict(bias=True, dissolution=False))
    assert cfg.control.bias
    assert not cfg.control.dissolution


def test_dissolution_only_is_allowed() -> None:
    """Dissolution without bias still validates."""
    cfg = Config.model_validate(_config_dict(bias=False, dissolution=True))
    assert cfg.control.dissolution
    assert not cfg.control.bias


# --------------------------------------------------- element symbol checks ----


def test_elements_rejects_unknown_symbol() -> None:
    """A non-element token in elements is rejected loudly at config time."""
    with pytest.raises(ValidationError):
        DissolutionConfig(elements="Xx", nu_d=10.0, E_b=0.1)


def test_elements_rejects_unknown_symbol_in_list() -> None:
    """A bad symbol anywhere in a comma list is rejected."""
    with pytest.raises(ValidationError):
        DissolutionConfig(elements="Cr,Zz", nu_d=10.0, E_b=0.1)


# ---------------------------------------------------- type-universe helper ----


def test_elements_from_pair_coeff() -> None:
    """pair_coeff element symbols are parsed in type order; junk -> None."""
    assert elements_from_pair_coeff("* * FeNiCr.eam Fe Ni Cr") == ["Fe", "Ni", "Cr"]
    assert elements_from_pair_coeff("* * pot Ni") == ["Ni"]
    assert elements_from_pair_coeff("") is None
    assert elements_from_pair_coeff(None) is None


def test_type_universe_includes_missing_species() -> None:
    """The AV type universe declares every pair_coeff element (alphabetical).

    This is the terminal-state fix: a crop (or a whole system after dealloying
    deleted the last atom of a species) that lacks a species still yields the
    full, correctly-ordered type universe -- so create_box/pair_coeff agree.
    """
    cfg = _types.SimpleNamespace(
        lammps=_types.SimpleNamespace(pair_coeff="* * pot Cr Ni")
    )
    # crop with only Ni -> full Cr,Ni universe (alphabetical == pair_coeff order)
    assert _type_universe(["Ni", "Ni"], cfg) == ["Cr", "Ni"]
    # behaviour-preserving when both species present
    assert _type_universe(["Cr", "Ni", "Ni"], cfg) == ["Cr", "Ni"]
