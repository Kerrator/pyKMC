"""Scalar/identity input contract; pure, no native or manager dispatch."""

from types import SimpleNamespace

import numpy as np
import pytest

from pykmc.config import RateConstantConfig, RegionConfig
from pykmc.htst import HTSTRequestError, HTSTSettings
from pykmc.physics import ResolvedConstraints
from pykmc.rate_constant import create_rate_constant
from pykmc.rate_constant.prefactors import PrefactorService, settings_from_config


def prepared_service():
    initial = np.array([[2.0, 3.0, 4.0], [3.0, 5.0, 4.0], [3.0, 3.0, 6.0]])
    saddle, final = initial.copy(), initial.copy()
    saddle[0, 0], final[0, 0] = 3.0, 4.0
    policy = RegionConfig(indices=[1])
    cell = np.diag([10.0, 12.0, 14.0])
    pbc = (True, False, True)
    authority = ResolvedConstraints.resolve(
        initial, ("Ni",) * 3, policy, (0, 1, 2), cell=cell, pbc=pbc
    )
    config = SimpleNamespace(
        frozen_atoms=policy,
        lammps=SimpleNamespace(
            pair_style="zero 10.0",
            pair_coeff="* *",
            min_style="cg",
            minimize="0.0 1e-12 100 1000",
            frz_min="0.0 1e-12 100 1000",
        ),
        rateconstant=RateConstantConfig(style="htst"),
    )
    service = PrefactorService(
        config,
        object(),
        create_rate_constant(config.rateconstant),
        species_masses=(("Ni",), (58.6934,)),
        global_constraints=authority,
    )
    rows = np.array([1, 0, 2])
    payload = dict(
        event_key=("identity-scalar-boundary",),
        min1_positions=initial[rows].copy(),
        saddle_positions=saddle[rows].copy(),
        min2_positions=final[rows].copy(),
        types=("Ni",) * 3,
        cell=cell.copy(),
        pbc=pbc,
        center_index=1,
    )
    return service, payload, authority


@pytest.mark.parametrize(
    "alias", [True, 1.0, np.bool_(True)], ids=["bool", "float", "numpy-bool"]
)
def test_malformed_identity_cannot_alias_integer_source_id(alias):
    service, payload, authority = prepared_service()
    before = {
        key: value.copy()
        for key, value in payload.items()
        if isinstance(value, np.ndarray)
    }
    # All coordinates and the desired permutation are coherent. Only the raw
    # identity's scalar type is invalid; equality with integer 1 cannot fix it.
    assert alias == 1
    with pytest.raises(
        (HTSTRequestError, ValueError), match="(?i)(identit|integer|atom_ids|source)"
    ):
        service.build_request(**payload, atom_ids=(alias, 0, 2))
    assert service.global_constraints is authority
    assert authority.atom_ids == (0, 1, 2) and authority.fixed_ids == (1,)
    for key, expected in before.items():
        np.testing.assert_array_equal(payload[key], expected)


@pytest.mark.parametrize(
    "ids",
    [(1, 0, 2), (np.int64(1), np.int64(0), np.int64(2))],
    ids=["python-integers", "numpy-integers"],
)
def test_genuine_integer_permutation_retains_authoritative_fixed_identity(ids):
    service, payload, authority = prepared_service()
    request = service.build_request(**payload, atom_ids=ids)
    assert request.constraints.source_ids == (0, 1, 2)
    assert request.constraints.atom_ids == (1, 0, 2)
    assert request.constraints.fixed_ids == (1,)
    assert request.constraints.local_fixed_indices == (0,)
    assert request.user_constraints == authority
    np.testing.assert_array_equal(request.min1_positions, payload["min1_positions"])


@pytest.mark.parametrize(
    "value", [np.bool_(True), np.bool_(False)], ids=["numpy-true", "numpy-false"]
)
def test_numpy_booleans_cannot_become_numeric_force_tolerances(value):
    with pytest.raises((ValueError, TypeError)):
        RateConstantConfig(style="htst", force_tol=value)
    with pytest.raises((ValueError, TypeError)):
        HTSTSettings(force_tol=value)


@pytest.mark.parametrize(
    "value",
    [np.float64(0.005), np.float32(0.007), np.int64(1)],
    ids=["float64", "float32", "integer64"],
)
def test_numpy_real_force_tolerances_are_preserved(value):
    expected = float(value)
    config = RateConstantConfig(style="htst", force_tol=value)
    assert config.force_tol == expected
    assert HTSTSettings(force_tol=value).force_tol == expected
    assert settings_from_config(config).force_tol == expected


def test_ini_numeric_spelling_still_reaches_frozen_settings():
    config = RateConstantConfig(style="htst", force_tol="0.007")
    assert config.force_tol == 0.007
    assert settings_from_config(config).force_tol == 0.007
