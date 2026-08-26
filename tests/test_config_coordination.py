import pytest
from pydantic import ValidationError
from pykmc.config import AtomicEnvironmentConfig


def test_coordination_style_requires_threshold():
    with pytest.raises(ValidationError, match="coordination_threshold is required"):
        AtomicEnvironmentConfig(style="coordination", rnei=3.0)


def test_coordination_graph_style_requires_threshold():
    with pytest.raises(ValidationError, match="coordination_threshold is required"):
        AtomicEnvironmentConfig(style="coordination/graph", rnei=3.0, rcut=7.0)


def test_cna_style_does_not_require_threshold():
    cfg = AtomicEnvironmentConfig(style="cna", rnei=3.0)
    assert cfg.coordination_threshold is None
