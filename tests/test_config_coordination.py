"""Validation tests for the [AtomicEnvironment] style and threshold rules."""

import pytest
from pydantic import ValidationError
from pykmc.config import AtomicEnvironmentConfig


def test_pure_cna_style_rejected() -> None:
    """Pure 'cna' is rejected: crystal/noncrystal labels cannot match event IDs."""
    with pytest.raises(ValidationError, match="can never be matched and reused"):
        AtomicEnvironmentConfig(style="cna", rnei=3.0)


def test_pure_coordination_style_rejected() -> None:
    """Pure 'coordination' is rejected even when a threshold is provided."""
    with pytest.raises(ValidationError, match="can never be matched and reused"):
        AtomicEnvironmentConfig(
            style="coordination", rnei=3.0, coordination_threshold=12
        )


def test_coordination_graph_style_requires_threshold() -> None:
    """'coordination/graph' without coordination_threshold fails validation."""
    with pytest.raises(ValidationError, match="coordination_threshold is required"):
        AtomicEnvironmentConfig(style="coordination/graph", rnei=3.0, rcut=7.0)


def test_cna_graph_style_does_not_require_threshold() -> None:
    """'cna/graph' validates without a coordination_threshold."""
    cfg = AtomicEnvironmentConfig(style="cna/graph", rnei=3.0, rcut=7.0)
    assert cfg.coordination_threshold is None
