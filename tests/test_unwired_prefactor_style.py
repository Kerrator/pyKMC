"""The htst/rpa rate styles are refused until the KMC loop computes prefactors.

``RateConstantConfig`` and ``pykmc.rate_constant`` accept ``htst``/``rpa``, but
this loop requests no per-event prefactor, so every event would silently take
the ``k0`` fallback. Both entry points refuse the styles: ``run.main`` before
any worker is launched (so no rank is left waiting), and ``KMC`` itself.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pykmc.run as run_module
import pytest
from pykmc.config import Config
from pykmc.kmc import KMC

DATA_INPUT = "./tests/data/input.in"


def _ini(tmp_path: Path, style: str) -> Path:
    text = Path(DATA_INPUT).read_text()
    text = text.replace("style = constant", f"style = {style}").replace(
        "k0 = 1e12", "k0 = 1.0"
    )
    ini = tmp_path / "input.in"
    ini.write_text(text)
    return ini


def _drive(monkeypatch: pytest.MonkeyPatch, ini: Path) -> list[dict[str, Any]]:
    """Run ``run.main`` on ``ini`` with a factory that launches no worker."""
    created: list[dict[str, Any]] = []

    class FakeFactory:
        def __init__(self, **kwargs: Any) -> None:
            created.append(kwargs)

        def launch(self) -> None:
            return None  # a worker rank: no KMC is built

    monkeypatch.setattr(run_module, "EngineManagerFactory", FakeFactory)
    monkeypatch.setattr(sys, "argv", ["pykmc", "-in", str(ini)])
    run_module.main()
    return created


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_run_refuses_the_style_before_launching_workers(
    style: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No factory is built, so no rank is left waiting for work."""
    created: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="not wired into the KMC loop"):
        created = _drive(monkeypatch, _ini(tmp_path, style))
    assert created == []


@pytest.mark.parametrize("style", ["htst", "rpa"])
def test_kmc_refuses_the_style(style: str, tmp_path: Path) -> None:
    """A KMC built directly refuses the style as well."""
    config = Config.from_ini_file(str(_ini(tmp_path, style)))
    with pytest.raises(ValueError, match="k0 fallback"):
        KMC(config)


def test_constant_style_is_unaffected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The constant style still reaches the factory and builds a KMC."""
    assert len(_drive(monkeypatch, _ini(tmp_path, "constant"))) == 1
    KMC(Config.from_ini_file(DATA_INPUT))
