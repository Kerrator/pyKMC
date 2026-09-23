"""Import isolation: constant rates never load LAMMPS or the HTST kernel."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PROBE = textwrap.dedent(
    """
    import sys
    from pathlib import Path

    import pykmc.rate_constant as rate_constant
    from pykmc.config import RateConstantConfig

    root = Path(sys.argv[1]).resolve()
    assert Path(rate_constant.__file__).resolve().is_relative_to(root), (
        rate_constant.__file__
    )
    rc = rate_constant.create_rate_constant(
        RateConstantConfig(style="constant", k0=2.0, T=300.0)
    )
    out = rc.compute_rate(0.1)
    assert out.prefactor == 2.0
    assert 0.0 < out.rate < 2.0
    assert rate_constant.PrefactorBackend._import_errors == {}
    loaded = sorted(
        name
        for name in sys.modules
        if name == "lammps"
        or name.startswith("lammps.")
        or name == "pykmc.htst"
        or name.startswith("pykmc.htst.")
    )
    print("LOADED=" + ",".join(loaded))
    """
)


def test_constant_rate_does_not_import_lammps_or_htst() -> None:
    """Run a fresh interpreter and assert neither lammps nor pykmc.htst was imported."""
    proc = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    lines = [line for line in proc.stdout.splitlines() if line.startswith("LOADED=")]
    assert lines == ["LOADED="], proc.stdout
