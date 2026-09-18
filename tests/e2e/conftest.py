"""Opt-in live KMC harness: bounded ``python -m pykmc`` runs under ``mpirun``.

Every test in this directory is skipped unless the environment variable
``PYKMC_RUN_E2E`` is set to ``1``, so the default serial lane and CI never
launch MPI. The launcher is ``PYKMC_MPIRUN`` (a command line, split with
``shlex``) or the first ``mpirun`` on ``PATH``; ``PYKMC_E2E_TIMEOUT`` bounds
each child in seconds (default 900).

Each case runs in a fresh temporary directory (pyKMC opens its log files in
append mode) with ``sys.executable`` as the interpreter, ``PYTHONPATH`` set to
the repository root discovered from this file, the imported ``pykmc.__file__``
asserted inside the child, a fixed non-zero pARTn ``zseed`` and a seeded
Python-level ``random`` (pyKMC never seeds it: the central-atom choice and the
rejection-free draw use the global module, so a fixed ``zseed`` alone does not
make a run reproducible).
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pytest

E2E_ENV = "PYKMC_RUN_E2E"
REPO_ROOT = Path(__file__).resolve().parents[2]
SI_VAC = REPO_ROOT / "examples" / "Si_vac"
ZSEED = 12345
PY_SEED = 20260918

_CHILD = (
    "import os, pathlib, random, sys\n"
    "import pykmc\n"
    "p = pathlib.Path(pykmc.__file__).resolve()\n"
    "root = pathlib.Path(os.environ['PYKMC_E2E_ROOT']).resolve()\n"
    "assert p.is_relative_to(root), (str(p), str(root))\n"
    "print('[e2e] pykmc.__file__ =', p, flush=True)\n"
    "random.seed(int(os.environ['PYKMC_E2E_SEED']))\n"
    "sys.argv = ['pykmc', '-in', 'input.in']\n"
    "import pykmc.run\n"
    "pykmc.run.main()\n"
)

SI_VAC_INPUT = """[Control]
initial_config = initial_config.xyz
n_steps = {n_steps}
engine = lammps
verbosity = 2
n_sessions = {n_sessions}

[Lammps]
pair_style = sw
pair_coeff = * * Si.sw Si
min_style = cg
minimize = 1e-10 1e-12 10000 10000

[AtomicEnvironment]
style = diamond/graph
rnei = 3.0
rcut = 6.3

[EventSearch]
style = partn
nsearch = {nsearch}
emax_event = 3.5
emin_event = 0.0

[pARTn]
zseed = {zseed}

[RateConstant]
style = {style}
k0 = 10
T = 300.0

[PSR]
style = ira

[IRA]
"""


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip every test under this directory unless ``PYKMC_RUN_E2E=1``.

    Parameters
    ----------
    config : pytest.Config
        The session configuration (unused; required by the hook signature).
    items : list of pytest.Item
        The collected items; those under this directory receive a skip marker.

    """
    if os.environ.get(E2E_ENV) == "1":
        return
    here = Path(__file__).resolve().parent
    skip = pytest.mark.skip(reason=f"opt-in live MPI harness: set {E2E_ENV}=1 to run")
    for item in items:
        if here in Path(str(item.path)).resolve().parents:
            item.add_marker(skip)


@dataclass(frozen=True)
class LiveRun:
    """One completed child run: its directory, exit code and captured streams.

    Attributes
    ----------
    directory : Path
        The fresh run directory holding ``input.in`` and pyKMC's output files.
    returncode : int
        Exit status of the ``mpirun`` process.
    stdout, stderr : str
        Captured streams of the whole MPI job.
    n_ranks : int
        Number of ranks launched (manager plus workers).

    """

    directory: Path
    returncode: int
    stdout: str
    stderr: str
    n_ranks: int

    @property
    def log(self) -> str:
        """Return ``pykmc.log`` without ANSI colour codes.

        Returns
        -------
        str
            The log text with every ``ESC[...m`` sequence removed.

        """
        import re

        return re.sub(r"\x1b\[[0-9;]*m", "", (self.directory / "pykmc.log").read_text())


@pytest.fixture(scope="session")
def launcher() -> list[str]:
    """Return the MPI launcher command, skipping when none is available.

    Returns
    -------
    list of str
        ``PYKMC_MPIRUN`` split with ``shlex``, or ``[mpirun]`` from ``PATH``.

    """
    spec = os.environ.get("PYKMC_MPIRUN")
    if spec:
        return shlex.split(spec)
    found = shutil.which("mpirun")
    if found is None:
        pytest.skip("no mpirun on PATH and PYKMC_MPIRUN is not set")
    return [found]


@pytest.fixture(scope="session")
def si_vac_example() -> Path:
    """Return ``examples/Si_vac`` (potential and initial configuration).

    Returns
    -------
    Path
        The example directory, once both input files are known to exist.

    """
    for name in ("initial_config.xyz", "Si.sw"):
        if not (SI_VAC / name).is_file():
            pytest.skip(f"{SI_VAC / name} not present")
    return SI_VAC


@pytest.fixture
def run_si_vac(
    tmp_path: Path, launcher: list[str], si_vac_example: Path
) -> Callable[..., LiveRun]:
    """Return a runner that launches one bounded Si_vac case in ``tmp_path``.

    Parameters
    ----------
    tmp_path : Path
        pytest's fresh temporary directory for this test.
    launcher : list of str
        The MPI launcher command from the :func:`launcher` fixture.
    si_vac_example : Path
        The example directory from the :func:`si_vac_example` fixture.

    Returns
    -------
    Callable[..., LiveRun]
        ``run(*, style, n_sessions, ranks_per_worker=1, n_steps=3, nsearch=2)``;
        see :func:`run` below.

    """
    timeout = float(os.environ.get("PYKMC_E2E_TIMEOUT", "900"))

    def run(
        *,
        style: str,
        n_sessions: int,
        ranks_per_worker: int = 1,
        n_steps: int = 3,
        nsearch: int = 2,
    ) -> LiveRun:
        """Launch one bounded case and return its captured outcome.

        Launches ``1 + n_sessions * ranks_per_worker`` ranks (rank 0 is the
        manager), writes ``input.in`` from :data:`SI_VAC_INPUT` and copies the
        example files next to it. A child that exceeds the time budget is
        killed and the test fails with the captured output.

        Parameters
        ----------
        style : str
            ``[RateConstant] style`` (``constant`` or ``htst``).
        n_sessions : int
            Number of worker sessions (``[Control] n_sessions``).
        ranks_per_worker : int, optional
            MPI ranks per worker session (default 1).
        n_steps : int, optional
            KMC steps to run (default 3).
        nsearch : int, optional
            pARTn searches per new environment (default 2).

        Returns
        -------
        LiveRun
            The run directory, exit code, captured streams and rank count.

        """
        n_ranks = 1 + n_sessions * ranks_per_worker
        directory = tmp_path / f"{style}_s{n_sessions}_n{n_ranks}"
        directory.mkdir()
        for name in ("initial_config.xyz", "Si.sw"):
            shutil.copy(si_vac_example / name, directory / name)
        (directory / "input.in").write_text(
            SI_VAC_INPUT.format(
                n_steps=n_steps,
                n_sessions=n_sessions,
                nsearch=nsearch,
                zseed=ZSEED,
                style=style,
            )
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
        )
        env["PYKMC_E2E_ROOT"] = str(REPO_ROOT)
        env["PYKMC_E2E_SEED"] = str(PY_SEED)
        cmd = [*launcher, "-n", str(n_ranks), sys.executable, "-c", _CHILD]
        try:
            proc = subprocess.run(
                cmd,
                cwd=directory,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            pytest.fail(
                f"live run exceeded {timeout:.0f} s in {directory}\n"
                f"stdout:\n{exc.stdout}\nstderr:\n{exc.stderr}"
            )
        (directory / "child_stdout.txt").write_text(proc.stdout)
        (directory / "child_stderr.txt").write_text(proc.stderr)
        return LiveRun(directory, proc.returncode, proc.stdout, proc.stderr, n_ranks)

    return run
