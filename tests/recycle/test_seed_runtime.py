"""Constructing KMC with a seed must make Python's RNG reproducible."""
import random
import textwrap
from pykmc.config import Config
from pykmc.kmc import KMC

from tests.recycle.test_seed_config import _MINIMAL_INI


def _config_with_seed(tmp_path, seed_line):
    ini_text = _MINIMAL_INI.replace("seed = 12345\n", seed_line)
    ini = tmp_path / "input.in"
    ini.write_text(textwrap.dedent(ini_text))
    return Config.from_ini_file(str(ini))


def test_seed_makes_random_reproducible(tmp_path):
    config = _config_with_seed(tmp_path, "seed = 777\n")
    KMC(config)
    first = [random.random() for _ in range(5)]
    KMC(config)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_no_seed_does_not_pin_rng(tmp_path):
    # With no seed, KMC.__init__ must not call random.seed (so a pre-set state survives).
    config = _config_with_seed(tmp_path, "")
    random.seed(999)
    sentinel = random.random()
    random.seed(999)
    KMC(config)
    assert random.random() == sentinel  # KMC did not reseed
