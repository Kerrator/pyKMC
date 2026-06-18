"""Constructing KMC with a seed must make Python's and numpy's RNG reproducible."""

import random
import textwrap
import numpy as np
from pykmc.config import Config
from pykmc.kmc import KMC

from tests.test_config_seed import _MINIMAL_INI


def _config_with_seed(tmp_path, seed_line):
    ini_text = _MINIMAL_INI.replace("seed = 12345\n", seed_line)
    ini = tmp_path / "input.in"
    ini.write_text(textwrap.dedent(ini_text))
    return Config.from_ini_file(str(ini))


def test_seed_makes_random_reproducible(tmp_path):
    config = _config_with_seed(tmp_path, "seed = 777\n")
    KMC(config)
    first = [random.random() for _ in range(5)]
    first_np = np.random.random(5).tolist()
    KMC(config)
    second = [random.random() for _ in range(5)]
    second_np = np.random.random(5).tolist()
    assert first == second
    assert first_np == second_np


def test_no_seed_does_not_pin_rng(tmp_path):
    # With no seed, KMC.__init__ must not call random.seed or np.random.seed (so a
    # pre-set state survives).
    config = _config_with_seed(tmp_path, "")
    random.seed(999)
    np.random.seed(999)
    sentinel = random.random()
    sentinel_np = np.random.random()
    random.seed(999)
    np.random.seed(999)
    KMC(config)
    assert random.random() == sentinel  # KMC did not reseed
    assert np.random.random() == sentinel_np
