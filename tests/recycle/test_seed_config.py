"""The [Control] seed key must parse into config.control.seed."""
import textwrap
from pykmc.config import Config


_MINIMAL_INI = """
[Control]
initial_config = ./initial_config.xyz
n_steps = 1
engine = lammps
seed = 12345

[Lammps]
pair_style = eam/alloy
pair_coeff = * * ./pot.eam Ni
min_style = cg
minimize = 1e-10 1e-12 10000 10000

[AtomicEnvironment]
style = cna/graph
rnei = 3.0
rcut = 6.5

[EventSearch]
style = partn
nsearch = 10
emax_event = 5.0
emin_event = 0.0

[pARTn]

[RateConstant]
style = constant
k0 = 10.0
T = 300.0

[PSR]
style = ira
matching_score_thr = 0.4

[IRA]
kmax_factor = 2.0
"""


def test_seed_parsed(tmp_path):
    ini = tmp_path / "input.in"
    ini.write_text(textwrap.dedent(_MINIMAL_INI))
    config = Config.from_ini_file(str(ini))
    assert config.control.seed == 12345


def test_seed_defaults_to_none(tmp_path):
    ini_text = _MINIMAL_INI.replace("seed = 12345\n", "")
    ini = tmp_path / "input.in"
    ini.write_text(textwrap.dedent(ini_text))
    config = Config.from_ini_file(str(ini))
    assert config.control.seed is None
