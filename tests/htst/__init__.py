"""Pure-NumPy tests for the HTST numerical layer (no engine, no MPI).

The subpackage under test imports only NumPy and the standard library, but this
lane still needs a full install: ``import pykmc.htst`` runs ``pykmc/__init__.py``,
which loads pandas, mpi4py and ase. Only ``test_import_isolation.py`` (with a
stub parent package) exercises the subpackage without them.
"""
