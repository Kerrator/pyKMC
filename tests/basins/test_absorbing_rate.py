"""Regression test for the basin absorbing-event rate stored in the connectivity table.

``RateConstant.compute_rate`` returns a ``RateComponents`` object, not a scalar. The basin
connectivity table's ``k_forward`` column is a float that gets summed
(``BasinsGenericEvents.exit`` computes ``k_tot = ...["k_forward"].sum()``), so the basin
must store the scalar ``.rate``. Storing the whole ``RateComponents`` object made the
``.sum()`` raise. This pins the scalar contract without needing the MPI manager.
"""

from typing import Any

import pandas as pd

from pykmc.basins import BasinsGenericEvents


def test_absorbing_rate_returns_scalar_float(
    config_Cu: Any,
    reference_table_Cu_fake: Any,
    visited_environments_Cu: Any,
) -> None:
    """The basin absorbing rate is the scalar ``.rate`` (float), not a RateComponents."""
    basin = BasinsGenericEvents(
        config=config_Cu,
        reference_table=reference_table_Cu_fake,
        known_environments=visited_environments_Cu,
        manager=None,
    )

    k = basin._absorbing_rate(0.5)

    assert isinstance(k, float), f"absorbing rate must be a float, got {type(k)!r}"
    # connectivity 'k_forward' is summed in BasinsGenericEvents.exit -> must stay numeric
    assert pd.Series([k, k]).sum() == k + k
