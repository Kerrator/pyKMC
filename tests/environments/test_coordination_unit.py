from pykmc.environments import coordination


def test_coordination_classifies_by_threshold():
    # atom 0 has 12 neighbours, atom 1 has 11, atom 2 has 12
    neighbors_list = [list(range(12)), list(range(11)), list(range(12))]
    result = coordination(neighbors_list, threshold=12)
    assert result == ["crystal", "noncrystal", "crystal"]


def test_coordination_all_crystal_when_threshold_met():
    neighbors_list = [list(range(12)) for _ in range(5)]
    result = coordination(neighbors_list, threshold=12)
    assert result == ["crystal"] * 5
