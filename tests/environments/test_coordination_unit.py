from pykmc.environments import coordination


def test_coordination_classifies_by_threshold():
    # atom 0 has 12 neighbours (==threshold), atom 1 has 11 (<), atom 2 has 12,
    # atom 3 has 13 (>) -- covers both sides of the `>=` boundary
    neighbors_list = [
        list(range(12)),
        list(range(11)),
        list(range(12)),
        list(range(13)),
    ]
    result = coordination(neighbors_list, threshold=12)
    assert result == ["crystal", "noncrystal", "crystal", "crystal"]
