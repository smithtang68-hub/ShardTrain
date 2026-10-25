import pytest
from hypothesis import given
from hypothesis import strategies as st

from sharded_runtime.distributed.topology import Topology


@given(fsdp=st.integers(1, 12), tp=st.integers(1, 12))
def test_topology_partition_and_orthogonality(fsdp, tp):
    topology = Topology(fsdp, tp)
    world = set(range(fsdp * tp))
    for groups in (topology.tp_groups, topology.fsdp_groups):
        assert sorted(rank for group in groups for rank in group) == sorted(world)
    for rank in world:
        coordinate = topology.coordinate(rank)
        assert topology.rank(coordinate.fsdp, coordinate.tp) == rank
        groups = topology.groups_for_rank(rank)
        assert set(groups["tp"]) & set(groups["fsdp"]) == {rank}
        assert len(groups["tp"]) == tp
        assert len(groups["fsdp"]) == fsdp
        assert groups["tp_grad"] == groups["tp"]
        assert groups["fsdp_grad"] == groups["fsdp"]
    # The sequence is independent of the current rank, including nonmember groups.
    assert len(topology.creation_order) == 2 * (fsdp + tp)


def test_four_rank_order():
    topology = Topology(2, 2)
    assert topology.tp_groups == ((0, 1), (2, 3))
    assert topology.fsdp_groups == ((0, 2), (1, 3))
    assert topology.creation_order == (
        ("tp", (0, 1)),
        ("tp", (2, 3)),
        ("fsdp", (0, 2)),
        ("fsdp", (1, 3)),
        ("tp_grad", (0, 1)),
        ("tp_grad", (2, 3)),
        ("fsdp_grad", (0, 2)),
        ("fsdp_grad", (1, 3)),
    )


@pytest.mark.parametrize("fsdp,tp", [(0, 1), (1, 0), (-1, 2), (True, 1), (1.5, 1)])
def test_invalid_degrees(fsdp, tp):
    with pytest.raises(ValueError):
        Topology(fsdp, tp)


def test_invalid_ranks_and_coordinates():
    topology = Topology(2, 2)
    for rank in (-1, 4, True, 1.5):
        with pytest.raises(ValueError):
            topology.coordinate(rank)
    for coordinate in ((2, 0), (0, 2), (-1, 0), (0, True)):
        with pytest.raises(ValueError):
            topology.rank(*coordinate)
