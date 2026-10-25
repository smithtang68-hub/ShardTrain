"""Pure rank arithmetic. Rank ordering: fsdp coordinate first, then TP."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RankCoordinate:
    fsdp: int
    tp: int


@dataclass(frozen=True)
class Topology:
    fsdp_degree: int
    tp_degree: int

    def __post_init__(self) -> None:
        for name, value in (("fsdp_degree", self.fsdp_degree), ("tp_degree", self.tp_degree)):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @property
    def world_size(self) -> int:
        return self.fsdp_degree * self.tp_degree

    def coordinate(self, rank: int) -> RankCoordinate:
        if type(rank) is not int or not 0 <= rank < self.world_size:
            raise ValueError(f"rank must be in [0, {self.world_size}); got {rank!r}")
        return RankCoordinate(*divmod(rank, self.tp_degree))

    def rank(self, fsdp: int, tp: int) -> int:
        if type(fsdp) is not int or not 0 <= fsdp < self.fsdp_degree:
            raise ValueError(f"Invalid FSDP coordinate: {fsdp!r}")
        if type(tp) is not int or not 0 <= tp < self.tp_degree:
            raise ValueError(f"Invalid TP coordinate: {tp!r}")
        return fsdp * self.tp_degree + tp

    @property
    def tp_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.rank(f, t) for t in range(self.tp_degree)) for f in range(self.fsdp_degree)
        )

    @property
    def fsdp_groups(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.rank(f, t) for f in range(self.fsdp_degree)) for t in range(self.tp_degree)
        )

    @property
    def creation_order(self) -> tuple[tuple[str, tuple[int, ...]], ...]:
        # Every rank walks this entire sequence, including groups it does not belong to.
        # Duplicate memberships are deliberately distinct communicators: parameter
        # AG must not inherit gradient RS completion, and replicated-gradient SUM
        # must not order activation SP collectives through the same communicator.
        return tuple(
            (role, ranks)
            for role, groups in (
                ("tp", self.tp_groups),
                ("fsdp", self.fsdp_groups),
                ("tp_grad", self.tp_groups),
                ("fsdp_grad", self.fsdp_groups),
            )
            for ranks in groups
        )

    def groups_for_rank(self, rank: int) -> dict[str, tuple[int, ...]]:
        coord = self.coordinate(rank)
        return {
            "tp": self.tp_groups[coord.fsdp],
            "fsdp": self.fsdp_groups[coord.tp],
            "tp_grad": self.tp_groups[coord.fsdp],
            "fsdp_grad": self.fsdp_groups[coord.tp],
        }
