import json
import logging

import pytest
import torch.distributed as dist

from sharded_runtime.config import ParallelConfig
from sharded_runtime.distributed.context import DistributedContext, WorkerEnvironment
from sharded_runtime.observability.logging import close_logger, make_rank_logger


def test_single_process_context_closes_and_rejects_reuse(tmp_path):
    logger = make_rank_logger(tmp_path, 0)
    ctx = DistributedContext(ParallelConfig(), WorkerEnvironment(), 5, logger)
    assert ctx.sp_group is ctx.tp_group
    assert ctx.smoke_collectives()["fsdp"]["sum"] == 1
    assert not dist.is_initialized()
    ctx.close()
    ctx.close()
    with pytest.raises(RuntimeError, match="closed"):
        ctx.smoke_collectives()
    close_logger(logger)


def test_gradient_communicators_have_independent_lifetimes_and_fixed_creation(monkeypatch):
    parallel = ParallelConfig(fsdp_degree=2, tp_degree=2, backend="gloo")
    worker = WorkerEnvironment(rank=1, local_rank=1, world_size=4, launched=True)
    state = {"initialized": False}
    created, destroyed = [], []
    monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
    monkeypatch.setenv("MASTER_PORT", "29501")
    monkeypatch.setattr("sharded_runtime.distributed.context.sys.platform", "linux")
    monkeypatch.setattr(dist, "is_available", lambda: True)
    monkeypatch.setattr(dist, "is_gloo_available", lambda: True)
    monkeypatch.setattr(dist, "is_initialized", lambda: state["initialized"])
    monkeypatch.setattr(dist, "init_process_group", lambda **kwargs: state.update(initialized=True))

    def new_group(*, ranks, **kwargs):
        group = object()
        created.append((tuple(ranks), group))
        return group

    def destroy(group=None):
        destroyed.append(group)
        if group is None:
            state["initialized"] = False

    monkeypatch.setattr(dist, "new_group", new_group)
    monkeypatch.setattr(dist, "destroy_process_group", destroy)
    ctx = DistributedContext(parallel, worker, 5, logging.getLogger("context_test"))
    assert [ranks for ranks, _ in created] == [ranks for _, ranks in ctx.topology.creation_order]
    assert ctx.tp_group is created[0][1]
    assert ctx.tp_grad_group is created[4][1]
    assert ctx.fsdp_group is created[3][1]
    assert ctx.fsdp_grad_group is created[7][1]
    assert ctx.tp_grad_group is not ctx.tp_group
    assert ctx.fsdp_grad_group is not ctx.fsdp_group
    ctx.close()
    assert destroyed == [created[index][1] for index in (7, 4, 3, 0)] + [None]
    assert ctx.tp_grad_group is ctx.fsdp_grad_group is None
    assert not state["initialized"]


def test_each_rank_exception_log_and_no_duplicate_handlers(tmp_path, capsys):
    logger = make_rank_logger(tmp_path, 1)
    logger = make_rank_logger(tmp_path, 1)
    logger.info("rank one diagnostic")
    try:
        raise ValueError("sample failure")
    except ValueError:
        logger.exception("failed")
    close_logger(logger)
    rows = [json.loads(row) for row in (tmp_path / "rank1.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["rank"] == 1 for row in rows)
    assert "ValueError: sample failure" in rows[1]["exception"]
    stderr = capsys.readouterr().err
    assert "sample failure" in stderr
    assert "rank one diagnostic" not in stderr
