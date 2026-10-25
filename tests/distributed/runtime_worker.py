"""P0 and four-layer acceptance worker; also the NCCL Reference comparator."""

import argparse
import gc
import weakref

import torch

from sharded_runtime.config import load_config
from sharded_runtime.launch.bootstrap import WorkerSession
from sharded_runtime.models.reference import build_reference_model, synthetic_batch
from sharded_runtime.observability.metadata import write_json
from sharded_runtime.observability.profiling import ProfileCapture
from sharded_runtime.runtime import ShardedRuntime
from sharded_runtime.training.reference import compare_tensor, dense_oracle


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config")
    parser.add_argument("--input-grad", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-mode", choices=["torch", "nsight", "off"], default="torch")
    args, overrides = parser.parse_known_args()
    config = load_config(args.config, overrides)
    with WorkerSession(config, "runtime") as session:
        ctx = session.context
        f, t = ctx.coordinate.fsdp, ctx.coordinate.tp
        dense = build_reference_model(config.model, config.training.seed)
        oracle_params = {
            name: p.detach().clone().requires_grad_() for name, p in dense.named_parameters()
        }
        # Deliberately different initial seeds on nonzero ranks: registration broadcasts WORLD0.
        model = build_reference_model(config.model, config.training.seed + session.worker.rank)
        runtime = ShardedRuntime(
            device=ctx.device,
            reshard_after_forward=config.runtime.reshard_after_forward,
            forward_prefetch=config.runtime.forward_prefetch,
            backward_prefetch=config.runtime.backward_prefetch,
            async_grad_rs=config.runtime.async_grad_rs,
            communication_policy=config.runtime.communication_policy,
            directory=session.directory,
        )
        runtime.register(model, ctx, sp=config.parallel.sp_enabled)
        optimizer = torch.optim.SGD(runtime.sharded_parameters(), lr=config.training.lr)
        runtime.bind_optimizer(optimizer)
        rows, saved, block_outputs = [], [], {}
        sublayer_outputs = {}
        retired_refs = {}
        for unit in runtime.units:

            def snapshot(module, inputs, unit=unit):
                retired_refs[unit.unit_id] = [
                    weakref.ref(lease.tensor.untyped_storage())
                    for lease in unit.lease.full_param_storages.values()
                ]

            unit.block.register_forward_pre_hook(snapshot)
            unit.block.register_forward_hook(
                lambda module, inputs, output, uid=unit.unit_id: block_outputs.update(
                    {uid: output.detach().clone()}
                )
            )
            for role in ("attention", "mlp"):
                getattr(unit.block, role).register_forward_hook(
                    lambda module,
                    inputs,
                    output,
                    uid=unit.unit_id,
                    role=role: sublayer_outputs.update({(uid, role): output.detach().clone()})
                )

        def save(tensor):
            live = {
                lease.tensor.untyped_storage().data_ptr()
                for unit in runtime.units
                if unit.lease is not None
                for lease in unit.lease.full_param_storages.values()
            }
            assert tensor.untyped_storage().data_ptr() not in live, "Autograd saved FullParam"
            saved.append(
                dict(
                    shape=list(tensor.shape),
                    stride=list(tensor.stride()),
                    storage_bytes=tensor.untyped_storage().nbytes(),
                    tensor_bytes=tensor.numel() * tensor.element_size(),
                )
            )
            return tensor

        memory = []
        profiler = (
            ProfileCapture(
                session.directory, session.worker.rank, ctx.device, mode=args.profile_mode
            )
            if args.profile and ctx.device.type == "cuda"
            else None
        )
        if profiler:
            profiler.start()
        try:
            for step in range(config.training.steps):
                with runtime.compute_context():
                    runtime.begin_step(step)
                    optimizer.zero_grad(set_to_none=True)
                    for p in oracle_params.values():
                        p.grad = None
                    inputs = [
                        synthetic_batch(
                            config.model,
                            config.training.local_batch_size,
                            9000 + step * 100 + coord,
                        )
                        for coord in range(config.parallel.fsdp_degree)
                    ]
                    targets = [
                        synthetic_batch(
                            config.model,
                            config.training.local_batch_size,
                            19000 + step * 100 + coord,
                        )
                        for coord in range(config.parallel.fsdp_degree)
                    ]
                    full_x = torch.cat(inputs).requires_grad_(args.input_grad)
                    full_target = torch.cat(targets)
                    # Keep oracle CPU and independent of Runtime operators/communication.
                    expected_sublayers = {}
                    expected, expected_blocks = dense_oracle(
                        full_x, oracle_params, config.model, sublayers=expected_sublayers
                    )
                    expected_loss = (expected - full_target).square().mean()
                    expected_loss.backward()
                    x = (
                        runtime.input_for_rank(inputs[f].to(ctx.device))
                        .detach()
                        .requires_grad_(args.input_grad)
                    )
                    target = runtime.input_for_rank(targets[f].to(ctx.device))
                    with torch.autograd.graph.saved_tensors_hooks(save, lambda tensor: tensor):
                        actual = model(x)
                        gc.collect()
                        if config.runtime.reshard_after_forward:
                            for uid in range(config.model.layers - 1):
                                assert all(ref() is None for ref in retired_refs[uid]), (
                                    f"FWD FullParam retained U{uid}"
                                )
                        assert runtime.units[-1].state == "FWD_RETAINED"
                        replica_slice = slice(
                            f * config.training.local_batch_size,
                            (f + 1) * config.training.local_batch_size,
                        )
                        for uid, output in block_outputs.items():
                            reference = runtime.input_for_rank(
                                expected_blocks[uid][replica_slice].to(ctx.device)
                            )
                            rows.append(
                                compare_tensor(output, reference, f"step{step}/block{uid}/output")
                            )
                        for (uid, role), output in sublayer_outputs.items():
                            reference = runtime.input_for_rank(
                                expected_sublayers[uid, role][replica_slice].to(ctx.device)
                            )
                            rows.append(
                                compare_tensor(output, reference, f"step{step}/block{uid}/{role}")
                            )
                        loss = runtime.local_loss(actual, target)
                        loss.backward()
                    runtime.finalize_backward()
                    rows.append(
                        compare_tensor(
                            runtime.reported_loss(loss), expected_loss, f"step{step}/global_loss"
                        )
                    )
                    if args.input_grad:
                        # Input adjoint is replica loss dX; /F belongs to parameter RS, not dX.
                        reference_dx = runtime.input_for_rank(
                            full_x.grad[replica_slice].to(ctx.device) * config.parallel.fsdp_degree
                        )
                        rows.append(compare_tensor(x.grad, reference_dx, f"step{step}/input_grad"))
                    reconstructed_grads = runtime.reconstruct_tp_local(gradients=True)
                    for name, handle in runtime.registry.items():
                        spec = handle.spec
                        expected_grad = oracle_params[name].grad
                        dim = {"column": 0, "row": 1, "replicated": None}[spec.tp_layout]
                        if dim is not None:
                            size = expected_grad.shape[dim] // config.parallel.tp_degree
                            expected_grad = expected_grad.narrow(dim, t * size, size)
                        rows.append(
                            compare_tensor(
                                reconstructed_grads[name],
                                expected_grad,
                                f"step{step}/{name}/TP_grad",
                            )
                        )
                        q, valid = spec.fsdp_shard_numel, spec.valid_len(f)
                        expected_shard = torch.zeros(q)
                        expected_shard[:valid].copy_(expected_grad.reshape(-1)[f * q :][:valid])
                        rows.append(
                            compare_tensor(
                                handle.local_shard.grad,
                                expected_shard,
                                f"step{step}/{name}/grad_shard",
                            )
                        )
                        if spec.tp_layout == "replicated" and runtime.tp.degree > 1:
                            peers = runtime.verification_gather(
                                handle.local_shard, ctx.tp_group, "tp"
                            )
                            for peer in peers:
                                rows.append(
                                    compare_tensor(
                                        handle.local_shard,
                                        peer,
                                        f"step{step}/{name}/norm_replica",
                                        atol=0,
                                        rtol=0,
                                    )
                                )
                    optimizer.step()
                    with torch.no_grad():
                        for p in oracle_params.values():
                            p.sub_(config.training.lr * p.grad)
                    runtime.end_step()
                    for name, value in runtime.reconstruct_tp_local().items():
                        spec = runtime.registry[name].spec
                        expected_param = oracle_params[name]
                        dim = {"column": 0, "row": 1, "replicated": None}[spec.tp_layout]
                        if dim is not None:
                            size = expected_param.shape[dim] // config.parallel.tp_degree
                            expected_param = expected_param.narrow(dim, t * size, size)
                        rows.append(
                            compare_tensor(value, expected_param, f"step{step}/{name}/update")
                        )
                    if ctx.device.type == "cuda":
                        torch.cuda.synchronize(ctx.device)
                        memory.append(
                            dict(
                                step=step,
                                allocated=torch.cuda.memory_allocated(ctx.device),
                                reserved=torch.cuda.memory_reserved(ctx.device),
                                max_allocated=torch.cuda.max_memory_allocated(ctx.device),
                            )
                        )
                session.logger.info(
                    "runtime correctness step=%d comparisons=%d passed", step, len(rows)
                )
                if profiler:
                    profiler.step(step)
        except BaseException as exc:
            runtime.fail(exc)
            write_json(
                session.directory / f"failure.rank{session.worker.rank}.json",
                dict(error=repr(exc), last_events=runtime.comm.events[-30:]),
            )
            raise
        finally:
            if profiler:
                profiler.close()
        write_json(
            session.directory / f"correctness.rank{session.worker.rank}.json",
            dict(
                status="passed",
                comparisons=rows,
                memory=memory,
                saved_tensors=saved,
                pg_order_gpu_waits=runtime.comm.gpu_wait_timings() if runtime.comm.streams else [],
                backend=config.parallel.backend,
                F=config.parallel.fsdp_degree,
                T=config.parallel.tp_degree,
                SP=config.parallel.sp_enabled,
                steps=config.training.steps,
                input_grad=args.input_grad,
                reshard=config.runtime.reshard_after_forward,
                forward_prefetch=config.runtime.forward_prefetch,
                backward_prefetch=config.runtime.backward_prefetch,
                async_grad_rs=config.runtime.async_grad_rs,
                communication_policy=config.runtime.communication_policy,
                leaf_identities_stable=True,
                live_buffers=len(runtime.comm.live_buffers),
            ),
        )


if __name__ == "__main__":
    main()
