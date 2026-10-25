"""Reuse independent GPU fixtures; add actual M0/PG and lifecycle audit."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

from .contracts import CASES, save
from .worker import PROJECT, native_implementation_proof, native_tp1_scope, placement_audit


def validate_amp_fixture_steps(clean_steps):
    # The upstream expected scale/growth sequence is explicitly three clean steps.
    # Increasing its global constant would index beyond the frozen expectations.
    if clean_steps != 3:
        raise ValueError(
            "Existing AMP fixture requires3clean+skip+recovery; no constant-only expansion"
        )


def load_fixture(relative):
    spec = importlib.util.spec_from_file_location("_p1_fixture", PROJECT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--case", choices=CASES, required=True)
    p.add_argument("--kind", choices=["fp32", "fp16"], required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    c = CASES[args.case]
    rank = int(os.environ["RANK"])
    payload = dict(
        status="FAIL", case=args.case, kind=args.kind, performance_claim_allowed=False, errors=[]
    )
    result_path = Path(args.output) / f"gate_audit.rank{rank}.json"
    if result_path.exists():
        raise FileExistsError(result_path)
    base = ["--f", str(c["F"]), "--t", str(c["T"])]
    if c["SP"]:
        base += ["--sp"]
    if c["engine"] == "native":
        fixture = load_fixture("tests/distributed/qwen2_native_worker.py")
        original = fixture.build_native_qwen

        def builder(config, mesh, device, **kwargs):
            model, policy = original(config, mesh, device, **kwargs)
            payload["native_implementation"] = native_implementation_proof(model)
            payload["placement"] = placement_audit(
                args.case,
                {"tp": mesh.get_group("tp"), "fsdp": mesh.get_group("fsdp")},
                device=device,
            )
            policy.update(
                actual_fsdp_api="torch.distributed.fsdp.fully_shard", tp_plan_disabled=c["T"] == 1
            )
            return model, policy

        fixture.build_native_qwen = builder
        argv = base + [
            "--dtype",
            args.kind,
            "--optimizer",
            "adamw",
            "--accumulation",
            "2",
            "--output-dir",
            args.output,
        ]
        payload["scope"] = (
            "Native independent dense gradient/update; FP16 clean/skip/recovery; "
            "no direct dX/logits oracle"
        )
    else:
        relative = (
            "tests/distributed/qwen2_worker.py"
            if args.kind == "fp32"
            else "tests/distributed/qwen2_amp_worker.py"
        )
        fixture = load_fixture(relative)
        from experiments.finalization_4gpu.mapping_worker import RuntimeObserver

        observer = RuntimeObserver(payload)
        old_session, old_runtime = fixture.WorkerSession, fixture.ShardedRuntime

        class Session(old_session):
            def __enter__(self):
                value = super().__enter__()
                try:
                    ctx = self.context
                    payload["placement"] = placement_audit(
                        args.case,
                        {
                            "tp": ctx.tp_group,
                            "fsdp": ctx.fsdp_group,
                            "tp_grad": ctx.tp_grad_group,
                            "fsdp_grad": ctx.fsdp_grad_group,
                        },
                        device=ctx.device,
                    )
                except BaseException:
                    super().__exit__(*sys.exc_info())
                    raise
                return value

        class Runtime(old_runtime):
            def register(self, *v, **kw):
                result = super().register(*v, **kw)
                observer.registered(self)
                return result

            def finalize_backward(self, **kw):
                result = super().finalize_backward(**kw)
                observer.finalized(self)
                return result

            def end_step(self):
                result = super().end_step()
                observer.ended(self)
                return result

        fixture.WorkerSession, fixture.ShardedRuntime = Session, Runtime
        argv = base + [
            "--layers",
            "4",
            "--backend",
            "nccl",
            "--device",
            "cuda",
            "--policy",
            "stream_ordered",
            "--flags",
            "111",
            "--debug.output_dir",
            args.output,
        ]
        if args.kind == "fp32":
            argv += ["--steps", "10"]
            payload["scope"] = "Self FP32 independent HF logits/loss/dX/dW/shard/update; ten steps"
        else:
            if not c["SP"]:
                argv += ["--no-sp"]
            validate_amp_fixture_steps(fixture.CLEAN_STEPS)
            argv += ["--optimizer", "adamw"]
            payload["scope"] = (
                "Self FP16 three clean same-start updates plus overflow/recovery "
                "and actual-gradient optimizer replay; four effective updates, "
                "not independent rollout"
            )
    sys.argv = [str(PROJECT / relative) if c["engine"] != "native" else "native_oracle", *argv]
    try:
        with native_tp1_scope():
            fixture.main()
        payload["status"] = "PASS"
    except BaseException as error:
        payload["errors"].append(repr(error))
        raise
    finally:
        save(result_path, payload)


if __name__ == "__main__":
    main()
