import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from experiments.finalization_4gpu.placement import make_manifest
from experiments.finalization_4gpu.tests.test_launch import environment
from experiments.q5_topology_revalidation_4gpu import cli
from experiments.q5_topology_revalidation_4gpu.analysis import trend
from experiments.q5_topology_revalidation_4gpu.contracts import (
    POLICIES,
    TOKEN_SHA,
    assert_m0,
    assert_workload,
    audit_waits,
    dependency_predecessors,
    load,
    normalized_rows,
    save,
    sha,
    validate_p0_review,
    validate_schedule,
)
from experiments.q5_topology_revalidation_4gpu.mechanism import trace_facts


def args(root):
    return SimpleNamespace(
        run_root=str(root),
        run_id="CPU_SYNTHETIC",
        project=str(cli.PROJECT),
        python="python",
        placement_manifest=str(root / "placement.json"),
        source_manifest=str(root / "source.json"),
        data=str(root / "tokens.npy"),
        config="experiments/fsdp_interview/configs/primary.yaml",
        nsys="/qualified/nsys",
        timeout=600,
    )


def rows(rank, policy="C0_guarded"):
    tails, result = {}, []
    for ordinal, (kind, family, channel, pg) in enumerate(
        (
            ("fsdp_ag", "fsdp_parameter_ag", "fsdp_ag", "fsdp"),
            ("fsdp_rs", "fsdp_gradient_rs", "fsdp_rs", "fsdp"),
            ("sp_ag", "sp_activation_ag", "sp_ag", "tp"),
        )
    ):
        members = [rank % 2, rank % 2 + 2] if pg == "fsdp" else [rank // 2 * 2, rank // 2 * 2 + 1]
        deps = dependency_predecessors("C0" if policy == "C0_guarded" else policy, family, tails)
        result.append(
            dict(
                rank=rank,
                world_size=4,
                process_group_label=pg,
                collective_type=kind,
                unit_id=1,
                phase="FWD",
                channel=channel,
                input_shape=[2],
                output_shapes=[[4]],
                dtype="torch.float32",
                group_size=2,
                group_role="CPU_SYNTHETIC",
                group_members=members,
                collective_sequence=ordinal,
                host_ordinal=ordinal,
                runtime_microstep=1,
                input_bytes=8,
                output_bytes=16,
                family=family,
                status="SUBMITTED",
                dependencies=[
                    dict(
                        type="EXPERIMENT_COMM_ORDER",
                        producer_host_ordinal=value["ordinal"],
                        cpu_synchronize=False,
                    )
                    for value in deps
                ],
            )
        )
        tails[channel] = dict(ordinal=ordinal, family=family)
    return result


def write_events(root, policy):
    paths = []
    for rank in range(4):
        path = root / f"rank{rank}.jsonl"
        path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows(rank, policy)), encoding="utf-8"
        )
        paths.append(path)
    return paths


class ProtocolTests(unittest.TestCase):
    def test_physical_residency_does_not_infer_gemm_or_wire(self):
        def parsed(serial):
            collectives, kernels = [], []
            for rank in range(4):
                for index, row in enumerate(rows(rank)):
                    start, end = (index * 10, index * 10 + 5) if serial else (index, 10 - index)
                    row.update(
                        trace_status="MATCHED",
                        runtime_op_id=index,
                        gpu_start_timestamp=start,
                        gpu_end_timestamp=end,
                        cpu_nsys_submit_ns=index,
                    )
                    collectives.append(row)
                    kernels.append(
                        dict(
                            is_nccl=True,
                            status="MATCHED",
                            rank=rank,
                            runtime_op_id=index,
                            kernel_rowid=rank * 10 + index,
                            gpu_start_ns=start,
                            gpu_end_ns=end,
                            pid=100 + rank,
                            device_id=rank,
                            stream_id=20 + index,
                            correlation_id=index,
                        )
                    )
            return dict(collectives=collectives, kernels=kernels)

        concurrent = trace_facts(parsed(False))
        serial = trace_facts(parsed(True))
        self.assertTrue(
            all(row["maximum_nccl_residency_depth"] == 3 for row in concurrent["per_rank"])
        )
        self.assertTrue(all(row["maximum_nccl_residency_depth"] == 1 for row in serial["per_rank"]))
        self.assertTrue(all(row["fsdp_nonfsdp_residency_ms"] == 0 for row in serial["per_rank"]))
        self.assertNotIn("gemm_overlap", concurrent)
        wrong = parsed(True)
        wrong["collectives"][0]["trace_status"] = "UNMATCHED"
        with self.assertRaises(ValueError):
            trace_facts(wrong)

    def test_formal_nine_runs_balanced_and_guarded(self):
        plan = cli.plan(args(Path("CPU_SYNTHETIC")), "benchmark")
        self.assertEqual(len(plan), 9)
        for repeat in range(3):
            self.assertEqual(
                {row["policy"] for row in plan if row["repeat"] == repeat}, set(POLICIES)
            )
        for row in plan:
            self.assertEqual((row["warmup"], row["steps"], row["mapping_id"]), (10, 30, "M0"))
            self.assertNotIn("--record-events", row["argv"])
            self.assertEqual("--guarded-baseline" in row["argv"], row["policy"] == "C0_guarded")
            self.assertIn("--contract", row["argv"])

    def test_two_short_profiles_not_old_minimal_or_c0_fast(self):
        plan = cli.plan(args(Path("CPU_SYNTHETIC")), "profile")
        self.assertEqual([row["policy"] for row in plan], ["C0_guarded", "C1"])
        for row in plan:
            self.assertEqual((row["warmup"], row["steps"]), (10, 1))
            self.assertIn("--trace=cuda,nvtx", row["argv"])
            self.assertIn("--profile", row["argv"])
            self.assertNotIn("--gemm-shapes", row["argv"])

    def test_tiny_contracts_are_distinct_from_full_contract(self):
        baseline = cli.plan(args(Path("CPU_SYNTHETIC")), "bootstrap")
        self.assertEqual(
            [row.get("mode", "full") for row in baseline],
            ["fp32", "same_start", "overflow", "full"],
        )
        gated = cli.plan(args(Path("CPU_SYNTHETIC")), "gates")
        self.assertEqual(len(gated), 9)
        self.assertEqual(len({row["argv"][-1] for row in gated}), 3)

    def test_m0_requires_measured_two_nv2_not_old_hypothesis(self):
        manifest = make_manifest(environment())
        assert_m0(manifest)
        value = environment()
        u = value["allowed_gpu_uuids"]
        value["topology"][u[0]][u[2]] = value["topology"][u[2]][u[0]] = "NV1"
        with self.assertRaises(ValueError):
            assert_m0(make_manifest(value))

    def test_fixed_full_workload_rejects_tiny_or_changed_flags(self):
        source = Path("experiments/fsdp_interview/configs/primary.yaml")
        config = assert_workload(source)
        import yaml

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "wrong.yaml"
            for section, key, value in (
                ("model", "layers", 4),
                ("runtime", "forward_prefetch", False),
                ("training", "local_batch_size", 1),
            ):
                wrong = copy.deepcopy(config)
                wrong[section][key] = value
                path.write_text(yaml.safe_dump(wrong), encoding="utf-8")
                with self.assertRaises(ValueError):
                    assert_workload(path)

    def test_original_wait_edges_and_global_contract(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for policy, waits in (("C0_guarded", 0), ("C1", 8), ("C2", 8)):
                directory = root / policy
                directory.mkdir()
                paths = write_events(directory, policy)
                self.assertEqual(validate_schedule(paths)["status"], "QUALIFIED")
                self.assertEqual(audit_waits(paths, policy)["added_waits"], waits)
                normalized_rows(paths)

    def test_reject_future_wait_or_unexpected_cpu_sync(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = write_events(Path(folder), "C1")
            values = [json.loads(line) for line in paths[0].read_text().splitlines()]
            values[1]["dependencies"][0]["producer_host_ordinal"] = 999
            paths[0].write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
            with self.assertRaises(ValueError):
                audit_waits(paths, "C1")

    def test_reject_missing_rank_and_changed_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = write_events(Path(folder), "C0_guarded")
            with self.assertRaises(ValueError):
                validate_schedule(paths[:-1])
            values = rows(0)
            values[0]["input_shape"] = [3]
            paths[0].write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_schedule(paths)

    def test_review_preserves_fail_and_rejects_unresolved_or_stale_raw(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, placement = root / "source.json", root / "placement.json"
            source.write_text("source", encoding="utf-8")
            placement.write_text("placement", encoding="utf-8")
            evidence = []
            for name in ("capture_summary", "replay", "control"):
                path = root / (name + ".json")
                path.write_text(name, encoding="utf-8")
                evidence.append(dict(label=name, path=str(path), sha256=sha(path)))
            value = dict(
                source_manifest_sha256=sha(source),
                placement_manifest_sha256=sha(placement),
                input_sha256=TOKEN_SHA,
                classification="EXPECTED_NUMERICAL_DRIFT_SUPPORTED",
                strict_independent_status="FAIL",
                training_runtime_bug_confirmed=False,
                continuation_authorized=True,
                signed_by="root_agent",
                scope="bounded_performance_with_retained_strict_fail",
                limits=["short independent drift"],
                evidence=evidence,
            )
            path = root / "review.json"
            save(path, value)
            self.assertEqual(
                validate_p0_review(path, source, placement)["strict_independent_status"], "FAIL"
            )
            value["classification"] = "UNRESOLVED"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                validate_p0_review(path, source, placement)

    def test_no_overwrite_and_trend_does_not_claim_independence(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "immutable.json"
            save(path, dict(status="FAIL"))
            with self.assertRaises(FileExistsError):
                save(path, dict(status="PASS"))
            self.assertEqual(load(path)["status"], "FAIL")
        self.assertFalse(trend([10.0] * 30)["descriptive_drift_flag"])
        self.assertTrue(trend(list(range(10, 40)))["descriptive_drift_flag"])


if __name__ == "__main__":
    unittest.main()
