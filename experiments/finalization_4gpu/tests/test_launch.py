"""CPU fixtures for placement safety and non-overwriting executable plans."""

import copy
import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from experiments.finalization_4gpu.audit import verify_source
from experiments.finalization_4gpu.cli import (
    execute_correctness,
    make_plan,
    parse_topology,
    save,
    sha,
    write_ledger,
)
from experiments.finalization_4gpu.placement import make_manifest, validate_manifest
from experiments.finalization_4gpu.reference import inventory_from_events, make_reference_plan


def environment(order=None):
    uuids = [f"GPU-{index:08x}-0000-0000-0000-000000000000" for index in range(4)]
    graph = {left: {} for left in uuids}
    labels = {
        (0, 1): "NV1",
        (0, 2): "NV2",
        (0, 3): "PHB",
        (1, 2): "NV1",
        (1, 3): "NV2",
        (2, 3): "PHB",
    }
    for (left, right), label in labels.items():
        graph[uuids[left]][uuids[right]] = label
        graph[uuids[right]][uuids[left]] = label
    return {
        "allowed_gpu_uuids": list(order or uuids),
        "devices": [
            {"uuid": value, "index": index, "pci_bus_id": f"0000:{index:02x}:00.0"}
            for index, value in enumerate(uuids)
        ],
        "topology": graph,
    }


class LauncherTests(unittest.TestCase):
    def test_all_24_bijections_keep_logical_mesh(self):
        baseline = environment()
        for order in itertools.permutations(baseline["allowed_gpu_uuids"]):
            result = make_manifest(environment(order))
            validate_manifest(result)
            self.assertEqual(result["logical_mesh"], [[0, 1], [2, 3]])
            for mapping in result["placements"].values():
                self.assertEqual(set(mapping["ordered_gpu_uuids"]), set(order))
                self.assertEqual([row["rank"] for row in mapping["ranks"]], list(range(4)))

    def test_measured_new_topology_is_tradeoff_not_old_hypothesis(self):
        result = make_manifest(environment())
        self.assertEqual(result["hypothesis_applicability"], "NOT_APPLICABLE_TOPOLOGY")
        self.assertEqual(result["placements"]["M1"]["status"], "PASS")
        self.assertEqual(result["placements"]["M1"]["name"], "M1_swap23_tradeoff")

    def test_old_hypothesis_detected_only_from_graph(self):
        value = environment()
        uuids = value["allowed_gpu_uuids"]
        for left, right, label in [(0, 3, "NV1"), (1, 3, "PHB")]:
            value["topology"][uuids[left]][uuids[right]] = label
            value["topology"][uuids[right]][uuids[left]] = label
        self.assertEqual(make_manifest(value)["hypothesis_applicability"], "PASS")

    def test_manifest_rejects_duplicates_missing_extra_groups_and_ranks(self):
        original = make_manifest(environment())
        mutations = [
            lambda m: m["logical_groups"]["tp"][0].reverse(),
            lambda m: m["placements"].pop("M1"),
            lambda m: m["placements"]["M1"]["ranks"].append(m["placements"]["M1"]["ranks"][0]),
            lambda m: m["placements"]["M0"]["groups"].update(extra=[]),
            lambda m: m["placements"]["M0"]["groups"]["tp"][0].update(
                ordered_logical_members=[1, 0]
            ),
            lambda m: m["placements"]["M0"]["groups"]["fsdp"][0].update(
                ordered_physical_uuids=["GPU-wrong", "GPU-wrong"]
            ),
            lambda m: m["placements"]["M1"]["ranks"][0].update(f=1),
        ]
        for mutate in mutations:
            candidate = copy.deepcopy(original)
            mutate(candidate)
            with self.assertRaises((ValueError, KeyError)):
                validate_manifest(candidate)

    def test_unknown_topology_cannot_be_qualified(self):
        value = environment()
        value["topology"] = {}
        with self.assertRaises(ValueError):
            make_manifest(value)

    def test_ansi_topology_and_incomplete_header(self):
        by_index = {str(i): item["uuid"] for i, item in enumerate(environment()["devices"])}
        text = (
            "\t\x1b[4mGPU0\tGPU1\tGPU2\tGPU3\x1b[0m\tCPU Affinity\n"
            "GPU0 X NV1 NV2 PHB 0-31\nGPU1 NV1 X NV1 NV2 0-31\n"
            "GPU2 NV2 NV1 X PHB 0-31\nGPU3 PHB NV2 PHB X 0-31\n"
        )
        graph = parse_topology(text, by_index)
        self.assertEqual(graph[by_index["0"]][by_index["2"]], "NV2")
        value = environment()
        value["topology"] = graph
        make_manifest(value)
        with self.assertRaises(ValueError):
            parse_topology("GPU0 GPU1\nGPU0 X NV1", by_index)

    def test_fixed_training_matrix_and_only_two_profiles(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(root / "placement.json", make_manifest(environment()))
            args = SimpleNamespace(
                placement_manifest=str(root / "placement.json"),
                run_root=str(root),
                run_id="fixture",
                python="python",
                config="experiments/fsdp_interview/configs/primary.yaml",
                data=None,
                nsys="nsys",
                source_manifest=None,
                timeout=100,
            )
            plan = make_plan(args, "benchmark")
            self.assertEqual(len(plan["stages"]), 12)
            self.assertEqual(sum(stage["steps"] for stage in plan["stages"]), 360)
            self.assertEqual({stage["warmup"] for stage in plan["stages"]}, {10})
            self.assertEqual(
                [stage["case"] for stage in make_plan(args, "profile")["stages"]], ["S0", "S1"]
            )
            self.assertFalse(any("--profile" in stage["argv"] for stage in plan["stages"]))

    def test_save_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "immutable.json"
            save(path, {"status": "FAIL"})
            with self.assertRaises(FileExistsError):
                save(path, {"status": "PASS"})
            self.assertEqual(json.loads(path.read_text())["status"], "FAIL")

    def test_source_guard_rejects_execution_change_and_records_doc_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "worker.py").write_text("pass\n")
            (root / "notes.md").write_text("old\n")
            save(
                root / "manifest.json",
                {
                    "files": {
                        name: {"sha256": sha(root / name)} for name in ["worker.py", "notes.md"]
                    }
                },
            )
            (root / "notes.md").write_text("new final report\n")
            self.assertEqual(
                verify_source(root, root / "manifest.json")["execution_files_checked"], 1
            )
            (root / "worker.py").write_text("changed\n")
            with self.assertRaises(ValueError):
                verify_source(root, root / "manifest.json")

    def test_ledger_uses_actual_status_exit_and_ownership_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            save(
                root / "stages/failed.json",
                {
                    "name": "failed",
                    "status": "FAIL",
                    "execution": {"exit_code": 9, "timed_out": False},
                },
            )
            rows = write_ledger(root)
            self.assertEqual(rows[0]["exit_code"], 9)
            self.assertIn("process_ownership.json", rows[0]["ownership_path"])

    def test_correctness_preserves_failure_and_only_continues_clean_independent(self):
        for case, residual, expected_calls in [
            ("independent_rollout", [], 2),
            ("fp32", [], 1),
            ("independent_rollout", [99], 1),
        ]:
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                args = SimpleNamespace(run_root=str(root))
                stages = [
                    {"name": "first", "case": case},
                    {"name": "second", "case": "independent_rollout"},
                ]
                calls = []

                def run(_args, stage, _manifest, calls=calls, root=root, residual=residual):
                    calls.append(stage["name"])
                    if stage["name"] == "first":
                        save(
                            root / "stages/first.json",
                            {
                                "execution": {
                                    "status": "failed",
                                    "timed_out": False,
                                    "remaining_active_owned_pids": residual,
                                    "ownership_errors": [],
                                    "driver_exception": None,
                                    "nvml": {"errors": []},
                                }
                            },
                        )
                        raise RuntimeError("preserved fixture failure")

                with (
                    mock.patch("experiments.finalization_4gpu.cli.run_stage", run),
                    mock.patch(
                        "experiments.finalization_4gpu.correctness.summarize",
                        return_value={"status": "FAIL"},
                    ) as summarize,
                ):
                    if expected_calls == 2:
                        self.assertEqual(
                            execute_correctness(args, {"stages": stages}, {})["status"], "FAIL"
                        )
                    else:
                        with self.assertRaises(RuntimeError):
                            execute_correctness(args, {"stages": stages}, {})
                    self.assertEqual(len(calls), expected_calls)
                    summarize.assert_called_once()

    def test_actual_inventory_not_historical_constants(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "communication.jsonl"
            rows = [
                {
                    "phase": "FWD",
                    "collective_type": kind,
                    "group_role": role,
                    "unit_id": unit,
                    "input_bytes": size,
                    "output_bytes": out,
                    "dtype": "torch.float32",
                }
                for kind, role, unit, size, out in [
                    ("fsdp_ag", "parameter_materialization", 2, 64, 128),
                    ("fsdp_ag", "parameter_materialization", 0, 640, 1280),
                    ("fsdp_rs", "main_gradient_bucket", 2, 256, 128),
                    ("sp_ag", "qkv", 2, 32, 64),
                    ("sp_rs", "qkv_dx", 2, 64, 32),
                    ("tp_norm_allreduce", "tp_replicated_gradient", 0, 640, 640),
                ]
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            points = inventory_from_events(path)
            self.assertEqual(points["points"][0]["full_bytes"], 128)
            self.assertEqual(points["points"][-1]["input_bytes_per_rank"], 640)
            build = {
                "status": "PASS",
                "commit": "fixture",
                "nccl": {"library_sha256": "fixture"},
                "binaries": {
                    kind + "_perf": {"path": kind, "sha256": "fixture"}
                    for kind in ["all_gather", "reduce_scatter", "all_reduce"]
                },
            }
            plan = make_reference_plan(make_manifest(environment()), points, build, repeats=3)
            self.assertEqual(len(plan), 63)
            self.assertEqual({len(stage["ordered_gpu_uuids"]) for stage in plan}, {2})


if __name__ == "__main__":
    unittest.main()
