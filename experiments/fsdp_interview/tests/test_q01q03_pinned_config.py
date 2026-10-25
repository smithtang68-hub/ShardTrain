"""CPU-only checks for bounded benchmark configuration and input provenance."""

from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

SPEC = importlib.util.spec_from_file_location(
    "pinned_case", Path(__file__).parents[1] / "cases" / "q01_q03_pinned_memory.py"
)
assert SPEC and SPEC.loader
CASE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CASE)


class PinnedConfigurationTests(unittest.TestCase):
    def arguments(self, *extra: str):
        return CASE.parser().parse_args(["--output-dir", "unused", *extra])

    def test_default_sweep_is_bounded(self):
        arguments = self.arguments()
        CASE.validate_args(arguments)
        self.assertEqual(arguments.sizes_mib, [4, 16, 64, 128])
        self.assertEqual(arguments.pool_slots, 2)

    def test_rejects_unbounded_pool_and_buffers(self):
        for extra in (("--pool-slots", "5"), ("--sizes-mib", "1024")):
            with self.assertRaises(ValueError):
                CASE.validate_args(self.arguments(*extra))

    def test_long_profile_is_rejected(self):
        with self.assertRaises(ValueError):
            CASE.validate_args(self.arguments("--profile"))
        CASE.validate_args(
            self.arguments(
                "--profile",
                "--iterations",
                "3",
                "--warmup",
                "2",
                "--repetitions",
                "1",
                "--sizes-mib",
                "128",
            )
        )

    def test_real_token_payload_preserves_exact_byte_pattern(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npy"
            original = np.arange(16, dtype=np.int64).reshape(2, 8)
            np.save(path, original)
            result, provenance = CASE.payload(512, 42, path)
            expected = torch.from_numpy(np.resize(original.reshape(-1).view(np.uint8), 512))
            self.assertTrue(torch.equal(result, expected))
            self.assertEqual(provenance["kind"], "real_token_bytes_repeated_for_transfer_sweep")

    def test_synthetic_payload_is_repeatable(self):
        first, _ = CASE.payload(1024, 123, None)
        second, _ = CASE.payload(1024, 123, None)
        self.assertTrue(torch.equal(first, second))

    def test_allocator_env_overrides_source_defaults(self):
        with patch.dict(
            "os.environ",
            {
                "PYTORCH_ALLOC_CONF": (
                    "pinned_use_cuda_host_register:True,pinned_num_register_threads:4"
                )
            },
            clear=True,
        ):
            config = CASE.allocator_configuration()["configured_or_source_default"]
            self.assertTrue(config["pinned_use_cuda_host_register"])
            self.assertEqual(config["pinned_num_register_threads"], "4")
            self.assertFalse(config["pinned_use_background_threads"])


if __name__ == "__main__":
    unittest.main()
