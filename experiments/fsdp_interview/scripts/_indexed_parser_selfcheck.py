"""Exact range-index equivalence and frozen parser fixtures, synthetic only."""

from __future__ import annotations

import unittest
from collections import defaultdict

from experiments.fsdp_interview.scripts import parse_trace
from experiments.fsdp_interview.scripts._trace_parser_selfcheck import ParserChecks
from experiments.fsdp_interview.scripts.parse_trace_indexed import IndexedScopes


class IndexedParserChecks(unittest.TestCase):
    def test_query_matches_frozen_linear_query_including_order(self):
        scopes = defaultdict(list)
        for scope_id in range(70):
            scopes[scope_id % 3].append(
                {
                    "scope_id": scope_id,
                    "start": scope_id % 17,
                    "end": 20 + scope_id % 15,
                    "label": "interview/collective/x" if scope_id % 2 else "other",
                }
            )
        index = IndexedScopes()
        for tid in range(4):
            for start in range(36):
                for end in range(start, 36):
                    for communication_only in (False, True):
                        api = {"globalTid": tid, "start": start, "end": end}
                        expected = parse_trace._enclosing_scopes(
                            scopes, api, communication_only=communication_only
                        )
                        self.assertEqual(
                            index(scopes, api, communication_only=communication_only), expected
                        )

    def test_original_parser_synthetic_fixtures_under_index(self):
        original = parse_trace._enclosing_scopes
        try:
            parse_trace._enclosing_scopes = IndexedScopes()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(ParserChecks)
            result = unittest.TestResult()
            suite.run(result)
        finally:
            parse_trace._enclosing_scopes = original
        self.assertEqual(result.errors, [])
        self.assertEqual(result.failures, [])
        self.assertEqual(result.testsRun, 4)


if __name__ == "__main__":
    unittest.main()
