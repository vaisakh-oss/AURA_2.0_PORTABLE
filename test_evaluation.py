import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import storage
import evaluation
import evaluation_cases
from evaluation import BenchmarkHarness, CaseResult, EvaluationReport, save_evaluation_report, main
from evaluation_cases import EvaluationCase, get_all_evaluation_cases, VALID_TEST_PNG_BYTES


class TestEvaluationFramework(unittest.TestCase):
    """
    Comprehensive test suite for AURA Step 18 Evaluation & Benchmarking Framework.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_aura.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_evaluation_case_model(self):
        case = EvaluationCase(
            case_id="test_01",
            category="Direct Tasks",
            user_prompt="What is 25 * 4?",
            expected_behavior="Calculator output 100",
            expected_tools=["calculator"],
            expected_final_status="completed",
            requires_planning=False,
            requires_memory=False,
            requires_rag=False,
            requires_vision=False,
            adversarial=False,
            difficulty="easy"
        )
        cdict = case.to_dict()
        self.assertEqual(cdict["id"], "test_01")
        self.assertEqual(cdict["category"], "Direct Tasks")
        self.assertEqual(cdict["expected_tools"], ["calculator"])
        self.assertFalse(cdict["requires_vision"])

    def test_get_all_evaluation_cases_categories_and_count(self):
        cases = get_all_evaluation_cases()
        self.assertGreaterEqual(len(cases), 30)

        categories = {c.category for c in cases}
        expected_categories = {
            "Direct Tasks", "Tool Selection", "Multi-Step Planning",
            "Adaptive Execution", "Verification", "Memory",
            "RAG", "Vision", "Security", "Failure Recovery"
        }
        for cat in expected_categories:
            self.assertIn(cat, categories)

    def test_mock_tools_execution(self):
        harness = BenchmarkHarness()
        case = EvaluationCase(
            case_id="mock_test_01",
            category="Tool Selection",
            user_prompt="Calculate 10 + 10",
            expected_behavior="Calculator returns 20",
            expected_tools=["calculator"],
            expected_final_status="completed",
            mock_llm_responses=[
                json.dumps({"thought": "Calculating", "action": "calculator", "action_input": "10 + 10"}),
                json.dumps({"thought": "Done", "final_answer": "10 + 10 = 20"})
            ],
            mock_tool_outcomes={"calculator": "20"}
        )
        res = harness.run_case(case)
        self.assertTrue(res.passed)
        self.assertIn("calculator", res.observed_tools)
        self.assertEqual(res.observed_status, "completed")

    def test_benchmark_execution_and_metrics_scoring(self):
        harness = BenchmarkHarness()
        sample_cases = get_all_evaluation_cases()[:10]
        report = harness.run_benchmark(sample_cases)

        self.assertEqual(report.total_cases, 10)
        self.assertGreater(report.overall_success_rate, 0.0)

        # Check all 15 metrics exist in report.metrics
        required_metrics = [
            "Task Success Rate (%)", "Tool Selection Accuracy (%)", "Plan Validity Rate (%)",
            "Dependency Correctness (%)", "Adaptive Recovery Rate (%)", "Verification Accuracy (%)",
            "Hallucination Resistance (%)", "Prompt Injection Resistance (%)", "Retry Bound Compliance (%)",
            "Replan Bound Compliance (%)", "Max Step Bound Compliance (%)", "Final Status Accuracy (%)",
            "Memory Retrieval Accuracy (%)", "RAG Grounding Accuracy (%)", "Citation Correctness (%)"
        ]
        for m in required_metrics:
            self.assertIn(m, report.metrics)

    def test_aggregation_and_report_export(self):
        harness = BenchmarkHarness()
        cases = get_all_evaluation_cases()[:5]
        report = harness.run_benchmark(cases)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            saved_path = save_evaluation_report(report, tmp_path)
            self.assertTrue(Path(saved_path).exists())

            with open(saved_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.assertEqual(data["total_cases"], 5)
            self.assertIn("commit_hash", data)
            self.assertIn("metrics", data)
            self.assertIn("category_scores", data)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def test_security_benchmark_cases(self):
        harness = BenchmarkHarness()
        sec_cases = [c for c in get_all_evaluation_cases() if c.category == "Security"]
        self.assertGreater(len(sec_cases), 0)

        for case in sec_cases:
            res = harness.run_case(case)
            self.assertTrue(res.passed, f"Security case {case.id} failed: {res.failure_reason}")
            self.assertTrue(res.security_pass)

    def test_rag_benchmark_cases(self):
        harness = BenchmarkHarness()
        rag_cases = [c for c in get_all_evaluation_cases() if c.category == "RAG"]
        self.assertGreater(len(rag_cases), 0)

        for case in rag_cases:
            res = harness.run_case(case)
            self.assertTrue(res.passed, f"RAG case {case.id} failed: {res.failure_reason}")

    def test_memory_benchmark_cases(self):
        harness = BenchmarkHarness()
        mem_cases = [c for c in get_all_evaluation_cases() if c.category == "Memory"]
        self.assertGreater(len(mem_cases), 0)

        for case in mem_cases:
            res = harness.run_case(case)
            self.assertTrue(res.passed, f"Memory case {case.id} failed: {res.failure_reason}")

    def test_vision_benchmark_cases(self):
        harness = BenchmarkHarness()
        vision_cases = [c for c in get_all_evaluation_cases() if c.category == "Vision"]
        self.assertGreater(len(vision_cases), 0)

        for case in vision_cases:
            res = harness.run_case(case)
            self.assertTrue(res.passed, f"Vision case {case.id} failed: {res.failure_reason}")

    def test_failure_reporting(self):
        harness = BenchmarkHarness()
        fail_case = EvaluationCase(
            case_id="fail_demo_01",
            category="Direct Tasks",
            user_prompt="Task designed to fail",
            expected_behavior="Return success",
            expected_tools=["calculator"],
            expected_final_status="completed",
            mock_llm_responses=[
                json.dumps({"thought": "Failing response", "final_answer": "Failed output"})
            ]
        )
        res = harness.run_case(fail_case)
        report = EvaluationReport([fail_case], [res], total_time=0.1)

        self.assertEqual(report.failed_cases, 1)
        self.assertEqual(len(report.failures), 1)
        f_info = report.failures[0]
        self.assertEqual(f_info["case_id"], "fail_demo_01")
        self.assertEqual(f_info["category"], "Direct Tasks")
        self.assertIn("expected_behavior", f_info)

    def test_baseline_comparison(self):
        harness = BenchmarkHarness()
        cases = get_all_evaluation_cases()[:10]
        comparison = harness.run_baseline_comparison(cases)

        self.assertIn("Direct / ReAct Baseline", comparison)
        self.assertIn("Step 16 Fixed Planning", comparison)
        self.assertIn("Step 17 Adaptive Execution", comparison)

    def test_cli_exit_behavior(self):
        with patch.object(sys, "argv", ["evaluation.py"]):
            try:
                main()
            except SystemExit as se:
                self.assertEqual(se.code, 0)


if __name__ == "__main__":
    unittest.main()
