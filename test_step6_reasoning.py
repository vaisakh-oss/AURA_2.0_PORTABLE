"""
Unit tests for AURA Step 6: Advanced Reasoning, Autonomous Planning, and Intelligent Execution.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import agent
import classifier
import knowledge
import planner
import storage
import tools


class TestStep6TaskClassifier(unittest.TestCase):
    """Test suite for classifier.py intelligent request classification."""

    def test_simple_question_no_tools(self):
        res = classifier.classify_task("What is the capital of France?")
        self.assertEqual(res.category, classifier.TaskCategory.SIMPLE_QUESTION)
        self.assertFalse(res.needs_plan)
        self.assertEqual(res.suggested_tools, [])

    def test_conversational_greeting_no_tools(self):
        res = classifier.classify_task("Hello! How are you doing today?")
        self.assertEqual(res.category, classifier.TaskCategory.SIMPLE_QUESTION)
        self.assertFalse(res.needs_plan)
        self.assertEqual(res.suggested_tools, [])

    def test_pure_calculation_routing(self):
        res = classifier.classify_task("Calculate 125 * 8")
        self.assertEqual(res.category, classifier.TaskCategory.CALCULATION)
        self.assertFalse(res.needs_plan)
        self.assertIn("calculator", res.suggested_tools)

    def test_multi_step_planning_requirement(self):
        res = classifier.classify_task("Calculate 125 * 8, then divide the result by 4, and summarize the steps.")
        self.assertEqual(res.category, classifier.TaskCategory.MULTI_STEP)
        self.assertTrue(res.needs_plan)
        self.assertIn("calculator", res.suggested_tools)

    def test_document_analysis_classification(self):
        res = classifier.classify_task("Summarize the main findings in the uploaded document.", has_document=True)
        self.assertEqual(res.category, classifier.TaskCategory.DOCUMENT_ANALYSIS)
        self.assertFalse(res.needs_plan)

    def test_unsupported_destructive_command(self):
        res = classifier.classify_task("Run bash command rm -rf / to clean up files.")
        self.assertEqual(res.category, classifier.TaskCategory.UNSUPPORTED)
        self.assertTrue(res.is_unsupported)
        self.assertEqual(res.suggested_tools, [])

    def test_ambiguous_request_classification(self):
        res = classifier.classify_task("Do it.")
        self.assertEqual(res.category, classifier.TaskCategory.AMBIGUOUS)
        self.assertTrue(res.is_ambiguous)


class TestStep6PlanningEngine(unittest.TestCase):
    """Test suite for planner.py subtasks, dependencies, and topological ordering."""

    def test_plan_step_attributes(self):
        step = planner.PlanStep(
            step_id=1,
            description="Compute initial balance",
            tool="calculator",
            tool_input="1000 * 1.05",
            depends_on=[],
            expected_output="1050",
            verification_criteria="Numerical value equals 1050",
            subtask_id="subtask_1"
        )
        self.assertEqual(step.id, 1)
        self.assertEqual(step.subtask_id, "subtask_1")
        self.assertEqual(step.tool_input, "1000 * 1.05")
        self.assertEqual(step.expected_output, "1050")
        self.assertEqual(step.verification_criteria, "Numerical value equals 1050")

        d = step.to_dict()
        self.assertEqual(d["required_tool"], "calculator")
        self.assertEqual(d["dependencies"], [])
        self.assertEqual(d["tool_input"], "1000 * 1.05")

    def test_topological_execution_order(self):
        step1 = planner.PlanStep(step_id=1, description="Step 1", tool="calculator", depends_on=[])
        step2 = planner.PlanStep(step_id=2, description="Step 2", tool="calculator", depends_on=[1])
        step3 = planner.PlanStep(step_id=3, description="Step 3", tool="calculator", depends_on=[2])

        # Provide steps out of order
        plan = planner.TaskPlan(goal="Chained calculation", needs_plan=True, steps=[step3, step1, step2])
        order = plan.get_execution_order()
        self.assertEqual([s.id for s in order], [1, 2, 3])

    def test_detect_missing_information_dates(self):
        goal = "What is the scheduled launch date for the project?"
        doc_without_dates = "This document describes the software architecture using FastAPI and React."
        missing = planner.detect_missing_information(goal, doc_without_dates)
        self.assertTrue(len(missing) >= 1)
        self.assertTrue(any("launch date" in m or "timeline" in m for m in missing))

    def test_detect_missing_information_budget(self):
        goal = "What is the total project budget?"
        doc_without_budget = "The project team consists of 4 engineers and 1 product manager."
        missing = planner.detect_missing_information(goal, doc_without_budget)
        self.assertTrue(len(missing) >= 1)
        self.assertTrue(any("Budget" in m or "cost" in m for m in missing))

    def test_detect_missing_information_present_not_flagged(self):
        goal = "What is the scheduled launch date?"
        doc_with_date = "The scheduled launch date is November 15, 2026."
        missing = planner.detect_missing_information(goal, doc_with_date)
        self.assertEqual(len(missing), 0)


class TestStep6AgentExecution(unittest.TestCase):
    """Test suite for agent.py execution loop, memory consent, and error handling."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_step6.db")
        self.orig_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)
        self.agent = agent.AuraAgent(api_key="mock", db_path=self.temp_db)

    def tearDown(self):
        if self.orig_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_unsupported_command_immediate_rejection(self):
        res = self.agent.run("Run bash command: rm -rf /")
        self.assertEqual(res["status"], "failed")
        self.assertIn("unsupported or restricted", res["final_answer"].lower())
        self.assertEqual(res["steps"], 0)

    def test_ambiguous_request_clarification(self):
        res = self.agent.run("Do it.")
        self.assertEqual(res["status"], "completed_with_warning")
        self.assertIn("clarify", res["final_answer"].lower())
        self.assertEqual(res["steps"], 0)

    def test_no_unconsented_memory_saving(self):
        # A normal query should NOT extract and save memories
        mock_resp = json.dumps({"thought": "Answer", "final_answer": "Paris is the capital of France."})
        with patch.object(self.agent, "call_llm", return_value=mock_resp):
            self.agent.run("What is the capital of France?")

        # Check DB memories table
        import sqlite3
        conn = sqlite3.connect(self.temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        count = cur.fetchone()[0]
        conn.close()
        self.assertEqual(count, 0)

    def test_explicit_memory_consent_saving(self):
        mock_resp = json.dumps({"thought": "Saving", "final_answer": "I have remembered that your preferred language is Python."})
        with patch.object(self.agent, "call_llm", return_value=mock_resp):
            self.agent.run("Please remember that my preferred programming language is Python.")

        import sqlite3
        conn = sqlite3.connect(self.temp_db)
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM memories")
        count = cur.fetchone()[0]
        conn.close()
        self.assertTrue(count >= 1)

    def test_quota_429_graceful_recovery(self):
        with patch.object(self.agent, "call_llm", side_effect=Exception("429 Resource has been exhausted (e.g. check quota)")):
            res = self.agent.run("What is quantum computing?")
            self.assertEqual(res["status"], "degraded")
            self.assertIn("quota", res["final_answer"].lower())

    def test_auth_401_graceful_recovery(self):
        with patch.object(self.agent, "call_llm", side_effect=agent.PermanentProviderError("401 Unauthorized / Invalid API Key")):
            res = self.agent.run("What is quantum computing?")
            self.assertEqual(res["status"], "failed")
            self.assertIn("authorization", res["final_answer"].lower())


if __name__ == "__main__":
    unittest.main()
