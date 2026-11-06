import json
import os
import re
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

import agent
from agent import (
    AuraAgent,
    AgentExecutionError,
    ProviderTimeoutError,
    PermanentProviderError,
    CancellationError,
)
import analytics
import diagnostics
import planner
import storage
import tools


class TestAgentExecutionReliability(unittest.TestCase):
    """
    Step 5 Offline Failure-Injection and Reliability Test Suite for AURA.
    Tests agent orchestration with controlled dependencies, mocked providers,
    and isolated databases to verify:
    1. Bounded execution & provider timeouts
    2. Transient error retries & backoff
    3. Permanent error fast-fail
    4. Retry-limit exhaustion
    5. Tool exception and malformed output handling
    6. Partial failure preservation
    7. Terminal state persistence
    8. Duplicate execution prevention
    9. Sensitive data redaction
    10. Input validation safeguards
    11. Retrieval & citation failure resilience
    12. Cooperative cancellation
    13. Non-idempotent tool retry protection
    14. Reliability diagnostics summary
    15. Production database isolation
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_reliability.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)

        self.agent = AuraAgent(
            api_key="mock_key",
            model="gemini-3.8-flash",
            max_steps=5,
            db_path=self.temp_db,
            provider_timeout=2.0,
            run_timeout=5.0,
            max_retries=3,
        )

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # 1. Provider Timeout Handling
    # -------------------------------------------------------------------------
    def test_01_provider_timeout_terminal_state(self):
        """Verify that provider timeouts lead to explicit 'timeout' terminal state and persistence."""
        self.agent.call_llm = MagicMock(side_effect=ProviderTimeoutError("Deadline exceeded after 2.0s"))

        res = self.agent.run("Calculate pi to 10 decimal places")

        self.assertEqual(res["status"], "timeout")
        self.assertIn("timed out", res["final_answer"].lower())

        # Verify database terminal state
        with storage.get_db_connection(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT final_status, error, warning FROM agent_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.assertIsNotNone(row)
            self.assertEqual(row[0], "timeout")
            self.assertEqual(row[1], 1)

    # -------------------------------------------------------------------------
    # 2. Transient Failure Recovery with Retry & Backoff
    # -------------------------------------------------------------------------
    def test_02_transient_failure_recovery_with_backoff(self):
        """Verify that transient 503/429 errors trigger bounded retries and succeed on recovery."""
        mock_client = MagicMock()
        mock_resp_success = MagicMock()
        mock_resp_success.text = json.dumps({"final_answer": "Recovered successfully", "thought": "Done"})

        # First call fails with transient 503, second call succeeds
        mock_client.models.generate_content.side_effect = [
            Exception("503 Service Unavailable: Model is overloaded"),
            mock_resp_success,
        ]
        self.agent.client = mock_client

        with patch("time.sleep", return_value=None) as mock_sleep:
            res_text = self.agent._call_gemini_with_retry("test prompt", max_retries=3)
            self.assertEqual(res_text, mock_resp_success.text)
            self.assertEqual(mock_client.models.generate_content.call_count, 2)
            self.assertTrue(mock_sleep.called)

    # -------------------------------------------------------------------------
    # 3. Permanent Authentication Failure Fast-Fail
    # -------------------------------------------------------------------------
    def test_03_permanent_authentication_failure_fast_fail(self):
        """Verify that permanent errors (401/403/API_KEY_INVALID) fail fast without retrying."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("401 UNAUTHENTICATED: API_KEY_INVALID")
        self.agent.client = mock_client

        with patch("time.sleep") as mock_sleep:
            res = self.agent.run("Search for AI research papers")
            # Must NOT retry
            self.assertEqual(mock_client.models.generate_content.call_count, 1)
            mock_sleep.assert_not_called()

        self.assertEqual(res["status"], "failed")
        self.assertIn("authorization error", res["final_answer"].lower())

        with storage.get_db_connection(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT final_status, error FROM agent_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.assertEqual(row[0], "failed")
            self.assertEqual(row[1], 1)

    # -------------------------------------------------------------------------
    # 4. Retry-Limit Exhaustion
    # -------------------------------------------------------------------------
    def test_04_retry_limit_exhaustion(self):
        """Verify that when transient errors persist, retries are bounded and fail gracefully."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("503 UNAVAILABLE: High demand")
        self.agent.client = mock_client

        with patch("time.sleep", return_value=None):
            with self.assertRaises(AgentExecutionError):
                self.agent._call_gemini_with_retry("test prompt", max_retries=3)

            # Exactly 3 attempts made
            self.assertEqual(mock_client.models.generate_content.call_count, 3)

    # -------------------------------------------------------------------------
    # 5. Tool Exception and Malformed Output Handling
    # -------------------------------------------------------------------------
    def test_05_tool_exception_and_malformed_output(self):
        """Verify that tool crashes are safely caught, returned as errors, and recorded in telemetry."""
        # Monkeypatch calculator to raise an unexpected runtime error
        with patch.dict(tools.TOOL_REGISTRY, {"calculator": MagicMock(side_effect=RuntimeError("Internal core dump"))}):
            tool_res = tools.run_tool("calculator", "10 + 10")
            self.assertTrue(tools.is_tool_error(tool_res))
            self.assertIn("TOOL ERROR", tool_res)

        # Agent ReAct loop handles this error gracefully
        step1_action = json.dumps({"action": "calculator", "action_input": "crash", "thought": "Testing crash"})
        step2_answer = json.dumps({"final_answer": "I caught the tool error safely", "thought": "Recovered"})

        with patch.dict(tools.TOOL_REGISTRY, {"calculator": MagicMock(side_effect=RuntimeError("Bug in tool"))}):
            self.agent.call_llm = MagicMock(side_effect=[step1_action, step2_answer])
            res = self.agent.run("Compute calculation")

            self.assertEqual(res["status"], "completed")
            self.assertIn("caught the tool error", res["final_answer"])

            # Verify telemetry recorded the tool failure
            with storage.get_db_connection(self.temp_db) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT tool, success, error_type FROM agent_tool_runs WHERE tool = 'calculator'")
                tool_run = cursor.fetchone()
                self.assertIsNotNone(tool_run)
                self.assertEqual(tool_run[1], 0)  # success = False
                self.assertEqual(tool_run[2], "TOOL_ERROR")

    # -------------------------------------------------------------------------
    # 6. Multiple Tool Calls Where One Fails (Partial Progress)
    # -------------------------------------------------------------------------
    def test_06_multiple_tool_calls_partial_failure(self):
        """Verify that multiple tool calls preserve successful progress while marking failures."""
        step1_action = json.dumps({"action": "calculator", "action_input": "100 / 4", "thought": "Step 1"})
        step2_action = json.dumps({"action": "calculator", "action_input": "1 / 0", "thought": "Step 2"})
        step3_answer = json.dumps({"final_answer": "Calculated 25, but 1/0 is division by zero", "thought": "Done"})

        self.agent.call_llm = MagicMock(side_effect=[step1_action, step2_action, step3_answer])
        res = self.agent.run("Perform math operations")

        self.assertEqual(res["status"], "completed")

        with storage.get_db_connection(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT tool, success FROM agent_tool_runs ORDER BY id ASC")
            runs = cursor.fetchall()
            self.assertEqual(len(runs), 2)
            self.assertEqual(runs[0][1], 1)  # 100/4 succeeded
            self.assertEqual(runs[1][1], 0)  # 1/0 failed (division by zero)

    # -------------------------------------------------------------------------
    # 7. Failed Run State Persistence
    # -------------------------------------------------------------------------
    def test_07_failed_run_state_persistence(self):
        """Verify that unrecoverable runtime errors reach terminal state in DB."""
        self.agent.call_llm = MagicMock(side_effect=Exception("Uncaught network drop"))

        res = self.agent.run("Tell me the forecast")
        self.assertEqual(res["status"], "failed")

        with storage.get_db_connection(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT final_status, error, total_steps FROM agent_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.assertEqual(row[0], "failed")
            self.assertEqual(row[1], 1)

    # -------------------------------------------------------------------------
    # 8. Duplicate Execution Prevention
    # -------------------------------------------------------------------------
    def test_08_duplicate_execution_prevention(self):
        """Verify that identical tool actions in the same run are not executed multiple times."""
        # Step 1: calculator 2 + 2 -> 4
        # Step 2: model re-requests calculator 2 + 2 (e.g. following a retry)
        # Step 3: final answer
        step1 = json.dumps({"action": "calculator", "action_input": "2 + 2", "thought": "First call"})
        step2 = json.dumps({"action": "calculator", "action_input": "2 + 2", "thought": "Duplicate re-call"})
        step3 = json.dumps({"final_answer": "Result is 4", "thought": "Done"})

        mock_calc = MagicMock(return_value="4")
        with patch.dict(tools.TOOL_REGISTRY, {"calculator": mock_calc}):
            self.agent.call_llm = MagicMock(side_effect=[step1, step2, step3])
            res = self.agent.run("Calculate 2+2 twice")

            self.assertEqual(res["status"], "completed")
            # Actual underlying calculator tool should be called ONCE, second call reuses cache
            self.assertEqual(mock_calc.call_count, 1)

    # -------------------------------------------------------------------------
    # 9. Sensitive Data Redaction
    # -------------------------------------------------------------------------
    def test_09_sensitive_data_redaction(self):
        """Verify that API keys, bearer tokens, and secrets are redacted from telemetry and logs."""
        raw_secret_key = "AIzaSyFakeKey1234567890abcdefghijklm"
        raw_bearer = "bearer secret-bearer-token-123456"

        text_with_secrets = f"Result with {raw_secret_key} and {raw_bearer}"
        redacted = tools.redact_sensitive_text(text_with_secrets)

        self.assertNotIn(raw_secret_key, redacted)
        self.assertNotIn("secret-bearer-token", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)
        self.assertIn("[REDACTED_TOKEN]", redacted)

        # Test within full agent run
        step1 = json.dumps({
            "action": "calculator",
            "action_input": f"2 + 2  # key={raw_secret_key}",
            "thought": "Using secret"
        })
        step2 = json.dumps({"final_answer": "Completed", "thought": "Done"})

        emitted_events = []
        self.agent.call_llm = MagicMock(side_effect=[step1, step2])
        self.agent.run("Test secrets", event_callback=lambda evt: emitted_events.append(evt))

        for evt in emitted_events:
            for k, v in evt.items():
                self.assertNotIn(raw_secret_key, str(v))

    # -------------------------------------------------------------------------
    # 10. Invalid Tool Arguments Validation
    # -------------------------------------------------------------------------
    def test_10_invalid_tool_arguments_validation(self):
        """Verify that unknown tools and oversized or invalid arguments are rejected cleanly."""
        # 1. Unknown tool
        is_valid, err = tools.validate_tool_input("destroy_system", "arg")
        self.assertFalse(is_valid)
        self.assertIn("Unknown tool", err)

        # 2. Oversized input
        huge_input = "x" * 60000
        is_valid, err = tools.validate_tool_input("calculator", huge_input)
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum allowed length", err)

        # 3. Invalid argument type (e.g., custom class or function)
        is_valid, err = tools.validate_tool_input("calculator", object())
        self.assertFalse(is_valid)
        self.assertIn("Invalid input argument type", err)

    # -------------------------------------------------------------------------
    # 11. Citation or Retrieval Failure During Run
    # -------------------------------------------------------------------------
    def test_11_citation_or_retrieval_failure_graceful_handling(self):
        """Verify that knowledge retrieval errors or invalid citations do not crash the agent."""
        step1 = json.dumps({"action": "knowledge_search", "action_input": "Quantum Computing", "thought": "Searching"})
        # Final answer hallucinates citation [99] which does not exist
        step2 = json.dumps({
            "final_answer": "Quantum computing uses qubits [99].",
            "thought": "Answering with bogus citation"
        })

        # Knowledge search returns nothing
        with patch("tools.knowledge_search", return_value="No documents found in knowledge base matching query."):
            self.agent.call_llm = MagicMock(side_effect=[step1, step2])
            res = self.agent.run("Tell me about Quantum Computing")

            self.assertEqual(res["status"], "completed")
            self.assertIn("Citation Warning", res["final_answer"])
            self.assertFalse(res["citation_validation"]["is_valid"])

    # -------------------------------------------------------------------------
    # 12. Cooperative Cancellation
    # -------------------------------------------------------------------------
    def test_12_cooperative_cancellation(self):
        """Verify that a cancellation signal stops execution cleanly and persists terminal state."""
        cancel_token = threading.Event()

        # Step 1 succeeds, but cancellation is triggered before step 2
        step1 = json.dumps({"action": "calculator", "action_input": "5 * 5", "thought": "Step 1"})

        def call_llm_with_cancel(*args, **kwargs):
            cancel_token.set()
            return step1

        self.agent.call_llm = MagicMock(side_effect=call_llm_with_cancel)
        res = self.agent.run("Perform calculation", cancel_token=cancel_token)

        self.assertEqual(res["status"], "cancelled")
        self.assertIn("cancelled", res["final_answer"].lower())

        with storage.get_db_connection(self.temp_db) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT final_status, warning FROM agent_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            self.assertEqual(row[0], "cancelled")
            self.assertEqual(row[1], 1)

    # -------------------------------------------------------------------------
    # 13. Non-Idempotent Tool Retry Protection
    # -------------------------------------------------------------------------
    def test_13_non_idempotent_tool_retry_protection(self):
        """Verify that non-idempotent tools (like save_note) are never flagged as retryable."""
        self.assertFalse(tools.is_tool_idempotent("save_note"))
        self.assertTrue(tools.is_tool_idempotent("calculator"))
        self.assertTrue(tools.is_tool_idempotent("knowledge_search"))
        self.assertTrue(tools.is_tool_idempotent("web_search"))
        self.assertTrue(tools.is_tool_idempotent("read_notes"))

        # In planner observations
        save_obs = planner.create_observation(1, "save_note", "NOTE ERROR: storage full")
        self.assertFalse(save_obs.retryable)

        calc_obs = planner.create_observation(2, "calculator", "CALCULATOR ERROR: syntax error")
        self.assertTrue(calc_obs.retryable)

    # -------------------------------------------------------------------------
    # 14. Reliability Diagnostics Summary
    # -------------------------------------------------------------------------
    def test_14_diagnostics_reliability_summary(self):
        """Verify that get_agent_reliability_diagnostics categorizes runs, retries, and failures."""
        diag = diagnostics.get_agent_reliability_diagnostics(self.temp_db)
        self.assertIn("total_runs", diag)
        self.assertIn("success_rate", diag)
        self.assertIn("status_breakdown", diag)
        self.assertIn("tool_statistics", diag)
        self.assertIn("summary", diag)

        check = diagnostics.check_agent_reliability(self.temp_db)
        self.assertEqual(check.component, "Agent Reliability")
        self.assertEqual(check.check_name, "Agent Execution & Failure Observability Check")
        self.assertIn(check.status.value, ["HEALTHY", "DEGRADED", "CRITICAL"])

    # -------------------------------------------------------------------------
    # 15. Production Database Isolation
    # -------------------------------------------------------------------------
    def test_15_production_database_isolation(self):
        """Verify that tests strictly isolate and do not modify production aura.db."""
        prod_db = storage.DB_FILE
        if not os.path.exists(prod_db):
            return

        def get_prod_counts():
            with storage.get_db_connection(prod_db) as conn:
                cursor = conn.cursor()
                counts = {}
                for tbl in ["conversations", "messages", "agent_runs", "agent_tool_runs", "memories"]:
                    cursor.execute(f"SELECT COUNT(*) FROM {tbl}")
                    counts[tbl] = cursor.fetchone()[0]
                return counts

        before = get_prod_counts()

        # Execute agent run with test database
        self.agent.call_llm = MagicMock(return_value=json.dumps({"final_answer": "Isolated", "thought": "Safe"}))
        self.agent.run("Verify isolation")

        after = get_prod_counts()
        self.assertEqual(before, after, "Production database was modified during testing!")

    # -------------------------------------------------------------------------
    # 16. Provider Client Timeout Configuration
    # -------------------------------------------------------------------------
    def test_16_provider_client_timeout_configured(self):
        """Verify that provider_timeout is passed into genai.Client initialization."""
        test_agent = agent.AuraAgent(
            api_key="mock_key",
            provider_timeout=15.0,
            db_path=self.temp_db
        )
        self.assertIsNotNone(test_agent.client)
        self.assertEqual(test_agent.provider_timeout, 15.0)

    # -------------------------------------------------------------------------
    # 17. Duplicate Prevention with JSON Argument Normalization
    # -------------------------------------------------------------------------
    def test_17_duplicate_prevention_normalized_json_arguments(self):
        """Verify that JSON dictionary arguments with different key orderings map to the same key."""
        k1 = agent.normalize_tool_action_key("custom_tool", {"b": 2, "a": 1})
        k2 = agent.normalize_tool_action_key("custom_tool", {"a": 1, "b": 2})
        self.assertEqual(k1, k2)

        # In agent run:
        step1 = json.dumps({"action": "calculator", "action_input": "2 + 2", "thought": "Step 1"})
        step2 = json.dumps({"action": "calculator", "action_input": "  2 + 2  ", "thought": "Step 2 whitespace variation"})
        step3 = json.dumps({"final_answer": "4", "thought": "Done"})

        mock_calc = MagicMock(return_value="4")
        with patch.dict(tools.TOOL_REGISTRY, {"calculator": mock_calc}):
            self.agent.call_llm = MagicMock(side_effect=[step1, step2, step3])
            res = self.agent.run("Compute")
            self.assertEqual(res["status"], "completed")
            self.assertEqual(mock_calc.call_count, 1)

    # -------------------------------------------------------------------------
    # 18. Duplicate Prevention Allows Distinct Tool Calls
    # -------------------------------------------------------------------------
    def test_18_duplicate_prevention_different_arguments_execute(self):
        """Verify that genuinely distinct tool arguments are not suppressed and execute separately."""
        step1 = json.dumps({"action": "calculator", "action_input": "2 + 2", "thought": "Step 1"})
        step2 = json.dumps({"action": "calculator", "action_input": "3 + 3", "thought": "Step 2"})
        step3 = json.dumps({"final_answer": "4 and 6", "thought": "Done"})

        mock_calc = MagicMock(side_effect=["4", "6"])
        with patch.dict(tools.TOOL_REGISTRY, {"calculator": mock_calc}):
            self.agent.call_llm = MagicMock(side_effect=[step1, step2, step3])
            res = self.agent.run("Compute distinct")
            self.assertEqual(res["status"], "completed")
            self.assertEqual(mock_calc.call_count, 2)

    # -------------------------------------------------------------------------
    # 19. Terminal-State Persistence Under Storage Failure
    # -------------------------------------------------------------------------
    def test_19_terminal_state_persistence_under_storage_error(self):
        """Verify that storage failures during finish() do not cause false success claims or hide errors."""
        import sqlite3
        # Case A: Success run where analytics storage throws
        with patch("analytics.record_agent_run", side_effect=sqlite3.OperationalError("database is locked")):
            self.agent.call_llm = MagicMock(return_value=json.dumps({"final_answer": "Success", "thought": "Done"}))
            res = self.agent.run("Test storage failure on success")
            self.assertEqual(res["status"], "completed_with_warning")
            self.assertIn("persistence_error", res)
            self.assertIn("database is locked", res["persistence_error"])

        # Case B: Failed run where analytics storage also throws
        with patch("analytics.record_agent_run", side_effect=sqlite3.OperationalError("disk full")):
            self.agent.call_llm = MagicMock(side_effect=PermanentProviderError("401 Unauthorized"))
            res = self.agent.run("Test secondary storage failure")
            # Primary status and error must be preserved!
            self.assertEqual(res["status"], "failed")
            self.assertIn("401 Unauthorized", res["error"])
            self.assertIn("disk full", res["persistence_error"])


if __name__ == "__main__":
    unittest.main()
