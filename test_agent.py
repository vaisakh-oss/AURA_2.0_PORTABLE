import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import agent
import storage
import tools


class TestAuraAgent(unittest.TestCase):
    """
    Comprehensive test suite for AuraAgent Core (agent.py).
    Tests all 15 required core behaviors using mocked Gemini API responses.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_aura.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)
        self.agent = agent.AuraAgent(api_key="mock_key", model="gemini-3.8-flash", max_steps=4, db_path=self.temp_db)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # 1. Final answer returned correctly
    def test_final_answer_direct(self):
        mock_resp = json.dumps({
            "thought": "No tool needed, answering directly",
            "final_answer": "42 is the answer."
        })
        self.agent.call_llm = MagicMock(return_value=mock_resp)

        res = self.agent.run("What is the answer to life?")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "42 is the answer.")
        self.assertEqual(res["steps"], 1)

    # 2. Single tool action followed by final answer
    def test_single_tool_then_final_answer(self):
        resp1 = json.dumps({
            "thought": "Need to calculate",
            "action": "calculator",
            "action_input": "25 * 4"
        })
        resp2 = json.dumps({
            "thought": "Got calculation result",
            "final_answer": "The result is 100."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("What is 25 * 4?")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "The result is 100.")
        self.assertEqual(res["steps"], 2)

    # 3. Multiple ReAct iterations
    def test_multiple_react_iterations(self):
        resp1 = json.dumps({
            "thought": "Step 1 calc",
            "action": "calculator",
            "action_input": "10 + 20"
        })
        resp2 = json.dumps({
            "thought": "Step 2 calc",
            "action": "calculator",
            "action_input": "30 * 2"
        })
        resp3 = json.dumps({
            "thought": "Final answer",
            "final_answer": "Result is 60."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2, resp3])

        res = self.agent.run("Complex multi-step math")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "Result is 60.")
        self.assertEqual(res["steps"], 3)

    # 4. Tool observation reaches next prompt
    def test_tool_observation_reaches_next_prompt(self):
        resp1 = json.dumps({
            "thought": "Calculating",
            "action": "calculator",
            "action_input": "2 + 2"
        })
        resp2 = json.dumps({
            "thought": "Finished",
            "final_answer": "4"
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        self.agent.run("What is 2 + 2?")
        second_call_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Tool: calculator", second_call_prompt)
        self.assertIn("Result: 4", second_call_prompt)

    # 5. Unknown tool is rejected
    def test_unknown_tool_rejected(self):
        resp1 = json.dumps({
            "thought": "Invalid tool call",
            "action": "unknown_tool_xyz",
            "action_input": "abc"
        })
        resp2 = json.dumps({
            "thought": "Correction",
            "final_answer": "Handled unknown tool error."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Run unknown tool")
        self.assertEqual(res["status"], "completed")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Unknown action 'unknown_tool_xyz'", second_prompt)

    # 6. Malformed JSON is handled
    def test_malformed_json_handled(self):
        resp1 = "Invalid JSON {{{ string"
        resp2 = json.dumps({
            "thought": "Recovered",
            "final_answer": "Recovered from bad JSON."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Test bad JSON")
        self.assertEqual(res["status"], "completed")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Malformed JSON output", second_prompt)

    # 7. Missing action_input is handled
    def test_missing_action_input_handled(self):
        resp1 = json.dumps({
            "thought": "Missing action_input",
            "action": "web_search"
        })
        resp2 = json.dumps({
            "thought": "Corrected",
            "final_answer": "Missing field handled."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Test missing input")
        self.assertEqual(res["status"], "completed")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Missing required field 'action_input'", second_prompt)

    # 8. Missing final_answer is handled
    def test_missing_final_answer_handled(self):
        resp1 = json.dumps({
            "thought": "Neither action nor final answer"
        })
        resp2 = json.dumps({
            "thought": "Fixed",
            "final_answer": "Fixed object."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Test empty schema")
        self.assertEqual(res["status"], "completed")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("must contain either 'action' or 'final_answer'", second_prompt)

    # 9. Maximum iteration limit works
    def test_max_steps_limit_reached(self):
        resp_tool = json.dumps({
            "thought": "Looping action",
            "action": "calculator",
            "action_input": "1 + 1"
        })
        agent_max2 = agent.AuraAgent(api_key="mock", model="gemini-3.8-flash", max_steps=2)
        agent_max2.call_llm = MagicMock(return_value=resp_tool)

        res = agent_max2.run("Infinite calculation request")
        self.assertEqual(res["status"], "max_steps_reached")
        self.assertEqual(res["steps"], 2)

    # 10. Gemini/API failure is handled
    def test_gemini_api_failure_handled(self):
        self.agent.call_llm = MagicMock(side_effect=RuntimeError("API Network Connection Timeout"))

        res = self.agent.run("Test API failure")
        self.assertEqual(res["status"], "failed")
        self.assertIn("API error", res["error"])

    # 11. Tool failure is handled without crashing the agent
    def test_tool_failure_handled(self):
        resp1 = json.dumps({
            "thought": "Causing division by zero",
            "action": "calculator",
            "action_input": "5 / 0"
        })
        resp2 = json.dumps({
            "thought": "Handled tool error",
            "final_answer": "Handled zero division."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Calculate 5 / 0")
        self.assertEqual(res["status"], "completed")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Division by zero", second_prompt)

    # 12. Memory context reaches the prompt
    @patch("agent.memory_as_text", return_value="- [2026-10-08] User is building AURA")
    def test_memory_context_reaches_prompt(self, mock_mem):
        resp = json.dumps({"thought": "Answering", "final_answer": "OK"})
        self.agent.call_llm = MagicMock(return_value=resp)

        self.agent.run("Test memory injection")
        prompt = self.agent.call_llm.call_args[0][0]
        self.assertIn("User is building AURA", prompt)

    # 13. Conversation context reaches the prompt
    def test_conversation_context_reaches_prompt(self):
        resp1 = json.dumps({
            "thought": "First tool call",
            "action": "calculator",
            "action_input": "10 * 10"
        })
        resp2 = json.dumps({
            "thought": "Final",
            "final_answer": "100"
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        self.agent.run("Goal statement")
        second_prompt = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("USER: Goal statement", second_prompt)

    # 14. User goal reaches the prompt
    def test_user_goal_reaches_prompt(self):
        resp = json.dumps({"thought": "Answering", "final_answer": "Done"})
        self.agent.call_llm = MagicMock(return_value=resp)

        self.agent.run("Specific unique goal string 12345")
        prompt = self.agent.call_llm.call_args[0][0]
        self.assertIn("Specific unique goal string 12345", prompt)

    # 16. Gemini synthesis fails after successful tool call -> completed_with_warning
    def test_gemini_failure_after_tool_success(self):
        resp_tool = json.dumps({
            "thought": "Need calculation",
            "action": "calculator",
            "action_input": "25 * 4"
        })
        self.agent.call_llm = MagicMock(side_effect=[resp_tool, RuntimeError("503 UNAVAILABLE Service Spike")])

        res = self.agent.run("Calculate 25 * 4")
        self.assertEqual(res["status"], "completed_with_warning")
        self.assertIn("100", res["final_answer"])
        self.assertIn("Verified tool results", res["final_answer"])

    # 17. Gemini failure before any tool execution -> status failed
    def test_gemini_failure_before_tool_execution(self):
        self.agent.call_llm = MagicMock(side_effect=RuntimeError("503 UNAVAILABLE"))

        res = self.agent.run("Calculate 25 * 4")
        self.assertEqual(res["status"], "failed")
        self.assertIn("couldn't complete", res["final_answer"])


class TestAgentObservability(unittest.TestCase):
    """
    Test suite for Live Agent Execution Timeline & Observability (Step 12).
    Verifies event emission, payload safety, event types, ordering, and tool label mapping.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_aura.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)
        self.agent = agent.AuraAgent(api_key="mock_key", model="gemini-3.8-flash", max_steps=3, db_path=self.temp_db)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    def test_agent_works_without_event_callback(self):
        """1. Agent works without event callback."""
        mock_resp = json.dumps({"thought": "No callback test", "final_answer": "Success"})
        self.agent.call_llm = MagicMock(return_value=mock_resp)
        res = self.agent.run("Test without callback")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "Success")

    def test_agent_start_and_completed_events(self):
        """2 & 6. agent_start and completed events emitted."""
        events = []
        mock_resp = json.dumps({"thought": "Direct answer", "final_answer": "Done"})
        self.agent.call_llm = MagicMock(return_value=mock_resp)

        res = self.agent.run("Test direct answer", event_callback=events.append)
        self.assertEqual(res["status"], "completed")

        types = [e["type"] for e in events]
        self.assertIn("agent_start", types)
        self.assertIn("final_start", types)
        self.assertIn("completed", types)

    def test_tool_start_and_tool_success_events(self):
        """3 & 4. tool_start and tool_success events emitted."""
        events = []
        resp1 = json.dumps({"thought": "Calc", "action": "calculator", "action_input": "25 * 4"})
        resp2 = json.dumps({"thought": "Done", "final_answer": "100"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Calculate 25 * 4", event_callback=events.append)
        self.assertEqual(res["status"], "completed")

        types = [e["type"] for e in events]
        self.assertIn("tool_start", types)
        self.assertIn("tool_success", types)

        tool_start_evt = next(e for e in events if e["type"] == "tool_start")
        self.assertEqual(tool_start_evt["tool"], "calculator")
        self.assertEqual(tool_start_evt["label"], "Calculating")

    def test_tool_error_event(self):
        """5. tool_error event emitted on tool failure."""
        events = []
        resp1 = json.dumps({"thought": "Div zero", "action": "calculator", "action_input": "10 / 0"})
        resp2 = json.dumps({"thought": "Recovered", "final_answer": "Done"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Divide by 0", event_callback=events.append)
        self.assertEqual(res["status"], "completed")

        types = [e["type"] for e in events]
        self.assertIn("tool_error", types)
        err_evt = next(e for e in events if e["type"] == "tool_error")
        self.assertEqual(err_evt["tool"], "calculator")
        self.assertTrue(err_evt["is_error"])

    def test_completed_with_warning_event(self):
        """7. completed_with_warning event emitted."""
        events = []
        resp1 = json.dumps({"thought": "Search", "action": "calculator", "action_input": "25 * 4"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, RuntimeError("503 UNAVAILABLE Spike")])

        res = self.agent.run("Calculate 25 * 4", event_callback=events.append)
        self.assertEqual(res["status"], "completed_with_warning")

        types = [e["type"] for e in events]
        self.assertIn("completed_with_warning", types)

    def test_failed_event(self):
        """8. failed event emitted on complete API breakdown or empty goal."""
        events = []
        self.agent.call_llm = MagicMock(side_effect=RuntimeError("API error"))

        res = self.agent.run("Failed run goal", event_callback=events.append)
        self.assertEqual(res["status"], "failed")

        types = [e["type"] for e in events]
        self.assertIn("failed", types)

    def test_max_steps_reached_event(self):
        """9. max_steps_reached event emitted."""
        events = []
        resp_tool = json.dumps({"thought": "Looping", "action": "calculator", "action_input": "1 + 1"})
        self.agent.call_llm = MagicMock(return_value=resp_tool)

        res = self.agent.run("Loop task", event_callback=events.append)
        self.assertEqual(res["status"], "max_steps_reached")

        types = [e["type"] for e in events]
        self.assertIn("max_steps_reached", types)

    def test_event_ordering_correct(self):
        """10. Event ordering is correct."""
        events = []
        resp1 = json.dumps({"thought": "Calc", "action": "calculator", "action_input": "5 + 5"})
        resp2 = json.dumps({"thought": "Done", "final_answer": "10"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        self.agent.run("Test ordering", event_callback=events.append)

        types = [e["type"] for e in events]
        expected = ["agent_start", "tool_start", "tool_success", "final_start", "completed"]
        self.assertEqual(types, expected)

    def test_event_payload_contains_only_safe_fields(self):
        """11 & 12. Event payload contains only safe fields; thought, prompt, keys, and action_input are never exposed."""
        events = []
        resp1 = json.dumps({"thought": "SECRET_THOUGHT_RARE_12345", "action": "save_note", "action_input": "SECRET_INPUT_NOTE_999"})
        resp2 = json.dumps({"thought": "SECRET_THOUGHT_FINAL", "final_answer": "Done"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        self.agent.run("Save note test", event_callback=events.append)

        forbidden_keys = {"thought", "action_input", "system_prompt", "prompt", "api_key", "raw_response"}
        allowed_keys = {"type", "label", "timestamp", "elapsed_time", "tool", "success", "is_error", "is_warning"}

        for evt in events:
            for f_key in forbidden_keys:
                self.assertNotIn(f_key, evt, f"Forbidden key '{f_key}' found in event payload: {evt}")
            for k in evt.keys():
                self.assertIn(k, allowed_keys, f"Unexpected unsafe key '{k}' found in event payload: {evt}")

    def test_tool_label_mappings(self):
        """Verify human-friendly tool label mapping."""
        self.assertEqual(agent.get_tool_label("web_search"), "Searching the web")
        self.assertEqual(agent.get_tool_label("calculator"), "Calculating")
        self.assertEqual(agent.get_tool_label("python_exec"), "Running analysis")
        self.assertEqual(agent.get_tool_label("save_note"), "Saving to memory")
        self.assertEqual(agent.get_tool_label("read_notes"), "Reading memory")
        self.assertEqual(agent.get_tool_label("unknown_tool"), "Running tool")

    # =================================================================
    # LOGGING SECURITY & SENSITIVE CREDENTIAL REDACTION TESTS
    # =================================================================
    def test_redact_sensitive_text_synthetic_secrets(self):
        """Verify redact_sensitive_text scrubs all synthetic credential patterns."""
        synthetic_google_key = "AIzaSyD" + "0" * 32
        synthetic_openai_key = "sk-" + "a" * 30
        synthetic_gh_token = "ghp_" + "1" * 36
        synthetic_bearer = "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.secretpayload"
        synthetic_pass = "password=SuperSecretPassword123!"

        test_string = (
            f"Here is key {synthetic_google_key} and {synthetic_openai_key}. "
            f"Token {synthetic_gh_token}, {synthetic_bearer}, {synthetic_pass}."
        )

        redacted = agent.redact_sensitive_text(test_string)

        self.assertNotIn(synthetic_google_key, redacted)
        self.assertNotIn(synthetic_openai_key, redacted)
        self.assertNotIn(synthetic_gh_token, redacted)
        self.assertNotIn("SuperSecretPassword123!", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)
        self.assertIn("[REDACTED_TOKEN]", redacted)
        self.assertIn("[REDACTED_SECRET]", redacted)

    def test_agent_run_logs_do_not_contain_raw_secrets(self):
        """Verify agent execution does not write raw user credentials into logs."""
        synthetic_secret = "AIzaSy" + "Z" * 33
        secret_goal = f"Calculate with {synthetic_secret}"

        # Mock agent model responses to execute calculator action then finish
        test_agent = agent.AuraAgent(api_key="mock_key")
        test_agent.call_llm = MagicMock(side_effect=[
            json.dumps({
                "thought": f"Using key {synthetic_secret}",
                "action": "calculator",
                "action_input": f"2 + 2 # secret {synthetic_secret}"
            }),
            json.dumps({
                "thought": "Calculation done",
                "final_answer": "4"
            })
        ])

        with self.assertLogs(agent.logger, level="INFO") as captured_logs:
            res = test_agent.run(secret_goal)

        # Confirm task succeeded without leaking raw secret in captured log messages
        for record in captured_logs.output:
            self.assertNotIn(synthetic_secret, record, f"Raw secret leaked into log record: {record}")


if __name__ == "__main__":
    unittest.main()

