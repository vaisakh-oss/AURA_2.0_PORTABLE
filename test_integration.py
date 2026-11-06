import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import agent
import memory
import storage
import tools


class TestAuraIntegrationAndHardening(unittest.TestCase):
    """
    Integration, End-to-End Validation & Hardening Test Suite for AURA.
    Tests full ReAct loop, memory flow, tool integration, security boundaries, and edge cases.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_aura.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)
        self.agent = agent.AuraAgent(api_key="mock_key", model="gemini-3.8-flash", max_steps=5, db_path=self.temp_db)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # =================================================================
    # 1. REALISTIC MOCKED SCENARIOS
    # =================================================================
    def test_scenario_1_calculator(self):
        """Scenario 1: User asks 'What is 25 * 4?' -> calculator -> 100 -> final answer."""
        resp1 = json.dumps({
            "thought": "Need to compute 25 * 4",
            "action": "calculator",
            "action_input": "25 * 4"
        })
        resp2 = json.dumps({
            "thought": "Calculation completed",
            "final_answer": "25 * 4 equals 100."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("What is 25 * 4?")
        self.assertEqual(res["status"], "completed")
        self.assertIn("100", res["final_answer"])
        self.assertEqual(res["steps"], 2)

    def test_scenario_2_notes_save_and_read(self):
        """Scenario 2: User requests to save note then read notes."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        if tmp_path.exists():
            tmp_path.unlink()

        orig_file = tools.NOTES_FILE
        try:
            tools.NOTES_FILE = str(tmp_path)

            resp1 = json.dumps({
                "thought": "Saving project name note",
                "action": "save_note",
                "action_input": "Project is named AURA"
            })
            resp2 = json.dumps({
                "thought": "Reading saved notes",
                "action": "read_notes",
                "action_input": "AURA"
            })
            resp3 = json.dumps({
                "thought": "Note workflow complete",
                "final_answer": "Saved note and verified: Project is named AURA."
            })
            self.agent.call_llm = MagicMock(side_effect=[resp1, resp2, resp3])

            res = self.agent.run("Save a note saying my project is called AURA, then read my notes.")
            self.assertEqual(res["status"], "completed")
            self.assertEqual(res["steps"], 3)
            self.assertIn("AURA", res["final_answer"])
        finally:
            tools.NOTES_FILE = orig_file
            if tmp_path.exists():
                tmp_path.unlink()

    def test_scenario_3_web_search_and_calculator(self):
        """Scenario 3: Search distance from Earth to Moon in km, then convert to miles using calculator."""
        resp1 = json.dumps({
            "thought": "Searching for distance to moon",
            "action": "web_search",
            "action_input": "distance from earth to moon in km"
        })
        resp2 = json.dumps({
            "thought": "Converting 384400 km to miles",
            "action": "calculator",
            "action_input": "384400 * 0.621371"
        })
        resp3 = json.dumps({
            "thought": "Synthesizing final answer",
            "final_answer": "The distance is approximately 238,855 miles."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2, resp3])

        res = self.agent.run("Find distance to moon in km and convert to miles.")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["steps"], 3)
        self.assertIn("238,855 miles", res["final_answer"])

    def test_scenario_4_tool_failure_and_recovery(self):
        """Scenario 4: Tool failure (10 / 0) fed back as observation, agent recovers with valid calculation."""
        resp1 = json.dumps({
            "thought": "Accidentally dividing by zero",
            "action": "calculator",
            "action_input": "10 / 0"
        })
        resp2 = json.dumps({
            "thought": "Fixing division by zero to valid division 10 / 2",
            "action": "calculator",
            "action_input": "10 / 2"
        })
        resp3 = json.dumps({
            "thought": "Calculation successful after recovery",
            "final_answer": "10 divided by 2 is 5."
        })
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2, resp3])

        res = self.agent.run("Divide 10 by 0 then recovery")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["steps"], 3)
        self.assertIn("10 divided by 2 is 5", res["final_answer"])

    def test_scenario_5_repeated_invalid_responses_until_max_steps(self):
        """Scenario 5: Repeated invalid model responses until MAX_STEPS is reached."""
        invalid_resp = "Invalid non-JSON model response"
        self.agent.call_llm = MagicMock(return_value=invalid_resp)

        res = self.agent.run("Perform task with broken model output")
        self.assertEqual(res["status"], "max_steps_reached")
        self.assertEqual(res["steps"], 5)

    # =================================================================
    # 2. REACT LOOP HARDENING (CASES A THROUGH O)
    # =================================================================
    def test_case_a_immediate_final_answer(self):
        """Case A: Immediate final answer without tool calls."""
        resp = json.dumps({"thought": "Direct answer", "final_answer": "Hello World"})
        self.agent.call_llm = MagicMock(return_value=resp)

        res = self.agent.run("Say hello")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "Hello World")
        self.assertEqual(res["steps"], 1)

    def test_case_b_one_tool_then_final_answer(self):
        """Case B: One tool execution followed by final answer."""
        resp1 = json.dumps({"thought": "Calculating", "action": "calculator", "action_input": "5 + 5"})
        resp2 = json.dumps({"thought": "Done", "final_answer": "10"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("What is 5 + 5?")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["steps"], 2)

    def test_case_c_multiple_tools_then_final_answer(self):
        """Case C: Multiple tools followed by final answer."""
        resp1 = json.dumps({"thought": "Calc 1", "action": "calculator", "action_input": "1 + 1"})
        resp2 = json.dumps({"thought": "Calc 2", "action": "calculator", "action_input": "2 + 2"})
        resp3 = json.dumps({"thought": "Done", "final_answer": "4"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2, resp3])

        res = self.agent.run("Multi tool call")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["steps"], 3)

    def test_case_d_unknown_tool(self):
        """Case D: Unknown tool action requested."""
        resp1 = json.dumps({"thought": "Unknown tool call", "action": "non_existent_tool", "action_input": "x"})
        resp2 = json.dumps({"thought": "Recovered", "final_answer": "Handled unknown tool"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Run bad tool")
        self.assertEqual(res["status"], "completed")
        prompt2 = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Unknown action 'non_existent_tool'", prompt2)

    def test_case_e_malformed_json(self):
        """Case E: Malformed JSON output from model."""
        resp1 = "{ malformed json string without closing"
        resp2 = json.dumps({"thought": "Fixed", "final_answer": "Recovered from bad JSON"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Fix JSON")
        self.assertEqual(res["status"], "completed")
        prompt2 = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Malformed JSON output", prompt2)

    def test_case_f_missing_action_input(self):
        """Case F: Tool action payload missing action_input field."""
        resp1 = json.dumps({"thought": "Missing input", "action": "web_search"})
        resp2 = json.dumps({"thought": "Fixed", "final_answer": "Fixed payload"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Test missing input")
        self.assertEqual(res["status"], "completed")
        prompt2 = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Missing required field 'action_input'", prompt2)

    def test_case_g_missing_final_answer(self):
        """Case G: JSON response missing both action and final_answer keys."""
        resp1 = json.dumps({"thought": "Empty response body"})
        resp2 = json.dumps({"thought": "Fixed", "final_answer": "Fixed object"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Test empty schema")
        self.assertEqual(res["status"], "completed")
        prompt2 = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("must contain either 'action' or 'final_answer'", prompt2)

    def test_case_h_tool_exception(self):
        """Case H: Tool execution raises exception or returns error payload."""
        resp1 = json.dumps({"thought": "Bad calc", "action": "calculator", "action_input": "1 / 0"})
        resp2 = json.dumps({"thought": "Handled", "final_answer": "Error handled"})
        self.agent.call_llm = MagicMock(side_effect=[resp1, resp2])

        res = self.agent.run("Divide zero")
        self.assertEqual(res["status"], "completed")
        prompt2 = self.agent.call_llm.call_args_list[1][0][0]
        self.assertIn("Division by zero", prompt2)

    def test_case_i_gemini_exception(self):
        """Case I: Gemini API raises unrecoverable RuntimeError."""
        self.agent.call_llm = MagicMock(side_effect=RuntimeError("API Key Invalid or Network Down"))

        res = self.agent.run("Test API exception")
        self.assertEqual(res["status"], "failed")
        self.assertIn("Gemini API error", res["error"])

    def test_case_j_repeated_invalid_model_responses(self):
        """Case J: Model outputs invalid data repeatedly until MAX_STEPS."""
        self.agent.call_llm = MagicMock(return_value="NOT_VALID_JSON")

        res = self.agent.run("Repeated invalid output")
        self.assertEqual(res["status"], "max_steps_reached")
        self.assertEqual(res["steps"], 5)

    def test_case_k_max_steps_reached(self):
        """Case K: Bounded reasoning loop reaches MAX_STEPS limit."""
        resp_tool = json.dumps({"thought": "Looping", "action": "calculator", "action_input": "1 + 1"})
        self.agent.call_llm = MagicMock(return_value=resp_tool)

        res = self.agent.run("Loop forever")
        self.assertEqual(res["status"], "max_steps_reached")
        self.assertEqual(res["steps"], 5)

    def test_case_l_empty_gemini_response(self):
        """Case L: Gemini returns empty response string."""
        self.agent.call_llm = MagicMock(return_value="")

        res = self.agent.run("Test empty response")
        self.assertEqual(res["status"], "max_steps_reached")
        self.assertEqual(res["steps"], 5)

    def test_case_m_markdown_wrapped_json(self):
        """Case M: Model wraps JSON response in Markdown code fences."""
        markdown_resp = "```json\n{\n  \"thought\": \"Markdown test\",\n  \"final_answer\": \"Parsed markdown successfully\"\n}\n```"
        self.agent.call_llm = MagicMock(return_value=markdown_resp)

        res = self.agent.run("Test markdown wrapping")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "Parsed markdown successfully")

    def test_case_n_unexpected_json_fields(self):
        """Case N: Model returns response with extra unexpected JSON fields."""
        extra_resp = json.dumps({
            "thought": "Extra fields test",
            "final_answer": "Parsed extra fields",
            "unexpected_key": "some_value",
            "chain_of_thought": "secret reasoning"
        })
        self.agent.call_llm = MagicMock(return_value=extra_resp)

        res = self.agent.run("Test extra fields")
        self.assertEqual(res["status"], "completed")
        self.assertEqual(res["final_answer"], "Parsed extra fields")

    def test_case_o_wrong_data_types(self):
        """Case O: action_input is a dict or integer instead of a string."""
        dict_input_resp = json.dumps({
            "thought": "Dict input test",
            "action": "calculator",
            "action_input": {"expr": "2+2"}
        })
        final_resp = json.dumps({"thought": "Finished", "final_answer": "Done"})
        self.agent.call_llm = MagicMock(side_effect=[dict_input_resp, final_resp])

        res = self.agent.run("Test dict input")
        self.assertEqual(res["status"], "completed")

    # =================================================================
    # 3. MEMORY & STATE INTEGRATION TESTS
    # =================================================================
    @patch("agent.memory_as_text", return_value="- [2026-10-08] Fact 1: Project is AURA")
    def test_long_term_memory_injected(self, mock_mem):
        """Verify long-term memory is injected into prompt."""
        resp = json.dumps({"thought": "Answer", "final_answer": "OK"})
        self.agent.call_llm = MagicMock(return_value=resp)

        self.agent.run("Test memory prompt injection")
        prompt = self.agent.call_llm.call_args[0][0]
        self.assertIn("Fact 1: Project is AURA", prompt)

    def test_short_term_memory_isolation(self):
        """Verify short-term memory does not leak between independent agent runs."""
        resp = json.dumps({"thought": "Answer", "final_answer": "Run 1 complete"})
        self.agent.call_llm = MagicMock(return_value=resp)

        self.agent.run("Run 1 goal")
        self.assertEqual(len(self.agent.short_term_memory.get_messages()), 2)

        # Second run
        resp2 = json.dumps({"thought": "Answer 2", "final_answer": "Run 2 complete"})
        self.agent.call_llm = MagicMock(return_value=resp2)

        self.agent.run("Run 2 goal")
        messages_run_2 = [m["content"] for m in self.agent.short_term_memory.get_messages()]
        self.assertIn("Run 2 goal", messages_run_2)
        self.assertNotIn("Run 1 goal", messages_run_2)

    # =================================================================
    # 4. SECURITY AUDIT TESTS
    # =================================================================
    def test_security_python_exec_subclasses_blocked(self):
        """Verify python_exec blocks introspection exploit using __subclasses__."""
        res = tools.python_exec("print(()._class_.__bases__[0].__subclasses__())".replace("_class_", "__class__"))
        self.assertIn("SECURITY ERROR", res)

    def test_security_python_exec_globals_blocked(self):
        """Verify python_exec blocks function call globals()."""
        res = tools.python_exec("print(globals())")
        self.assertIn("SECURITY ERROR", res)

    def test_security_python_exec_getattr_blocked(self):
        """Verify python_exec blocks getattr()."""
        res = tools.python_exec("getattr(object, '__subclasses__')()")
        self.assertIn("SECURITY ERROR", res)

    def test_security_calculator_import_blocked(self):
        """Verify calculator rejects import system calls."""
        res = tools.calculator("__import__('os').system('echo hacked')")
        self.assertIn("CALCULATOR ERROR", res)


if __name__ == "__main__":
    unittest.main()
