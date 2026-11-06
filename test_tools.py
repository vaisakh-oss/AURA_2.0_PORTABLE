import os
import tempfile
import unittest
from pathlib import Path
import tools


class TestAuraTools(unittest.TestCase):
    """
    Automated test suite for AURA tool layer (tools.py).
    Tests TOOL_REGISTRY, calculator, python_exec, save_note, read_notes, web_search, and error handling.
    """

    # -----------------------------------------------------------------
    # 1. TOOL REGISTRY TESTS
    # -----------------------------------------------------------------
    def test_tool_registry_keys(self):
        """Verify TOOL_REGISTRY contains exactly the 6 required tools."""
        expected_tools = {"web_search", "calculator", "python_exec", "save_note", "read_notes", "knowledge_search"}
        self.assertEqual(set(tools.TOOL_REGISTRY.keys()), expected_tools)

    def test_run_tool_unknown_tool(self):
        """Verify run_tool() handles unknown tools gracefully with a controlled error message."""
        result = tools.run_tool("unknown_tool_name", "sample_input")
        self.assertIn("TOOL ERROR: Unknown tool", result)

    # -----------------------------------------------------------------
    # 2. CALCULATOR TESTS (SAFE AST EVALUATION)
    # -----------------------------------------------------------------
    def test_calculator_basic_arithmetic(self):
        """Test basic mathematical expression 2 + 3 * 4."""
        result = tools.calculator("2 + 3 * 4")
        self.assertEqual(float(result), 14.0)

    def test_calculator_percentage(self):
        """Test calculation 15 / 100 * 2400."""
        result = tools.calculator("15 / 100 * 2400")
        self.assertEqual(float(result), 360.0)

    def test_calculator_sqrt(self):
        """Test safe math function sqrt(16)."""
        result = tools.calculator("sqrt(16)")
        self.assertEqual(float(result), 4.0)

    def test_calculator_unsafe_expression(self):
        """Verify that unsafe import/system calls are rejected."""
        result = tools.calculator('__import__("os").system("echo hacked")')
        self.assertTrue(result.startswith("CALCULATOR ERROR"))

    def test_calculator_division_by_zero(self):
        """Verify controlled error on division by zero."""
        result = tools.calculator("10 / 0")
        self.assertIn("Division by zero", result)

    def test_calculator_negative_exponent(self):
        """Verify negative exponent calculation behaves normally (2 ** -3 == 0.125)."""
        result = tools.calculator("2 ** -3")
        self.assertEqual(float(result), 0.125)

    def test_calculator_excessive_exponent_blocked(self):
        """Verify excessive exponents (e.g. 9**999999 or 10**1001) are rejected to prevent DoS."""
        res1 = tools.calculator("9 ** 999999")
        self.assertIn("CALCULATOR ERROR", res1)
        res2 = tools.calculator("10 ** 1001")
        self.assertIn("CALCULATOR ERROR", res2)

    def test_calculator_excessive_length_blocked(self):
        """Verify expressions exceeding length bounds are rejected."""
        long_expr = "1 + " * 300 + "1"
        res = tools.calculator(long_expr)
        self.assertIn("CALCULATOR ERROR", res)
        self.assertIn("exceeds maximum allowed limit", res)

    def test_calculator_deeply_nested_expression_blocked(self):
        """Verify deeply nested expressions exceeding AST depth limit are rejected."""
        nested = "-(" * 25 + "1" + ")" * 25
        res = tools.calculator(nested)
        self.assertIn("CALCULATOR ERROR", res)
        self.assertIn("too deeply nested", res)

    # -----------------------------------------------------------------
    # 3. PYTHON EXECUTOR SECURITY TESTS (DISABLED BY DEFAULT)
    # -----------------------------------------------------------------
    def test_python_exec_disabled_by_default(self):
        """Verify Python execution is disabled by default for security."""
        result = tools.python_exec("print([i for i in range(10)])")
        self.assertIn("PYTHON EXEC SECURITY ERROR", result)
        self.assertIn("disabled by default for security", result)

    def test_python_exec_host_execution_prevented(self):
        """Verify host execution is prevented for arbitrary computation."""
        result = tools.python_exec("print(sum(i * i for i in range(5)))")
        self.assertIn("PYTHON EXEC SECURITY ERROR", result)

    def test_python_exec_blocks_host_shell_commands(self):
        """Verify regression: default configuration cannot execute host shell commands."""
        test_file = Path("/tmp/aura_security_test_shell.tmp")
        if test_file.exists():
            test_file.unlink()

        result = tools.python_exec(f"import posixpath; posixpath.os.system('touch {test_file}')")
        self.assertIn("PYTHON EXEC SECURITY ERROR", result)
        self.assertFalse(test_file.exists())

    def test_python_exec_blocks_host_secrets_access(self):
        """Verify regression: default configuration cannot inspect host environment secrets."""
        result = tools.python_exec("import os; print(os.environ.get('GEMINI_API_KEY'))")
        self.assertIn("PYTHON EXEC SECURITY ERROR", result)

    def test_python_exec_blocks_network_requests(self):
        """Verify regression: default configuration cannot make outbound network requests."""
        result = tools.python_exec("import urllib.request; urllib.request.urlopen('https://httpbin.org/get')")
        self.assertIn("PYTHON EXEC SECURITY ERROR", result)

    def test_python_exec_empty_code_error(self):
        """Verify empty code returns controlled error."""
        result = tools.python_exec("   ")
        self.assertIn("PYTHON EXEC ERROR: Code cannot be empty", result)

    def test_python_ast_static_validator(self):
        """Verify static AST validator continues to detect direct forbidden imports."""
        with self.assertRaises(ValueError):
            tools._validate_python_ast("import os\nprint(os.getcwd())")
        with self.assertRaises(ValueError):
            tools._validate_python_ast("open('secret.txt', 'w')")

    # -----------------------------------------------------------------
    # 4. NOTES TESTS (ISOLATED TEMPORARY ENVIRONMENT)
    # -----------------------------------------------------------------
    def test_notes_operations(self):
        """Test save_note, read_notes, non-existent file handling, and keyword filtering."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)

        # Remove temp file to test non-existent file behavior
        if tmp_path.exists():
            tmp_path.unlink()

        original_notes_file = tools.NOTES_FILE
        try:
            tools.NOTES_FILE = str(tmp_path)

            # Test 1: Non-existent notes file read
            non_exist_res = tools.read_notes()
            self.assertIn("does not exist", non_exist_res)

            # Test 2: Save note
            save_res = tools.save_note("AURA test project note")
            self.assertIn("SUCCESS", save_res)

            # Test 3: Read saved note
            read_res = tools.read_notes()
            self.assertIn("AURA test project note", read_res)

            # Test 4: Keyword matching
            match_res = tools.read_notes("AURA")
            self.assertIn("AURA test project note", match_res)

            # Test 5: Keyword filtering with no matches
            no_match_res = tools.read_notes("NonExistentKeywordXYZ")
            self.assertIn("No notes matching query", no_match_res)

        finally:
            tools.NOTES_FILE = original_notes_file
            if tmp_path.exists():
                tmp_path.unlink()

    # -----------------------------------------------------------------
    # 5. WEB SEARCH TEST
    # -----------------------------------------------------------------
    def test_web_search(self):
        """Verify web_search returns results or a controlled error/no-results string."""
        result = tools.web_search("Python programming language")
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 0)
        self.assertTrue(
            "Title:" in result or "No search results" in result or "SEARCH ERROR" in result
        )

    # -----------------------------------------------------------------
    # 6. ERROR HANDLING TEST
    # -----------------------------------------------------------------
    def test_run_tool_error_handling(self):
        """Verify run_tool catches execution errors cleanly without crashing."""
        result = tools.run_tool("calculator", "")
        self.assertIn("CALCULATOR ERROR", result)


if __name__ == "__main__":
    unittest.main()
