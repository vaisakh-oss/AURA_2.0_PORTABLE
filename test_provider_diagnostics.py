"""
Dedicated Provider Diagnostics & Reliability Test Suite for AURA.
Verifies structured error classification across all 10 error categories,
sensitive credential redaction (including AQ. and AIza keys),
fail-fast on daily quota exhaustion, descriptive event timeline labels,
and provider health reporting.
"""

import os
import json
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import agent
from agent import (
    AuraAgent,
    AgentExecutionError,
    ProviderTimeoutError,
    PermanentProviderError,
    ProviderQuotaError,
    ProviderAuthError,
    ProviderNotFoundError,
    ProviderBadRequestError,
    ProviderUnavailableError,
    ProviderParsingError,
    parse_provider_error,
    redact_sensitive_text,
)
import storage


class TestProviderDiagnosticsAndErrorClassification(unittest.TestCase):
    """Unit tests for parse_provider_error and error categorization."""

    def test_01_classify_http_400_bad_request(self):
        err = Exception("400 INVALID_ARGUMENT: unsupported generation parameter 'top_k'")
        diag = parse_provider_error(err, model="gemini-2.5-flash")
        self.assertEqual(diag["http_status"], 400)
        self.assertEqual(diag["error_category"], "bad_request")
        self.assertEqual(diag["provider_error_code"], "INVALID_ARGUMENT")
        self.assertIn("Invalid request arguments", diag["actionable_guidance"])

    def test_02_classify_http_401_unauthenticated(self):
        err = Exception("401 UNAUTHENTICATED: API_KEY_INVALID. The provided API key is invalid.")
        diag = parse_provider_error(err, model="gemini-2.5-flash")
        self.assertEqual(diag["http_status"], 401)
        self.assertEqual(diag["error_category"], "unauthorized")
        self.assertEqual(diag["provider_error_code"], "UNAUTHENTICATED")
        self.assertIn("GEMINI_API_KEY", diag["actionable_guidance"])

    def test_03_classify_http_403_policy_restricted(self):
        err = Exception("Request to POST /v1beta/models/gemini-2.5-flash-lite on generativelanguage.googleapis.com not allowed by policy")
        diag = parse_provider_error(err, model="gemini-2.5-flash-lite")
        self.assertEqual(diag["error_category"], "policy_restricted")
        self.assertIn("Generative Language API", diag["actionable_guidance"])
        self.assertIn("generativelanguage.googleapis.com", diag["actionable_guidance"])

    def test_04_classify_http_404_model_not_found(self):
        err = Exception("404 NOT_FOUND: models/gemini-2.0-flash is no longer available. Please update your code.")
        diag = parse_provider_error(err, model="gemini-2.0-flash")
        self.assertEqual(diag["http_status"], 404)
        self.assertEqual(diag["error_category"], "model_not_found")
        self.assertEqual(diag["provider_error_code"], "NOT_FOUND")
        self.assertIn("Settings", diag["actionable_guidance"])

    def test_05_classify_http_408_and_504_timeout(self):
        err1 = Exception("504 DEADLINE_EXCEEDED: Deadline expired before operation could complete.")
        diag1 = parse_provider_error(err1, model="gemini-2.5-flash")
        self.assertEqual(diag1["http_status"], 504)
        self.assertEqual(diag1["error_category"], "timeout")
        self.assertEqual(diag1["provider_error_code"], "DEADLINE_EXCEEDED")

        err2 = Exception("Request timed out after 30.0s")
        diag2 = parse_provider_error(err2, model="gemini-2.5-flash")
        self.assertEqual(diag2["error_category"], "timeout")

    def test_06_classify_http_429_quota_exceeded_with_retry_info(self):
        raw_msg = (
            "429 RESOURCE_EXHAUSTED: You exceeded your current quota. "
            "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, "
            "limit: 20, model: gemini-2.5-flash-lite, description: Generate content requests per day limit exceeded on Free Tier. "
            "Please retry in 12h25m28s."
        )
        err = Exception(raw_msg)
        diag = parse_provider_error(err, model="gemini-2.5-flash-lite")
        self.assertEqual(diag["http_status"], 429)
        self.assertEqual(diag["error_category"], "quota_exceeded")
        self.assertEqual(diag["provider_error_code"], "RESOURCE_EXHAUSTED")
        self.assertEqual(diag["retry_after"], "12h25m28")
        self.assertIn("20 requests/day", diag["actionable_guidance"])
        self.assertIn("12h25m28", diag["actionable_guidance"])

    def test_07_classify_http_503_service_unavailable(self):
        err = Exception("503 UNAVAILABLE: This model is currently experiencing high demand. Spikes in demand are usually temporary.")
        diag = parse_provider_error(err, model="gemini-3.8-flash")
        self.assertEqual(diag["http_status"], 503)
        self.assertEqual(diag["error_category"], "service_unavailable")
        self.assertEqual(diag["provider_error_code"], "UNAVAILABLE")
        self.assertIn("high temporary demand", diag["actionable_guidance"])

    def test_08_classify_network_connection_error(self):
        err = Exception("Failed to connect to generativelanguage.googleapis.com: Connection refused (errno 61)")
        diag = parse_provider_error(err, model="gemini-2.5-flash")
        self.assertEqual(diag["error_category"], "network_error")

    def test_09_classify_empty_response_parsing_error(self):
        err = Exception("Gemini returned an empty response.")
        diag = parse_provider_error(err, model="gemini-2.5-flash")
        self.assertEqual(diag["error_category"], "parsing_error")

    def test_10_redact_sensitive_credentials_including_aq_keys(self):
        sample_text = (
            "Calling API with key AIzaSyA1234567890abcdefghijklmnopqrstuv and "
            "Vertex key AQ.AB1234567890abcdefghijklmnopqrstuvwxyz1234567890 "
            "and bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.token.signature"
        )
        redacted = redact_sensitive_text(sample_text)
        self.assertNotIn("AIzaSyA", redacted)
        self.assertNotIn("AQ.AB", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)


class TestProviderExecutionFailFastAndTimelineLabels(unittest.TestCase):
    """Integration tests verifying fast-fail, typed exceptions, and event labels."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_provider.db")
        storage.initialize_database(self.temp_db)
        self.agent = AuraAgent(
            api_key="mock_key",
            model="gemini-2.5-flash-lite",
            max_steps=4,
            db_path=self.temp_db,
            max_retries=3
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_11_fail_fast_on_daily_quota_429(self):
        """Verify that daily quota 429 fails fast immediately without wasting 3 sleep attempts."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception(
            "429 RESOURCE_EXHAUSTED: Generate content requests per day limit exceeded on Free Tier. Please retry in 12h."
        )
        self.agent.client = mock_client

        with patch("time.sleep") as mock_sleep:
            with self.assertRaises(ProviderQuotaError) as ctx:
                self.agent._call_gemini_with_retry("Prompt")
            self.assertEqual(mock_client.models.generate_content.call_count, 1)
            mock_sleep.assert_not_called()
            self.assertEqual(ctx.exception.diagnostic["error_category"], "quota_exceeded")

    def test_12_react_loop_emits_descriptive_quota_label(self):
        """Verify that the ReAct loop timeline emits clear HTTP 429 label instead of AgentExecutionError."""
        events = []
        def event_cb(ev):
            events.append(ev)

        self.agent.event_callback = event_cb
        self.agent.call_llm = MagicMock(side_effect=ProviderQuotaError(
            "429 RESOURCE_EXHAUSTED",
            diagnostic={"error_category": "quota_exceeded", "http_status": 429, "sanitized_message": "429 Quota", "retry_after": "12h"}
        ))

        res = self.agent.run("Explain thermodynamics")
        self.assertEqual(res["status"], "degraded")
        self.assertIn("Google Gemini Quota Limit Reached (HTTP 429)", res["final_answer"])
        self.assertIn("provider_diagnostic", res)

        # Check emitted timeline events
        provider_failed_events = [e for e in res["events"] if e.get("type") == "provider_failed"]
        self.assertTrue(len(provider_failed_events) >= 1)
        # MUST NOT contain generic 'AgentExecutionError'
        self.assertNotIn("AgentExecutionError", provider_failed_events[0]["label"])
        self.assertIn("HTTP 429", provider_failed_events[0]["label"])

    def test_13_react_loop_emits_descriptive_policy_403_label(self):
        """Verify that policy 403 emits clear HTTP 403 label."""
        self.agent.call_llm = MagicMock(side_effect=PermanentProviderError(
            "Request to POST /v1beta/models on generativelanguage.googleapis.com not allowed by policy",
            diagnostic={"error_category": "policy_restricted", "http_status": 403, "sanitized_message": "Policy blocked"}
        ))

        res = self.agent.run("Analyze report")
        self.assertEqual(res["status"], "failed")
        self.assertIn("403", res["final_answer"])

        provider_failed_events = [e for e in res["events"] if e.get("type") == "provider_failed"]
        self.assertTrue(len(provider_failed_events) >= 1)
        self.assertNotIn("AgentExecutionError", provider_failed_events[0]["label"])
        self.assertIn("HTTP 403", provider_failed_events[0]["label"])

    def test_14_subclass_polymorphism_preserves_existing_catches(self):
        """Ensure all new provider exceptions remain instances of AgentExecutionError."""
        quota_err = ProviderQuotaError("Quota error")
        auth_err = ProviderAuthError("Auth error")
        not_found_err = ProviderNotFoundError("Model not found")
        bad_req_err = ProviderBadRequestError("Bad request")
        unavail_err = ProviderUnavailableError("Unavailable")
        parsing_err = ProviderParsingError("Empty")

        for err in [quota_err, auth_err, not_found_err, bad_req_err, unavail_err, parsing_err]:
            self.assertIsInstance(err, AgentExecutionError)

        for perm_err in [auth_err, not_found_err, bad_req_err]:
            self.assertIsInstance(perm_err, PermanentProviderError)


if __name__ == "__main__":
    unittest.main()

