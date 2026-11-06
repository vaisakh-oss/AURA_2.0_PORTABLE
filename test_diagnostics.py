import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

import diagnostics
import storage
import knowledge_store


class TestSelfDiagnostics(unittest.TestCase):
    """
    Comprehensive unit and integration test suite for AURA Self-Diagnostic & Reliability Engine.
    Verifies component checks, severity classifications, overall health aggregation,
    recommendations, failure isolation, secret redaction, and quick vs full diagnostics.
    """

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.db_path
        storage.initialize_database(self.db_path)
        knowledge_store.initialize_knowledge_tables(self.db_path)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    # 1. Healthy Database
    def test_01_healthy_database(self):
        res = diagnostics.check_database(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)
        self.assertIn("Database accessible", res.message)

    # 2. Missing Database / Path Check
    def test_02_missing_database(self):
        non_existent_db = os.path.join(tempfile.gettempdir(), "non_existent_aura_test.db")
        if os.path.exists(non_existent_db):
            os.remove(non_existent_db)

        res = diagnostics.check_database(db_path=non_existent_db)
        self.assertIn(res.status, [diagnostics.HealthStatus.HEALTHY, diagnostics.HealthStatus.CRITICAL])
        if os.path.exists(non_existent_db):
            os.remove(non_existent_db)

    # 3. Missing Table
    def test_03_missing_table(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE IF EXISTS agent_experiences")
        conn.close()

        res = diagnostics.check_database(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("agent_experiences", res.message)

    # 4. Valid Tool Registry
    def test_04_valid_tool_registry(self):
        res = diagnostics.check_tools()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)
        self.assertIn("registered_tools", res.details)

    # 5. Invalid Tool Registry
    @patch("tools.TOOL_REGISTRY", {"web_search": lambda x: x})
    def test_05_invalid_tool_registry(self):
        res = diagnostics.check_tools()
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("Missing expected registered tool", res.message)

    # 6. Planner Health
    def test_06_planner_health(self):
        res = diagnostics.check_planner()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 7. Agent Core Health
    def test_07_agent_health(self):
        res = diagnostics.check_agent()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 8. Memory Health
    def test_08_memory_health(self):
        res = diagnostics.check_memory(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 9. RAG Health (Cold Start / Empty DB)
    def test_09_rag_health(self):
        res = diagnostics.check_rag(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)
        self.assertEqual(res.details.get("document_count"), 0)
        self.assertTrue(res.details.get("cold_start"))

    # 9b. RAG Health with Indexed Documents
    def test_09b_rag_health_with_documents(self):
        doc = knowledge_store.add_document("sample.txt", "txt", 100, "sample_hash_123", db_path=self.db_path)
        knowledge_store.add_chunks(doc["id"], [{"chunk_index": 1, "text": "Sample text", "embedding": [0.1]*384}], db_path=self.db_path)
        res = diagnostics.check_rag(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)
        self.assertEqual(res.details.get("document_count"), 1)
        self.assertFalse(res.details.get("cold_start"))

    # 9c. RAG Health Missing Database File
    def test_09c_rag_health_missing_database_file(self):
        res = diagnostics.check_rag(db_path="/tmp/non_existent_rag_test_db.db")
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("not found", res.message)

    # 9d. RAG Health Missing Schema Table
    def test_09d_rag_health_missing_schema(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("DROP TABLE IF EXISTS document_chunks")
        conn.close()
        res = diagnostics.check_rag(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("missing required table", res.message)

    # 9e. RAG Health Embedding Generation Failure
    @patch("knowledge.embed_query", side_effect=RuntimeError("Embedding model load error"))
    def test_09e_rag_health_embedding_failure(self, mock_emb):
        res = diagnostics.check_rag(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("embedding model failure", res.message)

    # 9f. RAG Health Retrieval Failure
    @patch("knowledge.search_knowledge", side_effect=RuntimeError("Vector search computation error"))
    def test_09f_rag_health_search_failure(self, mock_search):
        res = diagnostics.check_rag(db_path=self.db_path)
        self.assertEqual(res.status, diagnostics.HealthStatus.CRITICAL)
        self.assertIn("retrieval query failure", res.message)

    # 10. Vision Health
    def test_10_vision_health(self):
        res = diagnostics.check_vision()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 11. Experience Health
    def test_11_experience_health(self):
        res = diagnostics.check_experience(db_path=self.db_path)
        self.assertIn(res.status, [diagnostics.HealthStatus.HEALTHY, diagnostics.HealthStatus.DEGRADED])

    # 12. Analytics Health
    def test_12_analytics_health(self):
        res = diagnostics.check_analytics(db_path=self.db_path)
        self.assertIn(res.status, [diagnostics.HealthStatus.HEALTHY, diagnostics.HealthStatus.DEGRADED])

    # 13. Configuration Present
    @patch("config.GEMINI_API_KEY", "AIzaSyTestKey1234567890")
    def test_13_configuration_present(self):
        res = diagnostics.check_configuration()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 14. Configuration Missing
    @patch("config.GEMINI_API_KEY", "")
    def test_14_configuration_missing(self):
        res = diagnostics.check_configuration()
        self.assertEqual(res.status, diagnostics.HealthStatus.DEGRADED)

    # 15. Evaluation Harness Health
    def test_15_evaluation_harness_health(self):
        res = diagnostics.check_evaluation()
        self.assertEqual(res.status, diagnostics.HealthStatus.HEALTHY)

    # 16. Degraded Aggregation
    def test_16_degraded_aggregation(self):
        c1 = diagnostics.HealthCheckResult("Comp1", diagnostics.HealthStatus.HEALTHY, "Check1", "OK", 1.0)
        c2 = diagnostics.HealthCheckResult("Comp2", diagnostics.HealthStatus.DEGRADED, "Check2", "Degraded", 1.0)
        overall = diagnostics.calculate_overall_health([c1, c2])
        self.assertEqual(overall, diagnostics.HealthStatus.DEGRADED)

    # 17. Critical Aggregation
    def test_17_critical_aggregation(self):
        c1 = diagnostics.HealthCheckResult("Comp1", diagnostics.HealthStatus.HEALTHY, "Check1", "OK", 1.0)
        c2 = diagnostics.HealthCheckResult("Comp2", diagnostics.HealthStatus.DEGRADED, "Check2", "Degraded", 1.0)
        c3 = diagnostics.HealthCheckResult("Comp3", diagnostics.HealthStatus.CRITICAL, "Check3", "Broken", 1.0)
        overall = diagnostics.calculate_overall_health([c1, c2, c3])
        self.assertEqual(overall, diagnostics.HealthStatus.CRITICAL)

    # 18. Unknown Aggregation
    def test_18_unknown_aggregation(self):
        c1 = diagnostics.HealthCheckResult("Comp1", diagnostics.HealthStatus.UNKNOWN, "Check1", "Unknown", 1.0)
        overall = diagnostics.calculate_overall_health([c1])
        self.assertEqual(overall, diagnostics.HealthStatus.UNKNOWN)

    # 19. Recommendation Generation
    def test_19_recommendation_generation(self):
        res = diagnostics.HealthCheckResult("Database", diagnostics.HealthStatus.CRITICAL, "DB", "Error connecting", 1.0)
        rec = diagnostics.generate_recommendation(res)
        self.assertIsNotNone(rec)
        self.assertIn("SQLite database", rec)

    # 20. Exception Isolation
    def test_20_exception_isolation(self):
        def crashing_check():
            raise RuntimeError("Unexpected failure during diagnostic test")

        report = diagnostics._execute_check_list([crashing_check], db_path=self.db_path)
        self.assertIsNotNone(report)
        self.assertGreater(len(report.checks), 0)
        self.assertEqual(report.checks[0].status, diagnostics.HealthStatus.UNKNOWN)

    # 21. Secret Redaction
    @patch("config.GEMINI_API_KEY", "AIzaSySuperSecretApiKeyDoNotExpose12345")
    def test_21_secret_redaction(self):
        res = diagnostics.check_configuration()
        serialized = str(res.to_dict())
        self.assertNotIn("AIzaSySuperSecretApiKeyDoNotExpose12345", serialized)
        self.assertIn("gemini_api_key_configured", serialized)

    # 22. Quick Diagnostics
    def test_22_quick_diagnostics(self):
        report = diagnostics.run_quick_diagnostics(db_path=self.db_path)
        self.assertEqual(len(report["checks"]), 5)

    # 23. Full Diagnostics
    def test_23_full_diagnostics(self):
        report = diagnostics.run_system_diagnostics(db_path=self.db_path)
        self.assertEqual(len(report["checks"]), 11)

    # 24. Repeated Diagnostics
    def test_24_repeated_diagnostics(self):
        r1 = diagnostics.run_quick_diagnostics(db_path=self.db_path)
        r2 = diagnostics.run_quick_diagnostics(db_path=self.db_path)
        self.assertEqual(r1["healthy_count"], r2["healthy_count"])

    # 25. No Mutation of User Data
    def test_25_no_mutation_of_user_data(self):
        storage.add_memory("user prefers dark mode", "USER_PREFERENCE", 0.8, db_path=self.db_path)
        mems_before = storage.list_memories(db_path=self.db_path)

        diagnostics.run_system_diagnostics(db_path=self.db_path)

        mems_after = storage.list_memories(db_path=self.db_path)
        self.assertEqual(len(mems_before), len(mems_after))
        self.assertEqual(mems_before[0]["content"], mems_after[0]["content"])

    # 26. Integration End-to-End Test
    def test_26_integration_end_to_end(self):
        report = diagnostics.run_system_diagnostics(db_path=self.db_path)
        self.assertIn("overall_status", report)
        self.assertIn("checked_at", report)
        self.assertEqual(len(report["checks"]), 11)
        self.assertIn(report["overall_status"], ["HEALTHY", "DEGRADED", "CRITICAL", "UNKNOWN"])


if __name__ == "__main__":
    unittest.main()
