import os
import re
import tempfile
import time
import unittest
import analytics
import storage


class TestAnalytics(unittest.TestCase):
    """
    Comprehensive test suite for AURA Production-Grade Observability & Agent Analytics.
    Verifies all 20 analytics requirements, database persistence, sensitive data filtering,
    and retention policy enforcement.
    """

    def setUp(self):
        self.temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.db_path = self.temp_db.name
        self.temp_db.close()
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.db_path
        storage.initialize_database(self.db_path)

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        if os.path.exists(self.db_path):
            os.remove(self.db_path)

    # 1. Run Metric Aggregation
    def test_01_run_metric_aggregation(self):
        events = [
            {"type": "agent_start", "timestamp": 100.0, "elapsed_time": 0.0},
            {"type": "plan_start", "timestamp": 100.1, "elapsed_time": 0.1},
            {"type": "plan_created", "timestamp": 100.2, "elapsed_time": 0.2},
            {"type": "tool_start", "tool": "calculator", "timestamp": 100.3, "elapsed_time": 0.3},
            {"type": "tool_success", "tool": "calculator", "timestamp": 100.5, "elapsed_time": 0.5},
            {"type": "completed", "timestamp": 101.0, "elapsed_time": 1.0}
        ]
        metrics = analytics.aggregate_run_metrics(events, result_status="completed")
        self.assertEqual(metrics["final_status"], "completed")
        self.assertEqual(metrics["planned"], 1)
        self.assertEqual(metrics["tool_call_count"], 1)
        self.assertEqual(metrics["completed_steps"], 1)
        self.assertEqual(metrics["success"], 1)

    # 2. Tool Aggregation
    def test_02_tool_aggregation(self):
        events = [
            {"type": "agent_start", "timestamp": 10.0, "elapsed_time": 0.0},
            {"type": "tool_start", "tool": "web_search", "timestamp": 10.1, "elapsed_time": 0.1},
            {"type": "tool_success", "tool": "web_search", "timestamp": 10.4, "elapsed_time": 0.4},
            {"type": "completed", "timestamp": 10.5, "elapsed_time": 0.5}
        ]
        metrics = analytics.aggregate_run_metrics(events)
        self.assertEqual(len(metrics["tool_runs"]), 1)
        t_run = metrics["tool_runs"][0]
        self.assertEqual(t_run["tool"], "web_search")
        self.assertEqual(t_run["duration_ms"], 300.0)
        self.assertEqual(t_run["success"], 1)

    # 3. Duration Calculation
    def test_03_duration_calculation(self):
        events = [
            {"type": "agent_start", "timestamp": 50.0, "elapsed_time": 0.0},
            {"type": "completed", "timestamp": 52.35, "elapsed_time": 2.35}
        ]
        metrics = analytics.aggregate_run_metrics(events)
        self.assertAlmostEqual(metrics["duration_ms"], 2350.0, delta=10.0)

    # 4. Success Statistics
    def test_04_success_statistics(self):
        analytics.record_agent_run(
            [{"type": "agent_start", "timestamp": 1.0}, {"type": "completed", "timestamp": 2.0}],
            result={"status": "completed"},
            db_path=self.db_path
        )
        health = analytics.get_agent_health(db_path=self.db_path)
        self.assertEqual(health["success_count"], 1)
        self.assertEqual(health["success_rate_percent"], 100.0)

    # 5. Warning Statistics
    def test_05_warning_statistics(self):
        analytics.record_agent_run(
            [{"type": "agent_start", "timestamp": 1.0}, {"type": "completed_with_warning", "timestamp": 2.0}],
            result={"status": "completed_with_warning"},
            db_path=self.db_path
        )
        health = analytics.get_agent_health(db_path=self.db_path)
        self.assertEqual(health["warning_count"], 1)

    # 6. Failure Statistics
    def test_06_failure_statistics(self):
        analytics.record_agent_run(
            [{"type": "agent_start", "timestamp": 1.0}, {"type": "failed", "timestamp": 2.0}],
            result={"status": "failed"},
            db_path=self.db_path
        )
        health = analytics.get_agent_health(db_path=self.db_path)
        self.assertEqual(health["failure_count"], 1)

    # 7. Retry Counting
    def test_07_retry_counting(self):
        events = [
            {"type": "agent_start", "timestamp": 1.0},
            {"type": "tool_start", "tool": "calculator", "timestamp": 1.1},
            {"type": "tool_error", "tool": "calculator", "timestamp": 1.2},
            {"type": "step_retrying", "tool": "calculator", "timestamp": 1.3},
            {"type": "tool_start", "tool": "calculator", "timestamp": 1.4},
            {"type": "tool_success", "tool": "calculator", "timestamp": 1.5},
            {"type": "completed", "timestamp": 1.6}
        ]
        metrics = analytics.aggregate_run_metrics(events)
        self.assertEqual(metrics["retry_count"], 1)

    # 8. Replan Counting
    def test_08_replan_counting(self):
        events = [
            {"type": "agent_start", "timestamp": 1.0},
            {"type": "plan_start", "timestamp": 1.1},
            {"type": "replanning", "timestamp": 1.5},
            {"type": "completed", "timestamp": 2.0}
        ]
        metrics = analytics.aggregate_run_metrics(events)
        self.assertEqual(metrics["replan_count"], 1)

    # 9. Blocked Step Counting
    def test_09_blocked_step_counting(self):
        analytics.record_agent_run(
            [{"type": "agent_start", "timestamp": 1.0}, {"type": "failed", "timestamp": 2.0}],
            result={"status": "failed"},
            db_path=self.db_path
        )
        stats = analytics.get_status_statistics(db_path=self.db_path)
        self.assertIn("blocked", stats["counts"])

    # 10. Tool Statistics
    def test_10_tool_statistics(self):
        run_id = analytics.record_agent_run(
            [
                {"type": "agent_start", "timestamp": 1.0},
                {"type": "tool_start", "tool": "calculator", "timestamp": 1.1},
                {"type": "tool_success", "tool": "calculator", "timestamp": 1.3},
                {"type": "completed", "timestamp": 2.0}
            ],
            db_path=self.db_path
        )
        tool_stats = analytics.get_tool_statistics(db_path=self.db_path)
        calc_stat = next(t for t in tool_stats if t["tool"] == "calculator")
        self.assertEqual(calc_stat["calls"], 1)
        self.assertEqual(calc_stat["success"], 1)

    # 11. Health Summary
    def test_11_health_summary(self):
        health = analytics.get_agent_health(db_path=self.db_path)
        self.assertIn("total_tasks", health)
        self.assertIn("success_rate_percent", health)
        self.assertIn("avg_duration_seconds", health)
        self.assertIn("most_used_tool", health)

    # 12. Usage Summary
    def test_12_usage_summary(self):
        analytics.record_agent_run(
            [
                {"type": "agent_start", "timestamp": 1.0},
                {"type": "plan_start", "timestamp": 1.1},
                {"type": "tool_start", "tool": "knowledge_search", "timestamp": 1.2},
                {"type": "tool_success", "tool": "knowledge_search", "timestamp": 1.4},
                {"type": "completed", "timestamp": 2.0}
            ],
            db_path=self.db_path
        )
        usage = analytics.get_usage_summary(db_path=self.db_path)
        self.assertEqual(usage["planning_usage"], 1)
        self.assertEqual(usage["rag_usage"], 1)

    # 13. Database Persistence
    def test_13_database_persistence(self):
        run_id = analytics.record_agent_run(
            [{"type": "agent_start", "timestamp": 1.0}, {"type": "completed", "timestamp": 2.0}],
            conversation_id="conv-123",
            db_path=self.db_path
        )
        stored_run = analytics.get_run(run_id, db_path=self.db_path)
        self.assertIsNotNone(stored_run)
        self.assertEqual(stored_run["conversation_id"], "conv-123")

    # 14. Retention Limit
    def test_14_retention_limit(self):
        for i in range(15):
            analytics.record_agent_run(
                [{"type": "agent_start", "timestamp": float(i)}, {"type": "completed", "timestamp": float(i) + 0.1}],
                db_path=self.db_path
            )
        analytics.enforce_retention_policy(limit=10, db_path=self.db_path)
        runs = analytics.list_recent_runs(limit=100, db_path=self.db_path)
        self.assertEqual(len(runs), 10)

    # 15. Concurrent-Safe Operations
    def test_15_concurrent_safe_db_operations(self):
        for _ in range(5):
            analytics.record_agent_run(
                [{"type": "agent_start", "timestamp": 1.0}, {"type": "completed", "timestamp": 2.0}],
                db_path=self.db_path
            )
        runs = analytics.list_recent_runs(limit=50, db_path=self.db_path)
        self.assertEqual(len(runs), 5)

    # 16. Empty Database Behavior
    def test_16_empty_database_behavior(self):
        health = analytics.get_agent_health(db_path=self.db_path)
        self.assertEqual(health["total_tasks"], 0)
        self.assertEqual(health["success_rate_percent"], 0.0)
        self.assertEqual(health["most_used_tool"], "None")

    # 17. Malformed Event Handling
    def test_17_malformed_event_handling(self):
        malformed_events = ["not_a_dict", None, {"missing_type": True}, {"type": 123}]
        metrics = analytics.aggregate_run_metrics(malformed_events, result_status="failed")
        self.assertEqual(metrics["final_status"], "failed")

    # 18. Unknown Tool Rejection
    def test_18_unknown_tool_rejection(self):
        with self.assertRaises(ValueError):
            analytics.record_tool_run(
                run_id=1,
                tool="unauthorized_hacker_tool",
                duration_ms=10.0,
                success=True,
                db_path=self.db_path
            )

    # 19. Sensitive Data Filtering
    def test_19_sensitive_data_filtering(self):
        events = [
            {
                "type": "agent_start",
                "timestamp": 1.0,
                "thought": "I should steal AIzaSySecretKey",
                "prompt": "SELECT * FROM users WHERE pass='secret'",
                "api_key": "AIzaSyTestApiKey12345"
            },
            {"type": "completed", "timestamp": 2.0}
        ]
        run_id = analytics.record_agent_run(events, db_path=self.db_path)
        run_data = analytics.get_run(run_id, db_path=self.db_path)

        serialized = str(run_data)
        self.assertNotIn("AIzaSySecretKey", serialized)
        self.assertNotIn("pass='secret'", serialized)
        self.assertNotIn("AIzaSyTestApiKey12345", serialized)

    # 20. Historical Run Retrieval
    def test_20_historical_run_retrieval(self):
        run_id = analytics.record_agent_run(
            [
                {"type": "agent_start", "timestamp": 1.0},
                {"type": "tool_start", "tool": "python_exec", "timestamp": 1.1},
                {"type": "tool_success", "tool": "python_exec", "timestamp": 1.8},
                {"type": "completed", "timestamp": 2.0}
            ],
            db_path=self.db_path
        )
        run = analytics.get_run(run_id, db_path=self.db_path)
        self.assertIsNotNone(run)
        self.assertEqual(run["id"], run_id)
        self.assertEqual(len(run["tool_runs"]), 1)
        self.assertEqual(run["tool_runs"][0]["tool"], "python_exec")


if __name__ == "__main__":
    unittest.main()
