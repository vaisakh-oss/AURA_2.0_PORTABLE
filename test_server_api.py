import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import storage
import server

class TestServerApi(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_api.db")
        storage.initialize_database(self.db_path)
        self.patcher_db = patch("storage.get_default_db_file", return_value=self.db_path)
        self.patcher_db.start()
        self.client = TestClient(server.app)

    def tearDown(self):
        self.patcher_db.stop()
        self.temp_dir.cleanup()

    def test_01_status_endpoint(self):
        res = self.client.get("/api/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "online")
        self.assertEqual(data["agent"], "AURA 2.0")

    def test_02_conversations_crud(self):
        # 1. Create conversation
        res = self.client.post("/api/conversations", json={"title": "Test Chat"})
        self.assertEqual(res.status_code, 200)
        conv_id = res.json()["id"]
        self.assertTrue(conv_id)

        # 2. List conversations
        res = self.client.get("/api/conversations")
        self.assertEqual(res.status_code, 200)
        convs = res.json()["conversations"]
        self.assertIn(conv_id, [c["id"] for c in convs])

        # 3. Get history (empty initially)
        res = self.client.get(f"/api/conversations/{conv_id}")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["messages"], [])

        # 4. Delete conversation
        res = self.client.delete(f"/api/conversations/{conv_id}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deleted"])

    def test_03_chat_submission_with_agent_mock(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "final_answer": "42 is the answer.",
            "status": "completed",
            "steps": 1,
            "status_steps": [{"label": "Planning", "elapsed_time": 0.2}],
            "citations": [],
            "tool_results": []
        }

        with patch("server.get_agent", return_value=mock_agent):
            res = self.client.post(
                "/api/chat",
                data={"message": "What is the answer to life?"}
            )
            self.assertEqual(res.status_code, 200)
            data = res.json()
            self.assertEqual(data["role"], "assistant")
            self.assertEqual(data["content"], "42 is the answer.")
            self.assertEqual(data["status"], "completed")
            self.assertTrue(data["conversation_id"])

            # Verify saved messages
            conv_id = data["conversation_id"]
            hist_res = self.client.get(f"/api/conversations/{conv_id}")
            self.assertEqual(len(hist_res.json()["messages"]), 2)

    def test_04_memory_endpoints(self):
        # 1. Create memory
        res = self.client.post("/api/memory", json={
            "content": "User prefers concise markdown tables",
            "category": "USER_PREFERENCE",
            "importance": 0.9,
            "confidence": 0.95
        })
        self.assertEqual(res.status_code, 200)
        mem_id = res.json()["id"]

        # 2. List memories
        res = self.client.get("/api/memory")
        self.assertEqual(res.status_code, 200)
        self.assertGreaterEqual(res.json()["count"], 1)

        # 3. Filter memories
        res = self.client.get("/api/memory?category=USER_PREFERENCE")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(any(m["id"] == mem_id for m in res.json()["memories"]))

        # 4. Delete memory
        res = self.client.delete(f"/api/memory/{mem_id}")
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["deleted"])

    def test_05_analytics_and_settings(self):
        res = self.client.get("/api/analytics/metrics")
        self.assertEqual(res.status_code, 200)
        self.assertIn("metrics", res.json())

        res = self.client.get("/api/analytics/history")
        self.assertEqual(res.status_code, 200)
        self.assertIn("runs", res.json())

        res = self.client.get("/api/settings")
        self.assertEqual(res.status_code, 200)
        self.assertIn("model", res.json())

        res = self.client.post("/api/settings", json={"max_steps": 10})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["max_steps"], 10)

    def test_06_duplicate_prevention_and_retry(self):
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "final_answer": "Initial response.",
            "status": "completed",
            "steps": 1,
            "status_steps": [{"label": "Step 1", "elapsed_time": 0.1}],
            "citations": [],
            "tool_results": []
        }

        with patch("server.get_agent", return_value=mock_agent):
            # 1. First submission
            res = self.client.post("/api/chat", data={"message": "First message"})
            self.assertEqual(res.status_code, 200)
            conv_id = res.json()["conversation_id"]

            # 2. Concurrency test: simulate active run
            server._active_conversations.add(conv_id)
            conflict_res = self.client.post(
                "/api/chat",
                data={"message": "Second message", "conversation_id": conv_id}
            )
            self.assertEqual(conflict_res.status_code, 409)
            server._active_conversations.discard(conv_id)

            # 3. Retry submission (is_retry=True): verify no duplicate user message
            hist_before = self.client.get(f"/api/conversations/{conv_id}").json()["messages"]
            user_msgs_before = [m for m in hist_before if m["role"] == "user"]
            self.assertEqual(len(user_msgs_before), 1)

            retry_res = self.client.post(
                "/api/chat",
                data={"message": "First message", "conversation_id": conv_id, "is_retry": "true"}
            )
            self.assertEqual(retry_res.status_code, 200)

            hist_after = self.client.get(f"/api/conversations/{conv_id}").json()["messages"]
            user_msgs_after = [m for m in hist_after if m["role"] == "user"]
            self.assertEqual(len(user_msgs_after), 1)  # No duplicate user message added!

    def test_07_provider_health_transitions(self):
        # Test transition to unauthorized on 403 error
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "final_answer": "Provider error",
            "status": "failed",
            "error": "403 Forbidden. Request not allowed by policy",
            "steps": 1,
            "status_steps": [],
            "citations": [],
            "tool_results": []
        }

        with patch("server.get_agent", return_value=mock_agent):
            res = self.client.post("/api/chat", data={"message": "Test prompt"})
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.json()["provider_status"], "unauthorized")

            status_res = self.client.get("/api/status")
            self.assertEqual(status_res.json()["provider_status"], "unauthorized")

if __name__ == "__main__":
    unittest.main()

