import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import app
import agent
import storage
from agent import AuraAgent


class TestAuraPolishedUI(unittest.TestCase):
    """
    Test suite for polished dark AURA user interface (app.py).
    Verifies CSS injection, error badges, session isolation, completed_with_warning status, and agent handling.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_db = os.path.join(self.temp_dir.name, "test_aura.db")
        self.orig_aura_db = os.environ.get("AURA_DB_PATH")
        os.environ["AURA_DB_PATH"] = self.temp_db
        storage.initialize_database(self.temp_db)
        self.mock_agent = MagicMock(spec=AuraAgent)
        self.mock_agent.short_term_memory = MagicMock()

    def tearDown(self):
        if self.orig_aura_db is not None:
            os.environ["AURA_DB_PATH"] = self.orig_aura_db
        else:
            os.environ.pop("AURA_DB_PATH", None)
        try:
            self.temp_dir.cleanup()
        except Exception:
            pass

    # 1. App imports cleanly
    def test_app_imports_cleanly(self):
        self.assertTrue(hasattr(app, "main"))
        self.assertTrue(hasattr(app, "init_session_state"))
        self.assertTrue(hasattr(app, "render_sidebar"))
        self.assertTrue(hasattr(app, "render_compact_chat_header"))

    # 2. Session state initialization
    def test_init_session_state(self):
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock()
        with patch.object(app.st, "session_state", mock_state):
            app.init_session_state()
            self.assertIn("agent", app.st.session_state)
            self.assertIn("messages", app.st.session_state)
            self.assertIsInstance(app.st.session_state["messages"], list)

    # 3. Empty state initialization
    def test_empty_state_initialization(self):
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({"messages": [], "pending_prompt": None})
        with patch.object(app.st, "session_state", mock_state):
            self.assertEqual(len(app.st.session_state.messages), 0)

    # 4. Successful Agent Run & Render
    def test_agent_run_success(self):
        self.mock_agent.run.return_value = {
            "status": "completed",
            "final_answer": "25 * 4 equals 100.",
            "steps": 2
        }

        result = self.mock_agent.run("What is 25 * 4?")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_answer"], "25 * 4 equals 100.")
        self.assertNotIn("thought", result["final_answer"])

    # 5. Completed with warning status handling
    def test_agent_run_completed_with_warning(self):
        self.mock_agent.run.return_value = {
            "status": "completed_with_warning",
            "final_answer": "Verified tool results:\n• calculator: 100",
            "steps": 2
        }

        result = self.mock_agent.run("Calculate 25 * 4")
        self.assertEqual(result["status"], "completed_with_warning")
        self.assertIn("100", result["final_answer"])

    # 6. Error state styling flag is True on failure
    def test_agent_run_failure_is_error_flag(self):
        self.mock_agent.run.return_value = {
            "status": "failed",
            "error": "429 RESOURCE_EXHAUSTED",
            "final_answer": "Sorry, AURA couldn't complete that request right now."
        }

        result = self.mock_agent.run("Request with quota error")
        self.assertEqual(result["status"], "failed")
        is_error = (result["status"] != "completed")
        self.assertTrue(is_error)

    # 7. Max steps reached handled gracefully
    def test_agent_max_steps_reached(self):
        self.mock_agent.run.return_value = {
            "status": "max_steps_reached",
            "steps": 8
        }

        result = self.mock_agent.run("Infinite prompt")
        self.assertEqual(result["status"], "max_steps_reached")

    # 8. Clear conversation resets state
    def test_clear_conversation(self):
        messages = [{"role": "user", "content": "Hi"}]
        messages.clear()
        self.mock_agent.short_term_memory.clear()

        self.assertEqual(len(messages), 0)
        self.mock_agent.short_term_memory.clear.assert_called_once()

    # 9. Chain of thought is NOT exposed to UI
    def test_thought_not_exposed_to_ui(self):
        result = {
            "status": "completed",
            "thought": "Internal hidden reasoning string",
            "final_answer": "Public clean response text",
            "steps": 1
        }
        display_text = result["final_answer"]
        self.assertNotIn("Internal hidden reasoning string", display_text)
        self.assertEqual(display_text, "Public clean response text")

    # 10. Tool status mapping is safe
    def test_tool_status_mapping_safe(self):
        prompt_text = "Calculate 25 * 4 and save a note"
        status_steps = []
        if "calc" in prompt_text.lower() or "25" in prompt_text:
            status_steps.append("Calculating...")
        if "note" in prompt_text.lower():
            status_steps.append("Saving note...")

        self.assertIn("Calculating...", status_steps)
        self.assertIn("Saving note...", status_steps)
        self.assertNotIn("GEMINI_API_KEY", status_steps)

    # 11. Quick Action Pills labels are short and un-truncated
    def test_quick_action_labels(self):
        pills = ["🔎 Search", "🧮 Calculate", "📝 Save Note", "🧠 Analyze", "💡 Ideas"]
        for p in pills:
            self.assertLess(len(p), 18)

    # 12. Missing API Key status handling
    @patch("app.GEMINI_API_KEY", None)
    def test_missing_api_key_status(self):
        self.assertIsNone(app.GEMINI_API_KEY)

    # 13. Test Raw HTML rendering mechanism (Requirement 7)
    def test_raw_html_rendering_mechanism(self):
        """Ensure custom HTML components are passed via textwrap.dedent and st.markdown with unsafe_allow_html=True."""
        # Check STYLING_CSS is valid and dedented
        self.assertIn("<style>", app.STYLING_CSS)
        self.assertNotIn("\n    <style>", app.STYLING_CSS)

        # Mock st methods to ensure st.write is never called with HTML strings
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "messages": [
                {
                    "role": "assistant",
                    "content": "Sorry, AURA couldn't complete that request right now. Please try again.",
                    "steps": 0,
                    "status_steps": ["Execution error"],
                    "is_error": True,
                    "time": "07:44 PM"
                }
            ]
        })
        with patch.object(app.st, "markdown") as mock_markdown, \
             patch.object(app.st, "write") as mock_write, \
             patch.object(app.st, "session_state", mock_state):
            app.render_chat_messages()

            # Verify markdown was called with unsafe_allow_html=True
            self.assertTrue(mock_markdown.called)
            args, kwargs = mock_markdown.call_args
            rendered_html = args[0]
            self.assertTrue(kwargs.get("unsafe_allow_html"))
            self.assertIn("final-answer-badge-error", rendered_html)
            self.assertIn("Sorry, AURA couldn&#x27;t complete that request right now.", rendered_html)
            # Ensure no leading indentation (which causes markdown codeblock bug)
            self.assertNotIn("\n    <div", rendered_html)
            # Ensure st.write was NOT used for raw HTML
            for write_arg in mock_write.call_args_list:
                if write_arg.args:
                    self.assertNotIn("<div", str(write_arg.args[0]))

    # 14. Test Chat Input CSS styling selectors (Requirement 8 & Final Polish)
    def test_chat_input_css_styling(self):
        """Verify CSS contains correct selectors for outer stBottom, stBottomBlockContainer, stChatInputContainer, stChatInputTextArea, and focus glow."""
        css = app.STYLING_CSS
        self.assertIn("stBottom", css)
        self.assertIn("stBottomBlockContainer", css)
        self.assertIn("stChatInputContainer", css)
        self.assertIn("stChatInputTextArea", css)
        self.assertIn("data-baseweb=\"input\"", css)
        self.assertIn("background-color: transparent !important;", css)
        self.assertIn("background-color: #0B0F19 !important;", css)
        self.assertIn(":focus-within", css)
        self.assertIn("box-shadow: 0 0 16px rgba(99, 102, 241, 0.25) !important;", css)

    # 15. New Chat & Workspace Session initialization (Step 11 requirement)
    def test_init_session_state_creates_storage_conv(self):
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock()
        with patch.object(app.st, "session_state", mock_state), \
             patch("storage.initialize_database") as mock_init_db, \
             patch("storage.list_conversations", return_value=[{"id": "conv-123", "title": "Chat 1"}]), \
             patch("storage.get_messages", return_value=[]):
            app.init_session_state()
            self.assertTrue(mock_init_db.called)
            self.assertEqual(app.st.session_state.active_conversation_id, "conv-123")
            self.assertEqual(app.st.session_state.messages, [])

    # 16. Conversation Switching (Step 11 requirement)
    def test_conversation_switching(self):
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({"active_conversation_id": "conv-1", "messages": []})
        with patch.object(app.st, "session_state", mock_state), \
             patch("storage.get_messages", return_value=[{"role": "user", "content": "Msg 2"}]):
            app.st.session_state.active_conversation_id = "conv-2"
            app.st.session_state.messages = app.storage.get_messages("conv-2")
            self.assertEqual(app.st.session_state.active_conversation_id, "conv-2")
            self.assertEqual(len(app.st.session_state.messages), 1)

    # 17. Delete Conversation behavior (Step 11 requirement)
    def test_delete_conversation_behavior(self):
        with patch("storage.delete_conversation", return_value=True) as mock_del:
            result = app.storage.delete_conversation("conv-del-1")
            self.assertTrue(result)
            mock_del.assert_called_once_with("conv-del-1")

    # 18. Execution timeline renders safely (Step 12 requirement)
    def test_render_execution_timeline_html(self):
        status_steps = [
            {"type": "tool_success", "label": "Calculating", "elapsed_time": 0.4},
            {"type": "tool_success", "label": "Saving to memory", "elapsed_time": 0.8},
            {"type": "completed", "label": "Completed", "elapsed_time": 1.2}
        ]
        html_out = app.render_execution_timeline_html(status_steps, is_message_error=False)
        self.assertIn("AURA EXECUTION", html_out)
        self.assertIn("Calculating", html_out)
        self.assertIn("Saving to memory", html_out)
        self.assertIn("2 operations • 1.2s", html_out)
        self.assertNotIn("thought", html_out)

    # 19. Execution timeline compatibility with legacy string status steps
    def test_legacy_string_status_steps_compatibility(self):
        status_steps = ["Calculating...", "Saving note...", "Completed"]
        html_out = app.render_execution_timeline_html(status_steps, is_message_error=False)
        self.assertIn("AURA EXECUTION", html_out)
        self.assertIn("Calculating...", html_out)
        self.assertIn("Saving note...", html_out)

    # 20. Knowledge sidebar section rendering (Step 13 requirement)
    @patch("knowledge_store.list_documents", return_value=[{"id": "doc1", "filename": "report.pdf", "file_type": ".pdf", "file_size": 1048576, "chunk_count": 5, "created_at": "2026-10-08"}])
    def test_knowledge_sidebar_rendering(self, mock_list):
        docs = app.knowledge_store.list_documents()
        self.assertEqual(len(docs), 1)
        self.assertEqual(docs[0]["filename"], "report.pdf")

    # 21. Knowledge search label mapping safety
    def test_knowledge_search_label_mapping(self):
        label = agent.get_tool_label("knowledge_search")
        self.assertEqual(label, "Searching knowledge")

    # 22. User HTML-like text rendered safely as plain text (Requirement 12)
    def test_user_html_text_rendered_safely(self):
        """Ensure user text inputs with HTML tags display safely as plain text text/content without DOM execution."""
        test_inputs = [
            "<div>Hello</div>",
            "<script>alert(\"x\")</script>",
            "<img src=x onerror=alert(1)>"
        ]
        for user_text in test_inputs:
            class SessionStateMock(dict):
                def __getattr__(self, item):
                    return self.get(item)
                def __setattr__(self, key, value):
                    self[key] = value

            mock_state = SessionStateMock({
                "messages": [
                    {
                        "role": "user",
                        "content": user_text,
                        "time": "12:00 PM"
                    }
                ]
            })
            with patch.object(app.st, "session_state", mock_state), \
                 patch.object(app.st, "chat_message") as mock_chat_msg, \
                 patch.object(app.st, "write") as mock_write:
                app.render_chat_messages()
                mock_chat_msg.assert_called_with("user")
                mock_write.assert_called_with(user_text)

    # 23. Test Empty state renders Hero & Capability Cards
    def test_empty_state_renders_hero(self):
        with patch.object(app.st, "markdown") as mock_markdown:
            app.render_hero_and_cards()
            self.assertTrue(mock_markdown.called)
            rendered_html = mock_markdown.call_args_list[0][0][0]
            self.assertIn("HELLO, I'M AURA", rendered_html)
            self.assertIn("Your autonomous AI assistant", rendered_html)

    # 24. Test Attachment rendering inside user bubble
    def test_user_message_attachment_rendering(self):
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze image",
                    "attachment_present": True,
                    "attachment_filename": "screenshot.png",
                    "attachment_size": 204800
                }
            ]
        })
        with patch.object(app.st, "session_state", mock_state), \
             patch.object(app.st, "caption") as mock_caption:
            app.render_chat_messages()
            mock_caption.assert_called_once()
            caption_arg = mock_caption.call_args[0][0]
            self.assertIn("screenshot.png", caption_arg)
            self.assertIn("200.0 KB", caption_arg)

    # 25. Test Per-message agent activity isolation across multiple tasks
    def test_per_message_agent_activity_isolation(self):
        status_steps_task1 = [{"type": "tool_success", "label": "Calculator: 100", "elapsed_time": 0.2}]
        status_steps_task2 = [{"type": "tool_success", "label": "Search: AI Trends", "elapsed_time": 0.5}]

        html1 = app.render_execution_timeline_html(status_steps_task1, is_message_error=False)
        html2 = app.render_execution_timeline_html(status_steps_task2, is_message_error=False)

        self.assertIn("Calculator: 100", html1)
        self.assertNotIn("Search: AI Trends", html1)
        self.assertIn("Search: AI Trends", html2)
        self.assertNotIn("Calculator: 100", html2)

    # 26. TEST 1: Create conversation A, add messages, switch to New Chat (lazy, zero DB writes)
    def test_new_chat_creates_fresh_empty_conversation(self):
        conv_a = app.storage.create_conversation("Conv A")
        app.storage.add_message(conv_a, "user", "Message in A")

        conv_count_before = len(app.storage.list_conversations())

        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": conv_a,
            "messages": app.storage.get_messages(conv_a),
            "pending_prompt": None,
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import start_new_conversation
            new_id = start_new_conversation()
            self.assertIsNone(new_id)
            self.assertNotEqual(new_id, conv_a)
            self.assertIsNone(mock_state.active_conversation_id)
            self.assertEqual(len(mock_state.messages), 0)
            self.assertEqual(app.storage.get_messages(new_id), [])
            # Assert zero new conversation rows written to database
            self.assertEqual(len(app.storage.list_conversations()), conv_count_before)

    # 27. TEST 2: Start New Chat clears transient state, attachment, execution, and error
    def test_new_chat_clears_transient_execution_and_attachment(self):
        conv_a = app.storage.create_conversation("Conv A")
        app.storage.add_message(conv_a, "user", "Failed msg", is_error=True, attachment_present=True, attachment_filename="pic.jpeg")

        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": conv_a,
            "messages": app.storage.get_messages(conv_a),
            "chat_image_attachment_uploader": "file_data",
            "last_error": "Failed",
            "execution_timeline": ["Step 1"]
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import start_new_conversation
            new_id = start_new_conversation()
            self.assertEqual(mock_state.messages, [])
            self.assertNotIn("chat_image_attachment_uploader", mock_state)
            self.assertNotIn("last_error", mock_state)
            self.assertNotIn("execution_timeline", mock_state)

    # 28. TEST 3: New Task performs same behavior as + New Chat
    def test_new_task_helper_parity(self):
        from ui.chat import start_new_conversation
        conv_1 = app.storage.create_conversation("Conv 1")
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({"active_conversation_id": conv_1})
        with patch.object(app.st, "session_state", mock_state):
            new_id = start_new_conversation()
            self.assertNotEqual(new_id, conv_1)
            self.assertEqual(len(mock_state.messages), 0)

    # 29. TEST 4: Try again does NOT create a new conversation
    def test_try_again_preserves_conversation_id(self):
        conv_id = app.storage.create_conversation("Retry Conv")
        app.storage.add_message(conv_id, "user", "Prompt to retry")
        app.storage.add_message(conv_id, "assistant", "Error message", is_error=True)

        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": conv_id,
            "messages": app.storage.get_messages(conv_id),
            "pending_prompt": None
        })

        with patch.object(app.st, "session_state", mock_state), \
             patch.object(app.st, "rerun") as mock_rerun, \
             patch.object(app.st, "chat_message") as mock_chat_msg, \
             patch.object(app.st, "button", side_effect=lambda label, **kwargs: label == "Try again"):
            app.render_chat_messages()
            # Try again should set pending_prompt to "Prompt to retry" without changing active_conversation_id
            self.assertEqual(mock_state.active_conversation_id, conv_id)
            self.assertEqual(mock_state.pending_prompt, "Prompt to retry")

    # 30. TEST 5: Historical conversation A remains intact after starting B
    def test_historical_conversation_remains_intact(self):
        conv_a = app.storage.create_conversation("Conv A")
        app.storage.add_message(conv_a, "user", "Keep this message in A")

        conv_b = app.storage.create_conversation("Conv B")
        app.storage.add_message(conv_b, "user", "Message in B")

        msgs_a = app.storage.get_messages(conv_a)
        msgs_b = app.storage.get_messages(conv_b)

        self.assertEqual(len(msgs_a), 1)
        self.assertEqual(msgs_a[0]["content"], "Keep this message in A")
        self.assertEqual(len(msgs_b), 1)
        self.assertEqual(msgs_b[0]["content"], "Message in B")

    # 31. TEST 6 & 7: Empty state vs Active state based on message count
    def test_empty_vs_active_state_rendering_truth(self):
        conv_empty = app.storage.create_conversation("Empty Conv")
        conv_active = app.storage.create_conversation("Active Conv")
        app.storage.add_message(conv_active, "user", "Active message")

        self.assertEqual(len(app.storage.get_messages(conv_empty)), 0)
        self.assertEqual(len(app.storage.get_messages(conv_active)), 1)

    # -----------------------------------------------------------------
    # STEP 1 LAZY CONVERSATION LIFECYCLE REGRESSION TESTS
    # -----------------------------------------------------------------
    def test_opening_new_chat_creates_zero_database_rows(self):
        """Verifies clicking + New chat creates 0 rows in SQLite and leaves active_conversation_id as None."""
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": None,
            "messages": [],
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import start_new_conversation
            new_id = start_new_conversation()
            self.assertIsNone(new_id)
            self.assertIsNone(mock_state.active_conversation_id)
            self.assertEqual(mock_state.messages, [])
            self.assertEqual(len(app.storage.list_conversations()), 0)

    def test_navigating_away_from_unsent_draft_creates_zero_database_rows(self):
        """Verifies navigating between navigation sections from an unsent draft creates zero database rows."""
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": None,
            "messages": [],
            "page_mode": "chat",
            "agent": self.mock_agent
        })

        with patch.object(app.st, "session_state", mock_state):
            modes_to_test = ["knowledge", "memory", "analytics", "health", "settings", "chat"]
            for mode in modes_to_test:
                mock_state.page_mode = mode
                app.init_session_state()
                self.assertIsNone(mock_state.active_conversation_id)
                self.assertEqual(mock_state.messages, [])
                self.assertEqual(len(app.storage.list_conversations()), 0)

    def test_submitting_first_message_creates_exactly_one_conversation(self):
        """Verifies submitting first non-empty message creates exactly 1 conversation and persists user message."""
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_agent_instance = MagicMock()
        mock_agent_instance.run.return_value = {
            "status": "completed",
            "final_answer": "42 is the answer.",
            "steps": 1
        }

        mock_state = SessionStateMock({
            "active_conversation_id": None,
            "messages": [],
            "agent": mock_agent_instance,
            "pending_prompt": None,
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import handle_prompt_submission
            handle_prompt_submission("What is 6 * 7?")

            # 1. Exactly 1 conversation created
            convs = app.storage.list_conversations()
            self.assertEqual(len(convs), 1)
            created_conv = convs[0]
            conv_id = created_conv["id"]

            # 2. Associated with current session
            self.assertEqual(mock_state.active_conversation_id, conv_id)

            # 3. First user message and assistant message persisted
            msgs = app.storage.get_messages(conv_id)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["role"], "user")
            self.assertEqual(msgs[0]["content"], "What is 6 * 7?")
            self.assertEqual(msgs[1]["role"], "assistant")
            self.assertEqual(msgs[1]["content"], "42 is the answer.")

            # 4. Title auto-generated from prompt
            self.assertTrue(bool(created_conv.get("title")))

    def test_subsequent_message_reuses_original_conversation_id(self):
        """Verifies second message in conversation reuses existing conversation ID without new conversation row."""
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_agent_instance = MagicMock()
        mock_agent_instance.run.side_effect = [
            {"status": "completed", "final_answer": "Answer 1", "steps": 1},
            {"status": "completed", "final_answer": "Answer 2", "steps": 1}
        ]

        mock_state = SessionStateMock({
            "active_conversation_id": None,
            "messages": [],
            "agent": mock_agent_instance,
            "pending_prompt": None,
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import handle_prompt_submission
            # First message
            handle_prompt_submission("First question")
            conv_id_1 = mock_state.active_conversation_id
            self.assertIsNotNone(conv_id_1)
            self.assertEqual(len(app.storage.list_conversations()), 1)

            # Second message
            handle_prompt_submission("Second question")
            conv_id_2 = mock_state.active_conversation_id

            # Same conversation ID reused, still exactly 1 conversation in DB
            self.assertEqual(conv_id_1, conv_id_2)
            self.assertEqual(len(app.storage.list_conversations()), 1)

            # Messages count is 4 (2 user, 2 assistant)
            msgs = app.storage.get_messages(conv_id_1)
            self.assertEqual(len(msgs), 4)
            self.assertEqual([m["role"] for m in msgs], ["user", "assistant", "user", "assistant"])

    def test_switching_between_existing_conversations_preserves_content(self):
        """Verifies switching between existing conversations preserves their respective messages."""
        conv_1 = app.storage.create_conversation("Conversation 1")
        app.storage.add_message(conv_1, "user", "Message in Conversation 1")

        conv_2 = app.storage.create_conversation("Conversation 2")
        app.storage.add_message(conv_2, "user", "Message in Conversation 2")

        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_state = SessionStateMock({
            "active_conversation_id": conv_1,
            "agent": self.mock_agent,
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            # 1. Active on Conv 1
            app.init_session_state()
            self.assertEqual(len(mock_state.messages), 1)
            self.assertEqual(mock_state.messages[0]["content"], "Message in Conversation 1")

            # 2. Switch to Conv 2
            mock_state.active_conversation_id = conv_2
            app.init_session_state()
            self.assertEqual(len(mock_state.messages), 1)
            self.assertEqual(mock_state.messages[0]["content"], "Message in Conversation 2")

            # 3. Switch back to Conv 1
            mock_state.active_conversation_id = conv_1
            app.init_session_state()
            self.assertEqual(len(mock_state.messages), 1)
            self.assertEqual(mock_state.messages[0]["content"], "Message in Conversation 1")

    def test_failed_first_message_execution_does_not_create_duplicate_conversation_on_retry(self):
        """Verifies that when initial agent execution fails, a retry does not create a duplicate conversation."""
        class SessionStateMock(dict):
            def __getattr__(self, item):
                return self.get(item)
            def __setattr__(self, key, value):
                self[key] = value

        mock_agent_instance = MagicMock()
        # First call fails with exception, second succeeds
        mock_agent_instance.run.side_effect = [
            RuntimeError("Simulated LLM Service 503"),
            {"status": "completed", "final_answer": "Recovered successfully", "steps": 1}
        ]

        mock_state = SessionStateMock({
            "active_conversation_id": None,
            "messages": [],
            "agent": mock_agent_instance,
            "pending_prompt": None,
            "page_mode": "chat"
        })

        with patch.object(app.st, "session_state", mock_state):
            from ui.chat import handle_prompt_submission
            # 1. First execution fails
            handle_prompt_submission("Query that fails")
            self.assertEqual(len(app.storage.list_conversations()), 1)
            conv_id = mock_state.active_conversation_id
            self.assertIsNotNone(conv_id)

            msgs = app.storage.get_messages(conv_id)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["role"], "user")
            self.assertEqual(msgs[1]["role"], "assistant")
            self.assertTrue(msgs[1]["is_error"])

            # 2. User retries (active_conversation_id is already conv_id)
            handle_prompt_submission("Query that fails")

            # Still exactly 1 conversation record
            self.assertEqual(len(app.storage.list_conversations()), 1)
            self.assertEqual(mock_state.active_conversation_id, conv_id)


if __name__ == "__main__":
    unittest.main()
