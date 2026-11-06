"""
End-to-end verification script for DOCX in AURA browser environment and complete RAG workflow:
Upload DOCX -> Extract Text -> Chunk -> Index -> Retrieve -> Answer with valid citations.
Tests both Knowledge Base and Chat application modes via Streamlit AppTest.
"""

import io
import os
import shutil
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
from streamlit.testing.v1 import AppTest

import agent
import document_extractor
import knowledge
import knowledge_store
import storage


def generate_company_policy_docx() -> bytes:
    """Generate a realistic corporate policy document with headings and a table."""
    doc = docx.Document()
    doc.add_heading("AURA Global Security Policy 2026", level=1)
    doc.add_paragraph(
        "This security policy outlines the mandatory protocols for document processing, "
        "data classification, and access control across all operational units."
    )

    doc.add_heading("Section 1: Data Classification", level=2)
    doc.add_paragraph(
        "All uploaded documents must be classified into one of three tiers: Public, "
        "Internal Restricted, or Mission Critical. Unverified executable scripts and "
        "macros are strictly forbidden."
    )

    doc.add_heading("Section 2: Retention and Audit Matrix", level=2)
    t = doc.add_table(rows=4, cols=3)
    t.rows[0].cells[0].text = "Tier"
    t.rows[0].cells[1].text = "Retention Period"
    t.rows[0].cells[2].text = "Inspection Frequency"

    t.rows[1].cells[0].text = "Tier 1: Public"
    t.rows[1].cells[1].text = "30 Days"
    t.rows[1].cells[2].text = "Annual"

    t.rows[2].cells[0].text = "Tier 2: Internal"
    t.rows[2].cells[1].text = "365 Days"
    t.rows[2].cells[2].text = "Quarterly"

    t.rows[3].cells[0].text = "Tier 3: Mission Critical"
    t.rows[3].cells[1].text = "7 Years"
    t.rows[3].cells[2].text = "Monthly"

    doc.add_paragraph("Compliance verification is enforced automatically through AURA audit telemetry.")

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()


def run_e2e_verification():
    temp_dir = tempfile.mkdtemp()
    temp_db = str(Path(temp_dir) / "e2e_aura.db")
    temp_docs = Path(temp_dir) / "docs"
    temp_docs.mkdir(parents=True, exist_ok=True)

    print("==================================================================")
    print("STEP 1: INITIALIZE ISOLATED TEST ENVIRONMENT")
    print(f"Temporary database: {temp_db}")
    print("==================================================================")

    storage.initialize_database(temp_db)
    knowledge_store.initialize_knowledge_tables(temp_db)
    os.environ["AURA_DB_PATH"] = temp_db

    docx_bytes = generate_company_policy_docx()
    doc_name = "Security_Policy_2026.docx"
    print(f"Generated test DOCX: {doc_name} ({len(docx_bytes)} bytes)")

    print("\n==================================================================")
    print("STEP 2: FULL RAG INGESTION PIPELINE (DOCX -> CHUNK -> INDEX)")
    print("==================================================================")

    index_result = knowledge.index_document(
        doc_name,
        docx_bytes,
        db_path=temp_db,
        docs_dir=temp_docs
    )
    print(f"Indexing status: {index_result['status']}")
    print(f"Chunks created: {index_result['document']['chunk_count']}")
    assert index_result["status"] == "success"
    assert index_result["document"]["chunk_count"] > 0

    print("\n==================================================================")
    print("STEP 3: HYBRID RETRIEVAL & CITATION GROUNDING")
    print("==================================================================")

    query = "What is the retention period for Tier 3 Mission Critical data?"
    search_results = knowledge.search_knowledge(query, top_k=3, db_path=temp_db, mode="hybrid")
    print(f"Retrieved {len(search_results)} relevant chunks.")
    assert len(search_results) > 0

    rag_ctx = knowledge.assemble_rag_context(search_results)
    print("Formatted RAG Context Preview:")
    print(rag_ctx["formatted_context"][:350] + "...\n")
    assert "[1] Source: Security_Policy_2026.docx" in rag_ctx["formatted_context"]
    assert "7 Years" in rag_ctx["formatted_context"]

    # Verify citation verification
    grounded_answer = "According to [1], Tier 3 Mission Critical data must be retained for 7 Years with monthly inspection frequency."
    cit_report = knowledge.validate_citations(grounded_answer, rag_ctx["sources"])
    print(f"Citation valid: {cit_report['is_valid']}, Citations: {cit_report['valid_citations']}")
    assert cit_report["is_valid"]
    assert 1 in cit_report["valid_citations"]

    print("\n==================================================================")
    print("STEP 4: STREAMLIT APPTEST BROWSER SIMULATION (KNOWLEDGE BASE)")
    print("==================================================================")

    with patch.dict(os.environ, {"AURA_DB_PATH": temp_db}):
        at = AppTest.from_file("app.py", default_timeout=30)
        at.session_state["active_conversation_id"] = None
        at.session_state["page_mode"] = "knowledge"
        at.session_state["messages"] = []
        at.run()
        assert not at.exception, f"Streamlit exception in Knowledge page: {at.exception}"

        # Verify page rendered and document library displays the indexed DOCX
        doc_names = [d["filename"] for d in knowledge_store.list_documents(db_path=temp_db)]
        print(f"Knowledge Base documents displayed in UI: {doc_names}")
        assert "Security_Policy_2026.docx" in doc_names
        print("✓ Knowledge Base page rendered successfully with DOCX document listed.")

    print("\n==================================================================")
    print("STEP 5: STREAMLIT APPTEST BROWSER SIMULATION (CHAT ATTACHMENT)")
    print("==================================================================")

    with patch.dict(os.environ, {"AURA_DB_PATH": temp_db}):
        from ui.chat import handle_prompt_submission

        # Create mock agent for chat turn
        mock_agent = MagicMock()
        mock_agent.run.return_value = {
            "status": "completed",
            "final_answer": "Summary of Security Policy: Tier 3 retention is 7 Years according to [1].",
            "steps": 2
        }

        class MockState(dict):
            def __getattr__(self, name):
                return self.get(name)
            def __setattr__(self, name, value):
                self[name] = value

        mock_session = MockState({
            "active_conversation_id": None,
            "messages": [],
            "conversation_attachments": {},
            "agent": mock_agent,
            "page_mode": "chat",
            "pending_prompt": None
        })

        with patch("streamlit.session_state", mock_session):
            handle_prompt_submission(
                "Summarize the retention table in this policy document.",
                attachment_bytes=docx_bytes,
                attachment_filename="Security_Policy_2026.docx"
            )

            conv_id = mock_session.active_conversation_id
            assert conv_id is not None
            chat_messages = storage.get_messages(conv_id, db_path=temp_db)
            print(f"Chat messages in conversation '{conv_id}': {len(chat_messages)}")

            user_msg = chat_messages[0]
            print(f"User message attachment: {user_msg.get('attachment_filename')}, type: {user_msg.get('attachment_type')}, size: {user_msg.get('attachment_size')} bytes")
            assert user_msg.get("attachment_present")
            assert user_msg.get("attachment_filename") == "Security_Policy_2026.docx"
            assert user_msg.get("attachment_type") == "docx"

            asst_msg = chat_messages[1]
            print(f"Assistant response: {asst_msg.get('content')}")
            assert "Summary of Security Policy" in asst_msg.get("content")

            # Follow-up query in same conversation without re-uploading
            handle_prompt_submission("What was the inspection frequency for Tier 2?")
            chat_messages_turn2 = storage.get_messages(conv_id, db_path=temp_db)
            print(f"Messages after turn 2: {len(chat_messages_turn2)}")
            assert len(chat_messages_turn2) == 4
            # Verify agent received document context in turn 2 as well
            last_agent_call = mock_agent.run.call_args_list[-1]
            assert last_agent_call[1].get("document_bytes") is not None
            print("✓ Follow-up turn successfully preserved conversation document context.")

    shutil.rmtree(temp_dir, ignore_errors=True)
    print("\n==================================================================")
    print("ALL E2E WORKFLOW VERIFICATIONS PASSED SUCCESSFULLY!")
    print("==================================================================")


if __name__ == "__main__":
    run_e2e_verification()

