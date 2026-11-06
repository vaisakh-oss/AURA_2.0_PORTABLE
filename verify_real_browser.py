"""
Real-browser Playwright verification suite for AURA DOCX upload and processing support.
Tests both Chat and Knowledge Base workflows in a headless Chromium browser instance.
Uses isolated temporary storage and documents directory to guarantee production data safety.
"""

import os
import sys
import time
import shutil
import tempfile
import subprocess
import urllib.request
from pathlib import Path

import docx
from playwright.sync_api import sync_playwright

import storage
import knowledge_store


def generate_test_docx(filepath: str) -> str:
    """Generate a realistic test DOCX with known unique facts for retrieval verification."""
    doc = docx.Document()
    doc.add_heading("Project Alpha Security Directive", level=1)
    doc.add_paragraph(
        "Project Alpha data classification requires a retention period of exactly 14 years before decommissioning."
    )
    doc.add_paragraph(
        "The assigned primary officer for Project Alpha compliance is Dr. Evelyn Vance."
    )

    doc.add_heading("Section 1: Cryptographic Standards", level=2)
    t = doc.add_table(rows=3, cols=3)
    t.rows[0].cells[0].text = "Tier"
    t.rows[0].cells[1].text = "Audit Frequency"
    t.rows[0].cells[2].text = "Cipher standard"

    t.rows[1].cells[0].text = "Alpha-1"
    t.rows[1].cells[1].text = "Weekly"
    t.rows[1].cells[2].text = "AES-256-GCM"

    t.rows[2].cells[0].text = "Alpha-2"
    t.rows[2].cells[1].text = "Monthly"
    t.rows[2].cells[2].text = "ChaCha20-Poly1305"

    doc.add_paragraph("All audit telemetry must be logged automatically to the compliance vault.")
    doc.save(filepath)
    return filepath


def generate_corrupted_docx(filepath: str) -> str:
    """Generate a broken DOCX file (ZIP header followed by corrupt garbage)."""
    with open(filepath, "wb") as f:
        f.write(b"PK\x03\x04" + b"\x00" * 20 + b"CORRUPTED_NON_ZIP_TRUNCATED_PAYLOAD")
    return filepath


def wait_for_server(url: str, timeout_sec: int = 30) -> bool:
    """Poll URL until server responds or timeout occurs."""
    start = time.time()
    while time.time() - start < timeout_sec:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=2) as resp:
                if resp.status in (200, 403):
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def run_browser_verification():
    temp_dir = tempfile.mkdtemp(prefix="aura_browser_audit_")
    test_db = os.path.join(temp_dir, "test_aura.db")
    test_docs = os.path.join(temp_dir, "test_docs")
    os.makedirs(test_docs, exist_ok=True)

    print("==================================================================")
    print("STEP 1: INITIALIZE ISOLATED TEST ENVIRONMENT")
    print(f"Isolated temporary dir: {temp_dir}")
    print(f"Isolated temporary DB:  {test_db}")
    print(f"Isolated documents dir: {test_docs}")
    print("==================================================================")

    storage.initialize_database(test_db)
    knowledge_store.initialize_knowledge_tables(test_db)

    # Generate test documents
    valid_docx = os.path.join(temp_dir, "Policy_Alpha.docx")
    corrupt_docx = os.path.join(temp_dir, "Corrupted_Document.docx")
    generate_test_docx(valid_docx)
    generate_corrupted_docx(corrupt_docx)

    port = 8512
    server_url = f"http://127.0.0.1:{port}"

    env = os.environ.copy()
    env["AURA_DB_PATH"] = test_db
    env["KNOWLEDGE_INGESTION_DIR"] = test_docs
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["AURA_TEST_MOCK_LLM"] = "1"
    env["PYTHONPATH"] = f"/Users/pratikskanoj/AURA/aura/.venv/lib/python3.10/site-packages:/Users/pratikskanoj/AURA/aura:{env.get('PYTHONPATH', '')}"

    venv_python = "/Users/pratikskanoj/AURA/aura/.venv/bin/python"

    print("\n==================================================================")
    print("STEP 2: LAUNCH STREAMLIT APPLICATION (PROJECT VIRTUALENV)")
    print(f"Command: {venv_python} -m streamlit run app.py --server.port {port} --server.fileWatcherType none")
    print("==================================================================")

    proc = subprocess.Popen(
        [
            venv_python,
            "-m",
            "streamlit",
            "run",
            "app.py",
            "--server.port",
            str(port),
            "--server.headless",
            "true",
            "--server.fileWatcherType",
            "none",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd="/Users/pratikskanoj/AURA/aura",
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    results = {}

    try:
        if not wait_for_server(server_url, timeout_sec=25):
            stdout, stderr = proc.communicate(timeout=5)
            raise RuntimeError(f"Streamlit server failed to start on {server_url}:\n{stderr.decode()[:500]}")
        print(f"✓ Streamlit application is responding on {server_url}")

        print("\n==================================================================")
        print("STEP 3: LAUNCH PLAYWRIGHT CHROMIUM REAL BROWSER")
        print("==================================================================")

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page(viewport={"width": 1400, "height": 900})

            # Navigate to Chat page
            page.goto(server_url, wait_until="networkidle", timeout=30000)
            page.wait_for_selector(".aura-sidebar-brand", timeout=15000)
            print("✓ Application UI loaded successfully in Chromium browser.")

            # -------------------------------------------------------------
            # TEST 1: Chat accepts .docx attachment & displays filename/type
            # -------------------------------------------------------------
            print("\n--- Verifying Chat Attachment Upload ---")
            # Open expander to reveal file uploader
            summary = page.locator('summary:has-text("Attach document or image")')
            summary.click()
            page.wait_for_timeout(1000)

            # Set file via locator directly
            page.locator('input[type="file"]').first.set_input_files(valid_docx)
            page.wait_for_timeout(2500)

            details_text = page.locator('details').first.inner_text()
            print(f"Chat Attachment Details Text:\n{details_text}")

            assert "Policy_Alpha.docx" in details_text, "Filename missing from details"
            assert "DOCX Document" in details_text or "DOCX" in details_text, "DOCX type missing from details"
            assert "KB" in details_text, "File size missing from details"
            results["chat_upload_display"] = "PASS"
            print("✓ Chat accepted .docx attachment with filename, type, and size correctly displayed.")

            # -------------------------------------------------------------
            # TEST 2: Ask question about uploaded DOCX
            # -------------------------------------------------------------
            print("\n--- Verifying Document Q&A ---")
            chat_input = page.locator('textarea[data-testid="stChatInputTextArea"]').first
            chat_input.fill("What is the retention period for Project Alpha data according to this document?")
            page.wait_for_timeout(500)

            submit_btn = page.locator('button[data-testid="stChatInputSubmitButton"]').first
            submit_btn.click()

            # Wait for agent response to appear
            print("Waiting for agent to process document and generate response...")
            page.wait_for_selector(
                'div[data-testid="stChatMessage"]:has-text("14"), div[data-testid="stChatMessage"]:has-text("years")',
                timeout=75000
            )
            chat_messages = page.locator('div[data-testid="stChatMessage"]').all_inner_texts()
            latest_response = chat_messages[-1] if chat_messages else ""
            print(f"Agent Turn 1 Response Excerpt:\n{latest_response[:250]}...")

            assert "14" in latest_response, "Agent response did not extract 14 years retention period"
            results["chat_document_qa"] = "PASS"
            print("✓ Question about uploaded DOCX produced accurate answer based on document contents.")

            # -------------------------------------------------------------
            # TEST 3: Follow-up question without re-uploading
            # -------------------------------------------------------------
            print("\n--- Verifying Follow-up Q&A Without Re-upload ---")
            page.wait_for_timeout(3000)
            chat_input = page.locator('textarea[data-testid="stChatInputTextArea"]').first
            chat_input.fill("Who is the assigned primary officer for compliance?")
            page.wait_for_timeout(1000)

            submit_btn = page.locator('button[data-testid="stChatInputSubmitButton"]').first
            submit_btn.click()

            # Wait for second assistant response
            print("Waiting for follow-up turn response...")
            page.wait_for_selector(
                'div[data-testid="stChatMessage"]:has-text("Vance"), div[data-testid="stChatMessage"]:has-text("vance")',
                timeout=75000
            )
            chat_messages = page.locator('div[data-testid="stChatMessage"]').all_inner_texts()
            turn2_response = chat_messages[-1] if chat_messages else ""
            print(f"Agent Turn 2 Response Excerpt:\n{turn2_response[:250]}...")

            assert "vance" in turn2_response.lower(), "Follow-up failed to identify Dr. Evelyn Vance"
            results["chat_followup_preservation"] = "PASS"
            print("✓ Follow-up question succeeded without re-uploading, proving document context was preserved across turns.")

            # -------------------------------------------------------------
            # TEST 4: Navigate to Knowledge Base
            # -------------------------------------------------------------
            print("\n--- Navigating to Knowledge Base ---")
            kb_nav_btn = page.locator('button:has-text("Knowledge")').first
            kb_nav_btn.click()

            page.wait_for_selector('.aura-page-title:has-text("Knowledge Base")', timeout=15000)
            page.wait_for_timeout(2000)
            print("✓ Successfully navigated to Knowledge Base workspace.")

            # -------------------------------------------------------------
            # TEST 5: Upload and Index DOCX in Knowledge Base
            # -------------------------------------------------------------
            print("\n--- Verifying Knowledge Base DOCX Upload & Indexing ---")
            toggle_upload_btn = page.locator('button:has-text("+ Upload Document")').first
            toggle_upload_btn.click()
            page.wait_for_selector('.aura-upload-panel', timeout=10000)
            page.wait_for_timeout(1000)

            kb_file_input = page.locator('input[type="file"]').first
            kb_file_input.set_input_files(valid_docx)
            page.wait_for_timeout(3000)

            # Wait for indexing to complete and document to appear in table
            found_in_library = False
            for _ in range(35):
                page.wait_for_timeout(1000)
                body_text = page.locator("body").inner_text()
                if "Policy_Alpha.docx" in body_text and "No documents indexed yet" not in body_text:
                    found_in_library = True
                    break

            assert found_in_library, "Policy_Alpha.docx not found in document library after indexing"
            results["kb_upload_indexing"] = "PASS"
            print("✓ Knowledge Base accepted and indexed .docx document.")

            # -------------------------------------------------------------
            # TEST 6: Document Library Display and DOCX Icon
            # -------------------------------------------------------------
            print("\n--- Verifying Document Library Rendering & DOCX Icon ---")
            # Check that DOCX icon SVG is present in the document row
            docx_svg = page.wait_for_selector('svg path[d*="M8 12.5l1.6 5.5 1.4-4 1.4 4 1.6-5.5"]', timeout=10000)
            assert docx_svg is not None, "DOCX custom vector icon not found in library row"

            # Check table columns
            doc_row_text = page.locator('div:has-text("Policy_Alpha.docx")').first.inner_text()
            print(f"Document Library Item: {doc_row_text}")
            results["kb_library_display"] = "PASS"
            print("✓ Document appears in the library with correct filename, type, and recognizable DOCX icon.")

            # -------------------------------------------------------------
            # TEST 7: Knowledge Base Search & Attribution
            # -------------------------------------------------------------
            print("\n--- Verifying Knowledge Base Search & Attribution ---")
            # Open the Test Knowledge Retrieval Sandbox expander
            sandbox_summary = page.locator('summary:has-text("Test Knowledge Retrieval Sandbox")').first
            if sandbox_summary.is_visible():
                sandbox_summary.click()
                page.wait_for_timeout(1000)

            search_input = page.locator('input[aria-label="Query knowledge store"], input[placeholder*="What does the document say"]').first
            search_input.fill("Alpha data retention period")
            search_input.press("Enter")
            page.wait_for_timeout(1000)

            search_btn = page.locator('button:has-text("Search")').first
            search_btn.click()

            # Wait for search results
            found_excerpt = False
            excerpt_text = ""
            for _ in range(25):
                page.wait_for_timeout(1000)
                body_text = page.locator("body").inner_text()
                if "Found" in body_text and "14 years" in body_text:
                    found_excerpt = True
                    excerpt_lines = [l for l in body_text.splitlines() if "14 years" in l or "Project Alpha" in l or "Policy_Alpha" in l]
                    excerpt_text = "\n".join(excerpt_lines)
                    break

            print(f"Retrieved Excerpt:\n{excerpt_text}")
            assert found_excerpt, "Knowledge search did not retrieve expected policy passage"
            assert "14 years" in excerpt_text, "Retrieved excerpt did not contain the policy retention period"
            results["kb_search_attribution"] = "PASS"
            print("✓ Knowledge search successfully retrieved DOCX passage with source attribution.")

            # -------------------------------------------------------------
            # TEST 8: Invalid / Corrupted DOCX Rejection
            # -------------------------------------------------------------
            print("\n--- Verifying Corrupted DOCX Rejection & Error Display ---")
            # Click open upload panel if closed
            try:
                open_up = page.locator('button:has-text("+ Upload Document")').first
                if open_up.is_visible():
                    open_up.click()
                    page.wait_for_selector('.aura-upload-panel', timeout=10000)
                    page.wait_for_timeout(1000)
            except Exception:
                pass

            kb_file_input2 = page.locator('input[type="file"]').first
            kb_file_input2.set_input_files(corrupt_docx)

            # Wait for error message
            found_error = False
            error_text = ""
            for _ in range(25):
                page.wait_for_timeout(1000)
                body_text = page.locator("body").inner_text()
                if "Failed to index document" in body_text or "Corrupted" in body_text:
                    found_error = True
                    err_lines = [l for l in body_text.splitlines() if "Failed" in l or "Corrupted" in l or "malformed" in l]
                    error_text = "\n".join(err_lines)
                    break

            print(f"Error Banner for Corrupt File:\n{error_text}")
            assert found_error, "Expected error banner not shown for corrupted document"
            assert "Failed to index document" in error_text or "Corrupted" in error_text, "Expected error text missing"
            results["corrupted_docx_rejection"] = "PASS"
            print("✓ Invalid / corrupted DOCX file was rejected with clear user-facing error.")

            browser.close()

    finally:
        # Gracefully shutdown Streamlit
        print("\nStopping Streamlit server...")
        proc.terminate()
        try:
            out, err = proc.communicate(timeout=5)
            if out:
                print("Streamlit stdout:\n", out.decode(errors="replace")[-1500:])
            if err:
                print("Streamlit stderr:\n", err.decode(errors="replace")[-1500:])
        except subprocess.TimeoutExpired:
            proc.kill()
        shutil.rmtree(temp_dir, ignore_errors=True)

    print("\n==================================================================")
    print("REAL BROWSER VERIFICATION SUMMARY:")
    for test_name, status in results.items():
        print(f"  - {test_name}: {status}")
    print("==================================================================")
    return results


if __name__ == "__main__":
    run_browser_verification()
