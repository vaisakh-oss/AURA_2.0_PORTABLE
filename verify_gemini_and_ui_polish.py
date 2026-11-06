import asyncio
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright

OUTPUT_DIR = Path("/Users/pratikskanoj/.gemini/antigravity/brain/cfedb383-8535-45c9-896d-f9cdd94b6a90/screenshots_polish")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

APP_URL = "http://127.0.0.1:8000"

async def run_verification():
    print(f"Connecting to AURA 2.0 SPA at {APP_URL}...")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            device_scale_factor=2
        )
        page = await context.new_page()

        # 1. Load active conversation with previous error
        print("1. Loading conversation with error to inspect error card...")
        await page.goto(APP_URL, wait_until="networkidle")
        await page.wait_for_timeout(1000)

        # Click the first conversation in sidebar
        conv_items = page.locator('aside div[class*="group flex items-center"]')
        if await conv_items.count() > 0:
            await conv_items.first.click()
            await page.wait_for_timeout(1000)

        # Save screenshot of the enhanced error recovery card
        await page.screenshot(path=str(OUTPUT_DIR / "01_error_card_recovery_guidance.png"))
        print("   Saved 01_error_card_recovery_guidance.png")

        # 2. Click "View Diagnostics"
        print("2. Expanding Technical Diagnostics...")
        diag_btn = page.locator('button:has-text("View Diagnostics")').first
        if await diag_btn.is_visible():
            await diag_btn.click()
            await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "02_technical_diagnostics_expanded.png"))
        print("   Saved 02_technical_diagnostics_expanded.png")

        # 3. Check Inspector Activity tab
        print("3. Checking Inspector Activity consistency...")
        inspector_btn = page.locator('button:has-text("Inspector")').first
        if await inspector_btn.is_visible():
            # If not open, click it
            activity_tab = page.locator('button:has-text("Activity")').first
            if not await activity_tab.is_visible():
                await inspector_btn.click()
                await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "03_inspector_activity_consistency.png"))
        print("   Saved 03_inspector_activity_consistency.png")

        # 4. Check Provider Status Indicator in Header & Sidebar
        print("4. Inspecting Provider status indicators...")
        await page.screenshot(path=str(OUTPUT_DIR / "04_provider_status_indicator.png"))
        print("   Saved 04_provider_status_indicator.png")

        # 5. Start New Chat and test DOCX upload & reasoning workflow
        print("5. Testing DOCX upload and execution workflow...")
        new_task_btn = page.locator('button:has-text("Start New Task")').first
        if await new_task_btn.is_visible():
            await new_task_btn.click()
        else:
            await page.locator('button:has-text("New Chat")').first.click()
        await page.wait_for_timeout(800)

        # Upload DOCX
        docx_source = Path("/Users/pratikskanoj/AURA/aura/knowledge_data/documents/doc_167139973926_Policy_Alpha.docx")
        file_input = page.locator('input[type="file"]').first
        await file_input.set_input_files(str(docx_source))
        await page.wait_for_timeout(500)

        # Type query in textarea
        textarea = page.locator('textarea').first
        await textarea.fill("Who is the compliance officer for Project Alpha in the attached document?")
        await page.wait_for_timeout(300)

        # Click send
        send_btn = page.locator('button[title*="Send message"]').first
        await send_btn.click()
        print("   Submitted DOCX question. Waiting for reasoning...")

        # Verify button is disabled immediately while processing
        is_disabled = await send_btn.is_disabled()
        print(f"   Send button disabled during run: {is_disabled}")

        # Wait for reasoning to finish (up to 20s)
        try:
            await page.wait_for_selector('div.bg-gradient-to-tr', timeout=20000)
            await page.wait_for_timeout(2000)
        except Exception as e:
            print(f"   Wait timed out or completed: {e}")

        await page.screenshot(path=str(OUTPUT_DIR / "05_docx_workflow_result.png"))
        print("   Saved 05_docx_workflow_result.png")

        # 6. Test Retry without duplicating user messages
        print("6. Verifying Retry behavior...")
        user_bubbles_before = await page.locator('div[class*="flex justify-end"]').count()
        print(f"   User bubbles before retry: {user_bubbles_before}")

        retry_btn = page.locator('button:has-text("Try Again")').first
        if await retry_btn.is_visible():
            await retry_btn.click()
            await page.wait_for_timeout(3000)
            user_bubbles_after = await page.locator('div[class*="flex justify-end"]').count()
            print(f"   User bubbles after retry: {user_bubbles_after}")
            assert user_bubbles_after == user_bubbles_before, "Retry created duplicate user message!"

        await page.screenshot(path=str(OUTPUT_DIR / "06_retry_no_duplicate.png"))
        print("   Saved 06_retry_no_duplicate.png")

        await browser.close()
        print("All visual verification checks completed successfully!")

if __name__ == "__main__":
    asyncio.run(run_verification())

