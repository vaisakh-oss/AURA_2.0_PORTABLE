import asyncio
import os
import sys
from pathlib import Path
from playwright.async_api import async_playwright

OUTPUT_DIR = Path("/Users/pratikskanoj/.gemini/antigravity/brain/cfedb383-8535-45c9-896d-f9cdd94b6a90/screenshots_v2")
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

        # 1. New Chat (1440x900)
        print("1. Loading New Chat...")
        await page.goto(APP_URL, wait_until="networkidle")
        await page.wait_for_timeout(1000)
        # Click "+ New Chat" in sidebar to ensure clean empty welcome state
        new_chat_btn = page.locator('button:has-text("New Chat")').first
        if await new_chat_btn.is_visible():
            await new_chat_btn.click()
            await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "01_desktop_new_chat.png"))
        print("   Saved 01_desktop_new_chat.png")

        # 2. Chat with selected attachment (DOCX)
        print("2. Verifying DOCX attachment...")
        # Create small test docx
        docx_path = Path("/tmp/quarterly_audit.docx")
        try:
            import docx
            doc = docx.Document()
            doc.add_heading("AURA System Security Audit", level=1)
            doc.add_paragraph("All core subsystems are operating within nominal latency thresholds.")
            doc.save(str(docx_path))
        except Exception:
            docx_path.write_bytes(b"PK\x03\x04" + b"\x00" * 300)

        # Upload DOCX
        file_input = page.locator('input[type="file"]').first
        await file_input.set_input_files(str(docx_path))
        await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "02_desktop_chat_docx_attachment.png"))
        print("   Saved 02_desktop_chat_docx_attachment.png")

        # 3. Active Chat Conversation
        print("3. Loading existing active chat...")
        # Click first conversation in history if present
        conv_items = page.locator('aside div[class*="group flex items-center"]')
        if await conv_items.count() > 0:
            await conv_items.first.click()
            await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "03_desktop_active_chat.png"))
        print("   Saved 03_desktop_active_chat.png")

        # 4. Agent Activity & Inspector Sources
        print("4. Inspecting Agent Activity & Sources...")
        # Click Inspector button if closed
        inspector_btn = page.locator('button:has-text("Inspector")').first
        if await inspector_btn.is_visible():
            # If inspector panel is open, click Sources tab
            sources_tab = page.locator('button:has-text("Sources")').first
            if await sources_tab.is_visible():
                await sources_tab.click()
                await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "04_desktop_sources_panel.png"))
        print("   Saved 04_desktop_sources_panel.png")

        # Switch back to Activity tab in Inspector
        activity_tab = page.locator('button:has-text("Activity")').first
        if await activity_tab.is_visible():
            await activity_tab.click()
            await page.wait_for_timeout(500)
        await page.screenshot(path=str(OUTPUT_DIR / "05_desktop_agent_activity.png"))
        print("   Saved 05_desktop_agent_activity.png")

        # 5. Knowledge Base View
        print("5. Loading Knowledge Base View...")
        kb_nav = page.locator('button:has-text("Knowledge Base")').first
        await kb_nav.click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "06_desktop_knowledge_base.png"))
        print("   Saved 06_desktop_knowledge_base.png")

        # 6. Memory View
        print("6. Loading Memory View...")
        mem_nav = page.locator('button:has-text("Memory")').first
        await mem_nav.click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "07_desktop_memory.png"))
        print("   Saved 07_desktop_memory.png")

        # 7. Analytics View
        print("7. Loading Analytics View...")
        analytics_nav = page.locator('button:has-text("Analytics")').first
        await analytics_nav.click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "08_desktop_analytics.png"))
        print("   Saved 08_desktop_analytics.png")

        # 8. System Health View
        print("8. Loading System Health View...")
        health_nav = page.locator('button:has-text("System Health")').first
        await health_nav.click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "09_desktop_system_health.png"))
        print("   Saved 09_desktop_system_health.png")

        # 9. Settings View
        print("9. Loading Settings View...")
        settings_nav = page.locator('button:has-text("Settings")').first
        await settings_nav.click()
        await page.wait_for_timeout(1000)
        await page.screenshot(path=str(OUTPUT_DIR / "10_desktop_settings.png"))
        print("   Saved 10_desktop_settings.png")

        # 10. Responsive Laptop View (1280x800)
        print("10. Testing Laptop View (1280x800)...")
        await page.set_viewport_size({"width": 1280, "height": 800})
        chat_nav = page.locator('button:has-text("Chat")').first
        await chat_nav.click()
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUTPUT_DIR / "11_laptop_chat.png"))
        print("   Saved 11_laptop_chat.png")

        # 11. Responsive Tablet Landscape (1024x768)
        print("11. Testing Tablet Landscape (1024x768)...")
        await page.set_viewport_size({"width": 1024, "height": 768})
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUTPUT_DIR / "12_tablet_landscape.png"))
        print("   Saved 12_tablet_landscape.png")

        # 12. Responsive Tablet Portrait (768x1024)
        print("12. Testing Tablet Portrait (768x1024)...")
        await page.set_viewport_size({"width": 768, "height": 1024})
        await page.wait_for_timeout(800)
        await page.screenshot(path=str(OUTPUT_DIR / "13_tablet_portrait.png"))
        print("   Saved 13_tablet_portrait.png")

        await browser.close()
        print("All visual verification screenshots captured successfully!")

if __name__ == "__main__":
    asyncio.run(run_verification())

