import asyncio
from playwright.async_api import async_playwright


async def run_flow():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto("http://127.0.0.1:8000")
        # Click nav to Pre-Balance Diagnostics
        await page.click("text=Pre-Balance Diagnostics")
        # Wait for page to load
        await page.wait_for_selector("text=Pre-Balance Diagnostics")
        # Click the info button (btn_rpath_diag_info)
        await page.click("#btn_rpath_diag_info")
        # Wait for modal dialog
        await page.wait_for_selector("text=Rpath Diagnostics")
        content = await page.inner_text(".modal-body")
        print("MODAL_CONTENT:", content[:400])
        # Click Run Diagnostics to exercise report
        await page.click("#btn_run_diagnostics")
        # Wait for notification or report summary
        await page.wait_for_selector("text=Diagnostics complete", timeout=10000)
        await browser.close()


if __name__ == "__main__":
    asyncio.run(run_flow())
