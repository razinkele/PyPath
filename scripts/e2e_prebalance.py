#!/usr/bin/env python3
"""End-to-end smoke script for Pre-Balance Diagnostics UI using Playwright.

Behavior:
- Attempts to import Playwright; installs it if missing and runs browser install.
- Opens the app at http://127.0.0.1:8000 and navigates to "Pre-Balance Diagnostics".
- Clicks the small info button next to the Rpath badge and verifies the modal appears and contains "Verification output".
- Clicks "Run Diagnostics" and waits for the summary report to render (checks for "Biomass Diagnostics").

Exit codes: 0 = success, non-zero = failure.
"""
import sys
import time
from pathlib import Path

APP_URL = "http://127.0.0.1:8000"

try:
    from playwright.sync_api import sync_playwright
except Exception:
    # Try installing playwright and browser binaries
    import subprocess

    print("Playwright not found, installing...", file=sys.stderr)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "playwright"]) 
    # Install browsers
    subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"] )
    from playwright.sync_api import sync_playwright


import subprocess
import os

def _start_server():
    """Start the Shiny app server as a subprocess and wait until it's ready."""
    cmd = [sys.executable, "-m", "shiny", "run", "app/app.py", "--port", "8000"]
    env = os.environ.copy()
    # Use unbuffered output to read lines promptly
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # Wait for startup line
    for _ in range(60):
        line = proc.stdout.readline()
        if not line:
            time.sleep(0.1)
            continue
        print("[server]", line.strip())
        if "Application startup complete" in line:
            return proc
    # timed out
    proc.terminate()
    raise RuntimeError("Server failed to start in time")


def main():
    # Start server
    server_proc = _start_server()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context()
            page = context.new_page()
            print(f"Opening {APP_URL}")
            page.goto(APP_URL, wait_until="domcontentloaded")

            # Navigate to Pre-Balance Diagnostics
            # Navbar link has text 'Pre-Balance Diagnostics'
            try:
                page.get_by_role("link", name="Pre-Balance Diagnostics").click()
            except Exception:
                # fallback: find nav panel link
                page.locator("nav").get_by_text("Pre-Balance Diagnostics").click()

            # Wait for badge and info button
            info_btn = page.locator("#btn_rpath_diag_info")
            info_btn.wait_for(timeout=5000)
            print("Info button found; clicking to open modal")
            info_btn.click()

            # Modal may take some time while verification runs. Wait for either the title
            # or the verification output text to appear (up to 20s).
            try:
                page.wait_for_selector("text=Rpath Diagnostics", timeout=20000)
                print("Modal opened (title found)")
            except Exception:
                # Fallback: wait for Verification output text
                try:
                    page.wait_for_selector("text=Verification output", timeout=20000)
                    print("Modal opened (verification output found)")
                except Exception as e:
                    # Debugging: dump page content and server logs
                    html = page.content()
                    print("--- PAGE HTML (truncated) ---")
                    print(html[:2000])
                    print("--- END PAGE HTML ---")
                    # Attempt to read remaining server output
                    try:
                        rest = server_proc.stdout.read()
                        print("--- SERVER LOG ---")
                        print(rest)
                    except Exception:
                        pass
                    raise e

            # Check modal contains 'Verification output' text
            assert page.locator("text=Verification output").count() > 0, "Modal missing verification output"
            print("Modal contains verification output")

            # Close modal
            # click "Close" button if present, otherwise press Escape
            try:
                page.get_by_role("button", name="Close").click()
            except Exception:
                page.keyboard.press("Escape")

            # Click Run Diagnostics button
            print("Clicking Run Diagnostics")
            page.get_by_role("button", name="Run Diagnostics").click()

            # Wait for report summary to render (look for Biomass Diagnostics)
            page.wait_for_selector("text=Biomass Diagnostics", timeout=15000)
            print("Found Biomass Diagnostics in report summary")

            # Done
            browser.close()
            return 0
    finally:
        server_proc.terminate()
        server_proc.wait(timeout=5)



if __name__ == "__main__":
    try:
        rc = main()
        print("E2E flow succeeded")
        sys.exit(rc)
    except AssertionError as e:
        print("Assertion failed:", e, file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print("Error during E2E flow:", e, file=sys.stderr)
        sys.exit(1)
