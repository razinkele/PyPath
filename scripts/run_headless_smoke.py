from playwright.sync_api import sync_playwright
import sys
import time
import os
import urllib.request
from datetime import datetime

URL = "http://127.0.0.1:8000"
POLL_TIMEOUT = int(os.environ.get("SMOKE_POLL_TIMEOUT", "20"))
POLL_INTERVAL = float(os.environ.get("SMOKE_POLL_INTERVAL", "0.5"))
HEADLESS = os.environ.get("SMOKE_HEADLESS", "1") != "0"


def now():
    return datetime.utcnow().isoformat() + "Z"


def wait_for_server(url, timeout=POLL_TIMEOUT, interval=POLL_INTERVAL):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(url, timeout=2)
            if r.getcode() == 200:
                print(now(), "SERVER_UP")
                return True
        except Exception as e:
            print(now(), "server not ready:", e)
            time.sleep(interval)
    return False


try:
    print(now(), 'SMOKE: starting')

    if not wait_for_server(URL, timeout=POLL_TIMEOUT):
        print(now(), 'SMOKE_FAIL', 'server did not become ready within timeout')
        sys.exit(2)

    print(now(), 'SMOKE: launching Playwright')
    with sync_playwright() as p:
        print(now(), 'SMOKE: launching browser')
        browser = p.chromium.launch(headless=HEADLESS)
        # create a context so we can optionally record networks if needed later
        context = browser.new_context()
        page = context.new_page()

        try:
            print(now(), 'SMOKE: navigating to', URL)
            page.goto(URL, timeout=20000)

            print(now(), 'SMOKE: waiting for Pre-Balance tab')
            page.wait_for_selector("text=Pre-Balance Diagnostics", timeout=10000)
            print(now(), 'SMOKE: clicking Pre-Balance nav')
            page.click("text=Pre-Balance Diagnostics")

            print(now(), 'SMOKE: waiting for info button')
            page.wait_for_selector("#btn_rpath_diag_info", timeout=10000)
            print(now(), 'SMOKE: clicking info button')
            page.click("#btn_rpath_diag_info")

            print(now(), 'SMOKE: waiting for verification output (up to 60s)')
            # Verification can invoke a subprocess; allow up to 60s for output
            # Wait for the verification text to be present in the DOM (attached), visibility isn't required
            page.wait_for_selector("text=Verification output", timeout=60000, state="attached")

            # Prefer inline fallback element when `session.show_modal` is not available
            try:
                modal_text = page.inner_text("#rpath_modal_inline")
            except Exception:
                modal_text = page.inner_text("div.modal")

            print(now(), 'SMOKE_OK')
            print(modal_text[:1600])

            context.close()
            browser.close()
            sys.exit(0)

        except Exception as e:
            ts = now().replace(':', '-')
            print(ts, 'SMOKE_FAIL', repr(e))
            # Save artifacts to help debugging
            try:
                screenshot_path = f"smoke_failure_{ts}.png"
                page.screenshot(path=screenshot_path)
                print(ts, 'screenshot saved to', screenshot_path)
            except Exception as s:
                print(ts, 'screenshot failed', repr(s))
            try:
                html_path = f"smoke_failure_{ts}.html"
                with open(html_path, 'w', encoding='utf-8') as fh:
                    fh.write(page.content())
                print(ts, 'page content saved to', html_path)
            except Exception as h:
                print(ts, 'saving page content failed', repr(h))
            try:
                # attempt to save console logs from the page if any were captured
                logs_path = f"smoke_console_{ts}.log"
                # Playwright doesn't provide direct console dump, but we can leave placeholder
                with open(logs_path, 'w', encoding='utf-8') as fh:
                    fh.write('Playwright: console logs not captured in this run')
                print(ts, 'console placeholder saved to', logs_path)
            except Exception as c:
                print(ts, 'saving console placeholder failed', repr(c))

            try:
                context.close()
            except Exception:
                pass
            try:
                browser.close()
            except Exception:
                pass
            sys.exit(2)
except Exception as e:
    print(now(), 'SMOKE_FAIL', repr(e))
    sys.exit(2)
