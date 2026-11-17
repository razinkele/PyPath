"""Headless drives of the Ecosim page's prey-switching control.

They check what the page-level tests can only pin by reading the source: an
edit reaches the store without re-rendering the box, an out-of-range value is
refused and the box resets, Create then Run shows the "Prey switching" line
on the progress and Summary tabs, a re-balance keeps the settings and a load
clears them, and the EwE example route (create message, imported values,
edit, revert).

CI ignores ``tests/ui``. Run by hand (a few minutes):

    pytest packages/pypath-shiny/tests/ui/test_prey_switching_ui_drive.py -q

Needs Playwright with Chromium (``python -m playwright install chromium``).
Each test starts its own server on a free port, never the live 3838 or 8140,
and stops it by PID.
"""

from __future__ import annotations

import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import pypath  # noqa: E402

import pypath_shiny  # noqa: E402

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.slow]

APP = Path(pypath_shiny.__file__).resolve().parent / "app.py"
RESERVED_PORTS = (3838, 8140)
YEARS = "2"

CREATED = "Scenario created successfully"
CREATE_MESSAGE = "Create the scenario to see and change its switching powers."
CHANGED = "Changed values apply when you create the scenario again."
NOT_YET = "Values apply when you create the scenario."
KEPT = "Prey switching settings kept for 1 group(s) after re-balancing"
CLEARED = "Prey switching settings cleared (1 group(s))"
RANGE = "must be between 0 and 2"
BOXES = "#switching_control input[type=number]"


# --- the server ------------------------------------------------------------


def _free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port not in RESERVED_PORTS:
            return port


@dataclass
class Server:
    url: str
    pid: int
    log: Path

    def tracebacks(self) -> int:
        return self.log.read_text(encoding="utf-8", errors="replace").count("Traceback")


def _wait_http(url: str, proc: subprocess.Popen, timeout: float) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if proc.poll() is not None:
            raise RuntimeError(f"the server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except OSError:
            time.sleep(0.5)
    raise RuntimeError(f"no server at {url} after {timeout} s")


@pytest.fixture
def server(tmp_path):
    port = _free_port()
    log_path = tmp_path / "server.log"
    # The server imports the same packages as these tests.
    roots = [str(Path(m.__file__).resolve().parents[1]) for m in (pypath, pypath_shiny)]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        roots + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    cmd = [sys.executable, "-m", "shiny", "run", str(APP)]
    cmd += ["--host", "127.0.0.1", "--port", str(port)]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
        try:
            url = f"http://127.0.0.1:{port}"
            _wait_http(url, proc, 90)
            yield Server(url, proc.pid, log_path)
        finally:
            proc.terminate()  # by PID, never by name
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()


@pytest.fixture
def browser():
    with sync_api.sync_playwright() as pw:
        try:
            chromium = pw.chromium.launch(headless=True)
        except Exception as e:  # no Chromium installed
            pytest.skip(f"Chromium cannot launch: {e}")
        try:
            yield chromium
        finally:
            chromium.close()


def _new_page(browser, errors):
    page = browser.new_context(viewport={"width": 1400, "height": 1000}).new_page()
    page.on("pageerror", lambda e: errors.append(str(e)))
    return page


def _text(page, selector) -> str:
    loc = page.locator(selector)
    return loc.inner_text() if loc.count() else ""


def _notes(page) -> list[str]:
    return page.locator(".shiny-notification-content-text").all_inner_texts()


def _clear_notes(page) -> None:
    page.evaluate(
        "() => document.querySelectorAll('.shiny-notification')"
        ".forEach(n => n.remove())"
    )


def _wait_note(page, text, timeout=15) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if any(text in n for n in _notes(page)):
            return True
        time.sleep(0.2)
    return False


def _click_for_note(page, selector, text, timeout=15) -> bool:
    """Click with no notification showing; True once ``text`` is notified."""
    _clear_notes(page)
    page.click(selector)
    return _wait_note(page, text, timeout)


def _wait_text(page, selector, text, timeout) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if text in _text(page, selector):
            return True
        time.sleep(0.25)
    return False


def _boxes(page):
    return page.locator(BOXES)


def _wait_boxes(page) -> None:
    page.wait_for_function(
        f"() => document.querySelectorAll('{BOXES}').length > 0", timeout=20000
    )


def _label(page, box) -> str:
    return page.locator(f"label[for='{box.get_attribute('id')}']").inner_text().strip()


def _nav(page, value) -> None:
    page.click(f"a[data-value='{value}']")


def _connect(page, url) -> None:
    page.goto(url)
    page.wait_for_function(
        "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()",
        timeout=60000,
    )


def _wait_notes(page, texts, timeout=15) -> list[str]:
    """Poll until every text has been notified; return all notes seen.

    Notifications can come and go while another is awaited, so each poll's
    notes are recorded, not just the last ones.
    """
    seen: list[str] = []
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        for n in _notes(page):
            if n not in seen:
                seen.append(n)
        if all(any(t in n for n in seen) for t in texts):
            return seen
        time.sleep(0.2)
    raise AssertionError(f"not all of {texts} notified; seen {seen}")


def _wait_box(page, index, value, timeout=15) -> None:
    """Wait until box ``index`` shows ``value``."""
    end = time.monotonic() + timeout
    shown = None
    while time.monotonic() < end:
        box = _boxes(page).nth(index)
        shown = box.input_value() if box.count() else None
        if shown not in (None, "") and float(shown) == value:
            return
        time.sleep(0.2)
    raise AssertionError(f"box {index} shows {shown!r}, not {value}")


def _load_example(page, name, *also) -> list[str]:
    """Open an example; the notes seen, once ``also`` texts were notified too."""
    _nav(page, "Home")
    page.select_option("#example_model", name)
    _clear_notes(page)
    page.click("#btn_load_example_model")
    return _wait_notes(page, ["opened", *also], 60)


def _balance(page, *also) -> list[str]:
    """Balance; the notes seen, once ``also`` texts were notified too."""
    page.wait_for_selector("#btn_balance", state="visible", timeout=30000)
    _clear_notes(page)
    page.click("#btn_balance")
    return _wait_notes(page, ["Model balanced successfully", *also], 120)


def _to_ecosim(page) -> None:
    _nav(page, "Ecosim")
    page.wait_for_selector("#btn_create_scenario", state="visible")


def _to_model_route(page) -> None:
    """Select the balanced route and wait for its boxes to settle.

    The source radio is re-rendered when a model opens or balances and can
    fall back to its default after being checked, so the choice is repeated
    until the boxes have been there for two polls in a row.
    """
    stable = 0
    end = time.monotonic() + 30
    while time.monotonic() < end and stable < 2:
        radio = page.locator("input[name='scenario_source'][value='model']")
        if radio.count() and not radio.is_checked():
            radio.check()
            stable = 0
        stable = stable + 1 if _boxes(page).count() else 0
        time.sleep(0.4)
    assert stable >= 2, "the balanced route's boxes did not settle"


def _create(page) -> None:
    _clear_notes(page)
    page.click("#btn_create_scenario")
    assert _wait_note(page, CREATED, 60), _notes(page)


def _edit(box, value) -> None:
    box.fill(value)
    box.press("Tab")


def _progress_tab(page) -> None:
    page.locator("a.nav-link", has_text="Simulation Progress").first.click()


def _run(page) -> bool:
    """Run the scenario; True once the progress tab shows the switching line."""
    page.click("#btn_run_sim")
    _progress_tab(page)
    return _wait_text(page, "#switching_line", "Prey switching", 90)


def _summary_line(page) -> str:
    page.locator("a[data-value='Summary']:visible").first.click()
    _wait_text(page, "#switching_line_summary", "Prey switching", 15)
    return _text(page, "#switching_line_summary").strip()


def _pending(page) -> str:
    return _text(page, "#switching_pending_note").strip()


# --- the drives ------------------------------------------------------------


def test_balanced_route_edit_refuse_create_run(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _connect(page, server.url)
    _load_example(page, "Central_Baltic")
    _balance(page)
    _to_ecosim(page)
    _to_model_route(page)
    assert _boxes(page).count() > 5

    first = _boxes(page).nth(0)
    first_id = first.get_attribute("id")
    name = _label(page, first)
    assert float(first.input_value()) == 0
    _edit(first, "0.5")
    assert _wait_text(page, "#switching_pending_note", NOT_YET, 10), _pending(page)
    assert first.get_attribute("id") == first_id  # the edit did not re-render

    second = _boxes(page).nth(1)
    _clear_notes(page)
    second.fill("3")
    second.press("Tab")
    assert _wait_note(page, RANGE), _notes(page)
    _wait_box(page, 1, 0)  # refused: the box resets

    page.fill("#sim_years", YEARS)
    _create(page)
    page.wait_for_function(  # Create re-renders the boxes with fresh ids
        f"() => {{ const b = document.querySelector('{BOXES}');"
        f" return b && b.id !== '{first_id}'; }}",
        timeout=15000,
    )
    _wait_box(page, 0, 0.5)

    assert _run(page), _text(page, "#switching_line")
    assert _text(page, "#switching_line").strip() == f"Prey switching: {name} 0.5"
    assert _summary_line(page) == f"Prey switching: {name} 0.5"
    assert errors == []
    assert server.tracebacks() == 0


def test_rebalance_keeps_and_a_load_clears(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _connect(page, server.url)
    _load_example(page, "Central_Baltic")
    _balance(page)
    _to_ecosim(page)
    _to_model_route(page)
    _edit(_boxes(page).nth(0), "0.5")
    assert _wait_text(page, "#switching_pending_note", NOT_YET, 10), _pending(page)

    _nav(page, "Ecopath Model")
    _balance(page, KEPT)
    _to_ecosim(page)
    _to_model_route(page)
    _wait_box(page, 0, 0.5)

    # The notice may arrive before or after "opened": both are awaited.
    _load_example(page, "Curonian_Lagoon", CLEARED)
    _balance(page)
    _to_ecosim(page)
    _to_model_route(page)
    # Read all boxes in one go and poll: the old model's boxes may still be
    # there for a moment after the load.
    values: list[str] = []
    end = time.monotonic() + 15
    while time.monotonic() < end:
        values = page.eval_on_selector_all(BOXES, "els => els.map(e => e.value)")
        if values and all(v != "" and float(v) == 0 for v in values):
            break
        time.sleep(0.2)
    assert values and all(float(v) == 0 for v in values), values
    assert errors == []
    assert server.tracebacks() == 0


def test_ewe_example_route(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _connect(page, server.url)
    _load_example(page, "Central_Baltic")
    _balance(page)
    _to_ecosim(page)
    page.locator("input[name='scenario_source'][value='ewe']").wait_for(
        state="attached", timeout=20000
    )
    assert page.locator("input[name='scenario_source'][value='ewe']").is_checked()
    assert _wait_text(page, "#switching_control", CREATE_MESSAGE, 15)

    page.fill("#sim_years", YEARS)
    _create(page)
    _wait_boxes(page)
    values = [_boxes(page).nth(i).input_value() for i in range(_boxes(page).count())]
    assert len(values) > 5
    source_value = float(values[1])  # the example's own value for this group

    box = _boxes(page).nth(1)
    name = _label(page, box)
    _edit(box, "1.5")
    assert _wait_text(page, "#switching_pending_note", CHANGED, 10), _pending(page)
    _create(page)
    assert _run(page), _text(page, "#switching_line")
    assert _text(page, "#switching_line").strip() == f"Prey switching: {name} 1.5"

    # Back to the source value: the pending note appears, and after Create the
    # scenario is the source's own again, with nothing pending.
    box = _boxes(page).nth(1)
    _edit(box, f"{source_value:g}")
    assert _wait_text(page, "#switching_pending_note", CHANGED, 10), _pending(page)
    _create(page)
    _wait_box(page, 1, source_value)
    end = time.monotonic() + 10
    while _pending(page) and time.monotonic() < end:
        time.sleep(0.2)
    assert _pending(page) == ""
    assert errors == []
    assert server.tracebacks() == 0
