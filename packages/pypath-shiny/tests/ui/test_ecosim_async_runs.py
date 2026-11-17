"""Headless drives of the Ecosim page's background runs (Review Focus 5).

They check what the page-level AST tests can only pin by reading the source:
the month progress, Cancel stopping the run at its next month so a new run
starts at once, a stale result dropped, the queued text, a closed tab's
queued run, and a model load clearing the old results and scenario. One more
drive checks the browser's empty "Quasi-steady above" field.

CI ignores ``tests/ui``. Run by hand (several minutes):

    pytest packages/pypath-shiny/tests/ui/test_ecosim_async_runs.py -v

Needs Playwright with Chromium (``python -m playwright install chromium``)
and Linux: a cancelled or retired run's thread stops at its next month
boundary, and the drives read the server's CPU time from ``/proc`` to tell
that it has stopped. Each test starts its own server on a free port, never
the live 3838, and stops it by PID: the run executor is process-wide, so a
run left computing by one test would hold a worker in the next.
"""

from __future__ import annotations

import logging
import os
import re
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

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not Path("/proc/self/stat").exists(), reason="reads CPU time from /proc"
    ),
]

APP = Path(pypath_shiny.__file__).resolve().parent / "app.py"
LIVE_PORT = 3838
YEARS = "50"  # long enough on Central Baltic for LSODA to be cancelled mid-run

RUNNING = "A simulation is already running. Cancel it first."
FINISHING = "The cancelled run is finishing its current month."
CANCELLED = "Cancelled: the run stops at the end of its current month"
QUEUED = "Queued: the server runs at most 2 simulations at once"
MODEL_CHANGED = "Model changed: create the scenario again before running."
NOT_RUN = "Simulation not yet run"
NO_RESULTS = "No simulation results available"
ACCEPTED = "Running simulation..."
MONTH = re.compile(r"Month (\d+) of (\d+) \((\d+)%\)")
# A new run is accepted within this many seconds of the Cancel click: the
# cancelled run stops at its next month boundary (a 50-year run takes about a
# minute to end on its own).
RESTART_LIMIT = 5.0


# --- the server ------------------------------------------------------------


def _free_port() -> int:
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        if port != LIVE_PORT:
            return port


@dataclass
class Server:
    url: str
    pid: int
    log: Path

    def cpu_seconds(self) -> float:
        """The server's user + system CPU time."""
        with open(f"/proc/{self.pid}/stat") as f:
            fields = f.read().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    def wait_idle(self, quiet: int = 3, share: float = 0.15, limit: float = 300):
        """Seconds until the server used under ``share`` of a core for ``quiet``
        seconds in a row, or None after ``limit`` seconds."""
        start = time.monotonic()
        calm = 0
        previous = self.cpu_seconds()
        while time.monotonic() - start < limit:
            time.sleep(1)
            now = self.cpu_seconds()
            calm = calm + 1 if now - previous < share else 0
            previous = now
            if calm >= quiet:
                return time.monotonic() - start
        return None

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


# --- page helpers ----------------------------------------------------------


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


def _progress_tab(page) -> None:
    page.click("a[data-value='Simulation Progress']")
    time.sleep(0.5)


def _status(page) -> str:
    return _text(page, "#simulation_status")


def _setup(page, url, method="lsoda") -> None:
    """Central Baltic, balanced; Ecosim: the balanced model, 50 years."""
    page.goto(url)
    page.wait_for_function(
        "() => window.Shiny && Shiny.shinyapp && Shiny.shinyapp.isConnected()",
        timeout=60000,
    )
    page.select_option("#example_model", "Central_Baltic")
    page.click("#btn_load_example_model")
    assert _wait_note(page, "opened", 60), _notes(page)
    page.wait_for_selector("#btn_balance", state="visible", timeout=30000)
    page.click("#btn_balance")
    assert _wait_note(page, "Model balanced successfully", 120), _notes(page)
    page.click("a[data-value='Ecosim']")
    page.wait_for_selector("#btn_create_scenario", state="visible")
    radio = page.locator("input[name='scenario_source'][value='model']")
    radio.wait_for(state="attached", timeout=20000)
    radio.check()
    page.fill("#sim_years", YEARS)
    page.select_option("#integration_method", method)
    time.sleep(0.5)


def _create_scenario(page) -> None:
    assert _click_for_note(
        page, "#btn_create_scenario", "Scenario created successfully", 60
    ), _notes(page)
    time.sleep(0.5)


def _start_run(page) -> None:
    """Run, and wait until a worker integrates it."""
    page.click("#btn_run_sim")
    page.wait_for_selector("#btn_cancel_sim", state="visible", timeout=30000)
    assert _wait_text(page, "#simulation_status", "Integrating", 30), _status(page)


def _month(page) -> tuple[int, int, int] | None:
    """The progress block's ``(month, n_months, percent)``, or None."""
    match = MONTH.search(_text(page, "#sim_month_progress"))
    return None if match is None else tuple(int(g) for g in match.groups())


def _bar_width(page) -> str:
    loc = page.locator("#sim_month_progress .progress-bar")
    return (loc.get_attribute("style") or "") if loc.count() else ""


def _run_accepted(page, since: float, limit: float) -> tuple[float, int]:
    """Click Run until it is accepted; (seconds since ``since``, clicks).

    Only a FINISHING refusal is retried (the stopped run has not reached its
    month boundary yet); fails after ``limit`` seconds from ``since``.
    """
    clicks = 0
    while time.monotonic() - since < limit:
        clicks += 1
        if _click_for_note(page, "#btn_run_sim", ACCEPTED, timeout=3):
            return time.monotonic() - since, clicks
        notes = _notes(page)
        assert any(FINISHING in n for n in notes), notes
        time.sleep(0.25)
    raise AssertionError(f"no run accepted within {limit} s ({clicks} clicks)")


def _results_page_text(page) -> str:
    page.click("a.nav-link[data-value='Results']")
    page.click("a[data-value='Simulation Results']")
    time.sleep(1.5)
    text = _text(page, "#sim_summary_status")
    page.click("a.nav-link[data-value='Ecosim']")
    return text


# --- the drives ------------------------------------------------------------


def test_empty_quasi_steady_rate_warns(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _setup(page, server.url, method="dop853")
    _create_scenario(page)
    _progress_tab(page)
    page.click("button.accordion-button:has-text('Advanced: changes the model')")
    page.locator("#solver_quasi_steady").check()
    page.wait_for_selector("#solver_qs_rate", state="visible")
    page.fill("#solver_qs_rate", "")
    time.sleep(1)  # the numeric input's debounce
    assert _click_for_note(
        page, "#btn_run_sim", "Quasi-steady rate must be a number above 0 (per year)."
    ), _notes(page)
    time.sleep(1)
    assert page.locator("#btn_cancel_sim").count() == 0  # nothing started
    assert NOT_RUN in _status(page)
    assert errors == []
    assert server.tracebacks() == 0


def test_cancel_stops_at_a_month_and_a_new_run_starts_at_once(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _setup(page, server.url)
    _create_scenario(page)
    _progress_tab(page)

    _start_run(page)
    time.sleep(0.3)  # past the one Starting -> Integrating rebuild
    cancel = page.locator("#btn_cancel_sim").element_handle()
    ticks, months = [], []
    for _ in range(5):
        ticks.append(_text(page, "#sim_elapsed"))
        months.append(_month(page))
        time.sleep(1.2)
    assert all(t.startswith("Elapsed:") for t in ticks), ticks
    assert len(set(ticks)) >= 3, ticks  # it ticks while the run computes
    # The months done appear and grow, out of 50 years' 600.
    seen = [m for m in months if m is not None]
    assert len(seen) >= 3, months
    assert all(total == 600 for _, total, _ in seen), seen
    done = [m for m, _, _ in seen]
    assert done == sorted(done) and done[-1] > done[0], seen
    assert all(pct == round(100 * m / 600) for m, _, pct in seen), seen
    last = _month(page)
    assert last is not None and _bar_width(page) == f"width: {last[2]}%"
    assert cancel.evaluate("e => e.isConnected")  # not rebuilt every second
    assert page.locator("#btn_cancel_sim").is_enabled()

    assert _click_for_note(page, "#btn_run_sim", RUNNING), _notes(page)
    assert "Integrating" in _status(page)

    cancelled_at = time.monotonic()
    assert _click_for_note(page, "#btn_cancel_sim", CANCELLED), _notes(page)
    assert "Simulation was cancelled" in _status(page)
    page.select_option("#integration_method", "RK4")
    time.sleep(0.5)
    # The cancelled run stops at its next month boundary and frees the
    # session's slot: a new run is accepted within seconds, long before the
    # cancelled one would have ended.
    restart, clicks = _run_accepted(page, cancelled_at, RESTART_LIMIT)
    print(
        f"months seen {done}; new run accepted {restart:.2f} s after the "
        f"Cancel click ({clicks} click(s))"
    )
    assert "Simulation failed" not in _status(page)

    # The RK4 run lands and stays; the cancelled LSODA result never landed.
    assert _wait_text(page, "#simulation_status", "completed successfully", 180)
    details = _text(page, "#progress_display")
    assert "Runge-Kutta 4" in details
    time.sleep(3)
    assert _text(page, "#progress_display") == details
    assert _text(page, "#solver_diagnostics") == ""
    assert errors == []
    assert server.tracebacks() == 0


def test_create_scenario_mid_run_retires_it(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _setup(page, server.url)
    _create_scenario(page)
    _progress_tab(page)

    _start_run(page)
    time.sleep(3)
    _create_scenario(page)
    _progress_tab(page)
    assert NOT_RUN in _status(page)
    assert _text(page, "#progress_display") == ""

    # The retired run stops at its next month boundary: the server goes idle
    # long before the 50-year run would have ended.
    idle = server.wait_idle(limit=60)
    assert idle is not None and idle < 10, idle
    assert NOT_RUN in _status(page)
    assert _text(page, "#progress_display") == ""
    assert _text(page, "#solver_diagnostics") == ""

    # The slot is free: a new run is accepted at once (then cancelled).
    _run_accepted(page, time.monotonic(), RESTART_LIMIT)
    page.wait_for_selector("#btn_cancel_sim", state="visible", timeout=30000)
    assert _click_for_note(page, "#btn_cancel_sim", CANCELLED), _notes(page)
    idle = server.wait_idle(limit=60)
    assert idle is not None and idle < 10, idle
    assert errors == []
    assert server.tracebacks() == 0


def test_model_load_mid_run_clears_results_and_scenario(server, browser):
    errors = []
    page = _new_page(browser, errors)
    _setup(page, server.url, method="RK4")
    _create_scenario(page)
    _progress_tab(page)
    page.click("#btn_run_sim")
    assert _wait_text(page, "#simulation_status", "completed successfully", 180)
    assert "50 years" in _results_page_text(page)

    page.select_option("#integration_method", "lsoda")
    time.sleep(0.5)
    _progress_tab(page)
    _start_run(page)
    time.sleep(2)
    page.click("a.nav-link[data-value='Home']")
    page.select_option("#example_model", "Curonian_Lagoon")
    assert _click_for_note(page, "#btn_load_example_model", MODEL_CHANGED), _notes(page)

    assert NO_RESULTS in _results_page_text(page)
    _progress_tab(page)
    time.sleep(1.5)  # outputs of a hidden tab re-render when shown
    assert NOT_RUN in _status(page)
    page.click("a[data-value='Scenario Setup']")
    time.sleep(1.5)
    assert "No scenario created" in _text(page, "#scenario_status")
    _progress_tab(page)
    assert _click_for_note(page, "#btn_run_sim", "Create a scenario first")

    assert server.wait_idle() is not None, "the retired run never finished"
    assert NO_RESULTS in _results_page_text(page)
    _progress_tab(page)
    time.sleep(1.5)
    assert NOT_RUN in _status(page)
    assert _text(page, "#solver_diagnostics") == ""
    assert errors == []
    assert server.tracebacks() == 0


def test_third_run_queues_and_a_closed_tab_frees_its_place(server, browser):
    errors = []
    tabs = []
    for _ in range(3):
        page = _new_page(browser, errors)
        _setup(page, server.url)
        _create_scenario(page)
        _progress_tab(page)
        tabs.append(page)
    a, b, c = tabs

    _start_run(a)
    _start_run(b)
    c.click("#btn_run_sim")
    assert _wait_text(c, "#sim_elapsed", QUEUED, 15), _text(c, "#sim_elapsed")
    assert "Starting LSODA, 50 years" in _status(c)
    assert "%" not in _status(c)
    assert "Integrating" in _status(a) and "Integrating" in _status(b)

    # Cancelled while queued, the run never started: the slot frees at once.
    assert _click_for_note(c, "#btn_cancel_sim", CANCELLED), _notes(c)
    _clear_notes(c)
    c.click("#btn_run_sim")
    assert _wait_text(c, "#sim_elapsed", QUEUED, 15), _text(c, "#sim_elapsed")
    assert not any(FINISHING in n for n in _notes(c))

    # Closing the tab cancels its queued run: no ownerless run starts later.
    c.context.close()
    for page in (a, b):
        assert _wait_text(page, "#simulation_status", "completed successfully", 300)
    idle = server.wait_idle(quiet=3)
    # A third 50-year LSODA run would keep the server busy for far longer.
    assert idle is not None and idle < 10, idle
    assert errors == []
    assert server.tracebacks() == 0
