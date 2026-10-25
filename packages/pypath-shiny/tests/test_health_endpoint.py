"""Tests for ASGI health endpoint exposed by pypath-shiny app."""

import asyncio
import json

from pypath_shiny import __version__
from pypath_shiny.app import app


async def _request(path: str, method: str = "GET") -> tuple[int, dict[str, str], bytes]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode("ascii"),
        "query_string": b"",
        "headers": [],
        "client": ("127.0.0.1", 12345),
        "server": ("testserver", 80),
    }

    sent = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)

    start = next(m for m in sent if m["type"] == "http.response.start")
    body_messages = [m for m in sent if m["type"] == "http.response.body"]
    body = b"".join(m.get("body", b"") for m in body_messages)
    headers = {
        k.decode("latin1"): v.decode("latin1") for k, v in start.get("headers", [])
    }
    return start["status"], headers, body


def test_health_endpoint_returns_ok_json():
    status, headers, body = asyncio.run(_request("/api/health"))

    assert status == 200
    assert headers["content-type"].startswith("application/json")
    payload = json.loads(body.decode("utf-8"))
    assert payload == {
        "status": "ok",
        "service": "pypath-shiny",
        "version": __version__,
    }


def test_health_endpoint_works_with_deployment_prefix():
    status, _headers, _body = asyncio.run(_request("/pypath/api/health"))
    assert status == 200
