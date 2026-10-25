"""Tests for ASGI health endpoint exposed by pypath-shiny app."""

import asyncio
import json

from pypath import __version__

from pypath_shiny.app import app


async def _request(
    path: str,
    method: str = "GET",
    *,
    root_path: str = "",
    raw_path: bytes | None = None,
    headers: list[tuple[bytes, bytes]] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "root_path": root_path,
        "raw_path": raw_path if raw_path is not None else path.encode("ascii"),
        "query_string": b"",
        "headers": headers or [],
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
        "service": "pypath-ewe",
        "version": __version__,
    }


def test_health_endpoint_works_with_deployment_prefix():
    status, _headers, _body = asyncio.run(_request("/pypath/api/health"))
    assert status == 200


def test_health_endpoint_works_with_root_path_rewrite():
    status, _headers, _body = asyncio.run(
        _request("/api/health", root_path="/pypath")
    )
    assert status == 200


def test_health_endpoint_works_with_proxy_original_uri_header():
    status, _headers, _body = asyncio.run(
        _request(
            "/",
            headers=[(b"x-original-uri", b"/pypath/api/health")],
        )
    )
    assert status == 200
