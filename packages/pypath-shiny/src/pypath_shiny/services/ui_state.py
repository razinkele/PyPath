"""Serialization helpers for persisted UI state (wizard drafts)."""

from __future__ import annotations

import json
from typing import Any

WIZARD_DRAFT_VERSION = 1


def serialize_wizard_draft(payload: dict[str, Any]) -> str:
    """Serialize wizard draft payload to JSON with schema version."""
    body = dict(payload)
    body["draft_version"] = WIZARD_DRAFT_VERSION
    return json.dumps(body, ensure_ascii=True, sort_keys=True)


def deserialize_wizard_draft(data: str) -> dict[str, Any]:
    """Deserialize wizard draft JSON and ensure version marker exists."""
    body = json.loads(data)
    if not isinstance(body, dict):
        raise ValueError("Wizard draft must decode to a JSON object")
    if "draft_version" not in body:
        body["draft_version"] = WIZARD_DRAFT_VERSION
    return body
