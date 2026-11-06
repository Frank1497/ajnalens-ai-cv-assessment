"""
The two tools the LLM (Gemini) can call via native function calling.

- get_equipment_status(equipment_id): mock lookup table.
- log_inspection(equipment_id, note, severity): appends a timestamped record
  to task4/outputs/inspections.json.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

OUTPUTS_DIR = Path(__file__).resolve().parent.parent / "outputs"
INSPECTIONS_PATH = OUTPUTS_DIR / "inspections.json"

# Mock equipment status table -- deliberately varied so different equipment
# IDs produce meaningfully different answers / tool-use decisions.
_MOCK_EQUIPMENT_DB: Dict[str, Dict[str, Any]] = {
    "P-101": {
        "status": "running",
        "last_service_date": "2025-08-12",
        "active_alarms": [],
    },
    "P-102": {
        "status": "stopped",
        "last_service_date": "2025-03-02",
        "active_alarms": ["HIGH_VIBRATION", "SEAL_LEAK_WARNING"],
    },
    "P-103": {
        "status": "running",
        "last_service_date": "2025-09-28",
        "active_alarms": ["LOW_SUCTION_PRESSURE"],
    },
}


def get_equipment_status(equipment_id: str) -> Dict[str, Any]:
    """Return mock status JSON for the given equipment_id.

    Unknown IDs return a clear 'not found' status rather than fabricating data.
    """
    eq = equipment_id.strip().upper()
    record = _MOCK_EQUIPMENT_DB.get(eq)
    if record is None:
        return {
            "equipment_id": eq,
            "status": "unknown",
            "last_service_date": None,
            "active_alarms": [],
            "note": f"No record found for equipment_id '{eq}' in the mock equipment database.",
        }
    return {"equipment_id": eq, **record}


def log_inspection(equipment_id: str, note: str, severity: str) -> Dict[str, Any]:
    """Append a timestamped inspection record to outputs/inspections.json."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    if INSPECTIONS_PATH.exists():
        try:
            with open(INSPECTIONS_PATH, "r") as f:
                records = json.load(f)
            if not isinstance(records, list):
                records = []
        except (json.JSONDecodeError, OSError):
            records = []
    else:
        records = []

    entry = {
        "equipment_id": equipment_id.strip().upper(),
        "note": note,
        "severity": severity,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    records.append(entry)

    with open(INSPECTIONS_PATH, "w") as f:
        json.dump(records, f, indent=2)

    return {"logged": True, "entry": entry}


# ---------------------------------------------------------------------------
# Gemini function-calling declarations (JSON schema, Gemini's native
# function-calling format -- the equivalent of Anthropic tool_use blocks).
# ---------------------------------------------------------------------------
TOOL_DECLARATIONS = [
    {
        "name": "get_equipment_status",
        "description": (
            "Get the current mock operational status of a piece of equipment "
            "by its ID, including whether it is running/stopped, when it was "
            "last serviced, and any active alarms. Use this whenever a "
            "question asks about the current state, safety, or readiness of "
            "a specific equipment ID (e.g. 'Is pump P-102 safe to restart?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_id": {
                    "type": "string",
                    "description": "Equipment identifier, e.g. 'P-101', 'P-102', 'P-103'.",
                }
            },
            "required": ["equipment_id"],
        },
    },
    {
        "name": "log_inspection",
        "description": (
            "Log/record an inspection note for a piece of equipment, with a "
            "severity level. Use this when the user explicitly asks to log, "
            "record, or note down an inspection finding or observation."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "equipment_id": {
                    "type": "string",
                    "description": "Equipment identifier, e.g. 'P-101', 'P-102', 'P-103'.",
                },
                "note": {
                    "type": "string",
                    "description": "Free-text inspection note/finding.",
                },
                "severity": {
                    "type": "string",
                    "description": "Severity level, e.g. 'low', 'medium', 'high', 'critical'.",
                },
            },
            "required": ["equipment_id", "note", "severity"],
        },
    },
]

TOOL_IMPLS = {
    "get_equipment_status": get_equipment_status,
    "log_inspection": log_inspection,
}
