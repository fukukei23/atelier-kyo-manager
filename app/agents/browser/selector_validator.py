"""Selector response parsing and normalization utilities."""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json_from_text(text: str) -> dict[str, Any]:
    """LLM の応答テキストから JSON を抽出する（フォールバック）."""
    json_match = re.search(r"\{[\s\S]*\}", text)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    return {"candidates": []}


def normalize_proposal(
    proposal: dict[str, Any],
    site: str,
    page_type: str,
) -> dict[str, Any]:
    """LLM の応答を仕様書に準拠した形式に正規化する."""
    if "site" in proposal and "page_type" in proposal and "candidates" in proposal:
        return proposal

    normalized: dict[str, Any] = {
        "site": site,
        "page_type": page_type,
        "strategy": "llm_selector_healing_v1",
        "candidates": [],
    }

    if "proposed_selectors" in proposal:
        for idx, sel in enumerate(proposal.get("proposed_selectors", [])):
            normalized["candidates"].append(
                {
                    "target": f"field_{idx}",
                    "old_selector": "",
                    "new_selector": sel,
                    "confidence": 0.8,
                    "reason": proposal.get("rationale", "LLM proposal"),
                }
            )

    if "candidates" in proposal:
        normalized["candidates"] = proposal["candidates"]

    return normalized
