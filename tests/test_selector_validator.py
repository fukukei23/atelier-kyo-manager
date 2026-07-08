"""Tests for selector_validator module."""

from __future__ import annotations

from app.agents.browser.selector_validator import extract_json_from_text, normalize_proposal


class TestExtractJsonFromText:
    def test_extracts_valid_json_from_text(self):
        text = 'Here is the response: {"site": "example.com", "candidates": []} done.'
        result = extract_json_from_text(text)
        assert result == {"site": "example.com", "candidates": []}

    def test_extracts_nested_json(self):
        text = 'Result: {"a": {"b": 1}, "c": [1, 2]}'
        result = extract_json_from_text(text)
        assert result["a"]["b"] == 1
        assert result["c"] == [1, 2]

    def test_returns_empty_candidates_on_no_json(self):
        text = "No JSON here at all"
        result = extract_json_from_text(text)
        assert result == {"candidates": []}

    def test_returns_empty_candidates_on_invalid_json(self):
        text = "Here is: {invalid json content"
        result = extract_json_from_text(text)
        assert result == {"candidates": []}

    def test_greedy_regex_captures_outermost_block(self):
        # The regex \{[\s\S]*\} is greedy - captures from first { to last }
        text = '{"first": 1} and {"second": 2}'
        result = extract_json_from_text(text)
        # Greedy match results in invalid JSON string, returns fallback
        assert result == {"candidates": []}

    def test_extracts_empty_json_object(self):
        text = "empty: {}"
        result = extract_json_from_text(text)
        assert result == {}


class TestNormalizeProposal:
    def test_returns_as_is_if_already_normalized(self):
        proposal = {"site": "moncler.com", "page_type": "pdp", "candidates": [{"target": "title"}]}
        result = normalize_proposal(proposal, "moncler.com", "pdp")
        assert result is proposal

    def test_converts_proposed_selectors_to_candidates(self):
        proposal = {
            "proposed_selectors": ["h1.title", "div.name"],
            "rationale": "Changed structure",
        }
        result = normalize_proposal(proposal, "example.com", "plp")
        assert result["site"] == "example.com"
        assert result["page_type"] == "plp"
        assert result["strategy"] == "llm_selector_healing_v1"
        assert len(result["candidates"]) == 2
        assert result["candidates"][0]["new_selector"] == "h1.title"
        assert result["candidates"][0]["confidence"] == 0.8
        assert result["candidates"][1]["new_selector"] == "div.name"

    def test_preserves_existing_candidates(self):
        proposal = {"candidates": [{"target": "price", "new_selector": ".price-tag", "confidence": 0.95}]}
        result = normalize_proposal(proposal, "example.com", "pdp")
        assert len(result["candidates"]) == 1
        assert result["candidates"][0]["new_selector"] == ".price-tag"

    def test_empty_proposal_returns_empty_candidates(self):
        result = normalize_proposal({}, "site.com", "plp")
        assert result["site"] == "site.com"
        assert result["candidates"] == []
        assert result["strategy"] == "llm_selector_healing_v1"

    def test_proposed_selectors_without_rationale(self):
        proposal = {"proposed_selectors": [".item"]}
        result = normalize_proposal(proposal, "site.com", "plp")
        assert result["candidates"][0]["reason"] == "LLM proposal"
