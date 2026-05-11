"""Tests for selector_prompt_builder module."""

from __future__ import annotations

import json

from app.agents.browser.selector_prompt_builder import (
    build_selector_repair_prompt,
    extract_failed_selector_from_error,
    extract_site_constraints,
    optimize_dom_snippet,
)


class TestExtractFailedSelectorFromError:
    def test_extracts_selector_from_error(self):
        msg = "Timeout waiting for selector: .product-title"
        assert extract_failed_selector_from_error(msg) == ".product-title"

    def test_extracts_selector_case_insensitive(self):
        msg = "Selector: #main-content not found"
        assert extract_failed_selector_from_error(msg) == "#main-content"

    def test_returns_none_when_no_selector(self):
        assert extract_failed_selector_from_error("Generic error") is None

    def test_handles_colon_without_space(self):
        msg = "selector:.item"
        result = extract_failed_selector_from_error(msg)
        assert result == ".item"


class TestExtractSiteConstraints:
    def test_extracts_allowed_domain(self):
        constraints = extract_site_constraints(
            site="moncler.com",
            site_config={"allowed_domain": "moncler.com"},
            page_type="pdp",
        )
        assert constraints["allowed_domain"] == "moncler.com"

    def test_extracts_url_patterns(self):
        constraints = extract_site_constraints(
            site="example.com",
            site_config={"url_rules": {"allow_path_patterns": ["/products/jackets", "/p/123"]}},
            page_type="plp",
        )
        assert "/products/" in constraints["url_patterns"]
        assert "/p/" in constraints["url_patterns"]

    def test_empty_config_returns_empty(self):
        constraints = extract_site_constraints(site="test.com", site_config={}, page_type="pdp")
        assert constraints == {}

    def test_deduplicates_url_patterns(self):
        constraints = extract_site_constraints(
            site="test.com",
            site_config={"url_rules": {"allow_path_patterns": ["/products/a", "/products/b"]}},
            page_type="plp",
        )
        assert constraints["url_patterns"] == ["/products/"]


class TestOptimizeDomSnippet:
    def test_removes_script_tags(self):
        html = "<div><script>alert('xss')</script><p>Content</p></div>"
        result = optimize_dom_snippet(html)
        assert "<script>" not in result
        assert "Content" in result

    def test_removes_style_tags(self):
        html = "<div><style>.x{color:red}</style><p>Text</p></div>"
        result = optimize_dom_snippet(html)
        assert "<style>" not in result
        assert "Text" in result

    def test_removes_html_comments(self):
        html = "<div><!-- comment --><p>Text</p></div>"
        result = optimize_dom_snippet(html)
        assert "comment" not in result
        assert "Text" in result

    def test_truncates_long_html(self):
        html = "<div>" + "x" * 20000 + "</div>"
        result = optimize_dom_snippet(html, max_chars=1000)
        assert len(result) <= 1000

    def test_focuses_around_failed_selector(self):
        html = "<div><span class='target'>Found</span></div>"
        result = optimize_dom_snippet(html, failed_selector=".target")
        assert "Found" in result

    def test_handles_invalid_selector(self):
        html = "<div><p>Content</p></div>"
        result = optimize_dom_snippet(html, failed_selector="???invalid???")
        assert "Content" in result

    def test_none_selector_returns_full_soup(self):
        html = "<div><p>Hello</p></div>"
        result = optimize_dom_snippet(html, failed_selector=None)
        assert "Hello" in result


class TestBuildSelectorRepairPrompt:
    def _make_kwargs(self, **overrides):
        defaults = {
            "site": "example.com",
            "page_type": "pdp",
            "failure_context": {"error_type": "timeout", "error_message": "selector: .title not found", "error_class": "TimeoutError"},
            "failure_analysis": {"summary": "Selector not found", "root_causes": ["DOM changed"], "suggested_fixes": ["Try data-testid"]},
            "dom_snapshot_html": "<div><h1 class='title'>Product</h1></div>",
            "current_selectors": {"title": ".title"},
        }
        defaults.update(overrides)
        return defaults

    def test_basic_prompt_structure(self):
        prompt = build_selector_repair_prompt(**self._make_kwargs())
        assert "example.com" in prompt
        assert "PDP" in prompt
        assert "timeout" in prompt
        assert "JSON" in prompt

    def test_includes_failure_analysis(self):
        prompt = build_selector_repair_prompt(**self._make_kwargs())
        assert "Selector not found" in prompt
        assert "DOM changed" in prompt

    def test_includes_site_constraints(self):
        prompt = build_selector_repair_prompt(
            **self._make_kwargs(
                site_config={"allowed_domain": "example.com"},
                page_type="plp",
            ),
        )
        assert "example.com" in prompt
        assert "Domain" in prompt

    def test_plp_page_type_has_element_type_constraint(self):
        prompt = build_selector_repair_prompt(
            **self._make_kwargs(
                site_config={"allowed_domain": "example.com"},
                page_type="plp",
            ),
        )
        assert "<a>" in prompt

    def test_includes_previous_successes(self):
        prompt = build_selector_repair_prompt(
            **self._make_kwargs(
                previous_successes=[{"selector": ".old-title", "reason": "worked before"}],
            ),
        )
        assert ".old-title" in prompt
        assert "worked before" in prompt

    def test_includes_previous_failures(self):
        prompt = build_selector_repair_prompt(
            **self._make_kwargs(
                previous_failures=[{"selector": ".broken", "reason": "not found"}],
            ),
        )
        assert ".broken" in prompt
        assert "避けてください" in prompt

    def test_no_feedback_section_when_none(self):
        prompt = build_selector_repair_prompt(**self._make_kwargs())
        assert "Previous Successful" not in prompt
        assert "Previous Failed" not in prompt

    def test_limits_feedback_to_5_entries(self):
        successes = [{"selector": f".s-{i}", "reason": f"r-{i}"} for i in range(10)]
        prompt = build_selector_repair_prompt(**self._make_kwargs(previous_successes=successes))
        assert ".s-4" in prompt
        assert ".s-5" not in prompt
