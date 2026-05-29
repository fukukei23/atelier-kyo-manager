"""Tests for app/utils/scout_models.py and app/utils/llm_protocol.py"""
import pytest
from app.utils.scout_models import SiteSelectors, SiteConfig, default_sites
from app.utils.llm_protocol import LLMClient


class TestSiteSelectors:
    def test_default_values(self):
        sel = SiteSelectors()
        assert sel.search_open == []
        assert sel.search_input == []
        assert sel.search_submit == []
        assert sel.results_item is None
        assert sel.first_product_link is None
        assert sel.pdp_title == []
        assert sel.pdp_price == []

    def test_custom_values(self):
        sel = SiteSelectors(
            search_input=["input#search", "input[name='q']"],
            results_item=".product-card",
        )
        assert len(sel.search_input) == 2
        assert sel.results_item == ".product-card"


class TestSiteConfig:
    def test_create_minimal(self):
        cfg = SiteConfig(name="TEST", home_url="https://example.com")
        assert cfg.name == "TEST"
        assert cfg.home_url == "https://example.com"
        assert cfg.search_mode == "human"
        assert cfg.timeout_sec == 25
        assert cfg.force_ui_search is False
        assert cfg.currency_hint is None

    def test_create_full(self):
        sel = SiteSelectors(results_item=".item")
        cfg = SiteConfig(
            name="SHOP",
            home_url="https://shop.example.com",
            domains=["shop.example.com"],
            search_template="https://shop.example.com/search?q={q}",
            timeout_sec=30,
            currency_hint="JPY",
            selectors=sel,
        )
        assert cfg.domains == ["shop.example.com"]
        assert cfg.timeout_sec == 30
        assert cfg.currency_hint == "JPY"
        assert cfg.selectors.results_item == ".item"

    def test_selectors_default_factory(self):
        cfg1 = SiteConfig(name="A", home_url="http://a.com")
        cfg2 = SiteConfig(name="B", home_url="http://b.com")
        # Each instance has independent selectors
        assert cfg1.selectors is not cfg2.selectors


class TestDefaultSites:
    def test_returns_list(self):
        sites = default_sites()
        assert isinstance(sites, list)
        assert len(sites) > 0

    def test_all_have_name(self):
        for site in default_sites():
            assert site.name

    def test_named_sites_have_url(self):
        # GENERIC_PDP intentionally has empty home_url
        sites_with_url = [s for s in default_sites() if s.home_url]
        assert len(sites_with_url) > 0
        for site in sites_with_url:
            assert site.home_url.startswith("http")

    def test_known_sites_present(self):
        names = [s.name for s in default_sites()]
        assert "SSENSE" in names
        assert "BUYMA" in names
        assert "FARFETCH" in names

    def test_each_call_returns_new_list(self):
        sites1 = default_sites()
        sites2 = default_sites()
        assert sites1 is not sites2


class TestLLMClientProtocol:
    def test_concrete_class_satisfies_protocol(self):
        class MockLLM:
            def generate(self, prompt, task_type="default", tools=None, stream=False, chunk_callback=None):
                return {"text": "response"}

        assert isinstance(MockLLM(), LLMClient)

    def test_missing_generate_fails_protocol(self):
        class NoGenerate:
            pass

        assert not isinstance(NoGenerate(), LLMClient)

    def test_protocol_is_runtime_checkable(self):
        # LLMClient is decorated with @runtime_checkable
        assert hasattr(LLMClient, "__protocol_attrs__") or hasattr(LLMClient, "_is_protocol")
