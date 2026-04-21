"""
包括テスト: ai_vision_agent, config/protocols, config/paths
"""
from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ==========================================
# ai_vision_agent
# ==========================================
class TestAIVisionAgentInit:
    def test_init_without_context(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        assert agent.run_context is None

    def test_init_with_context(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        ctx = {"key": "value"}
        agent = AIVisionAgent(run_context=ctx)
        assert agent.run_context == ctx


class TestInspectParcelImages:
    def test_basic_inspection(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        result = agent.inspect_parcel_images(
            parcel_id="P001",
            image_urls=["http://img1.jpg", "http://img2.jpg"],
        )

        assert result["parcel_id"] == "P001"
        assert result["image_count"] == 2
        assert len(result["raw_analysis"]) == 2
        assert result["issues"] == []  # no damage, all logos visible

    def test_empty_image_list(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        result = agent.inspect_parcel_images(parcel_id="P002", image_urls=[])

        assert result["parcel_id"] == "P002"
        assert result["image_count"] == 0
        assert result["raw_analysis"] == []
        assert result["issues"] == []

    def test_damage_detected(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        with patch.object(agent, "_analyze_image", return_value={
            "image_url": "url",
            "damage_detected": True,
            "brand_logo_visible": True,
            "notes": "Damage found",
        }):
            result = agent.inspect_parcel_images(parcel_id="P003", image_urls=["url"])
            assert "Potential damage detected." in result["issues"]

    def test_no_brand_logo_visible(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        with patch.object(agent, "_analyze_image", return_value={
            "image_url": "url",
            "damage_detected": False,
            "brand_logo_visible": False,
            "notes": "No logo",
        }):
            result = agent.inspect_parcel_images(parcel_id="P004", image_urls=["url"])
            assert "Brand logo not visible in provided photos." in result["issues"]

    def test_analyze_image_returns_stub(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        result = agent._analyze_image("http://example.com/img.jpg")

        assert result["image_url"] == "http://example.com/img.jpg"
        assert result["damage_detected"] is False
        assert result["brand_logo_visible"] is True
        assert "not implemented" in result["notes"].lower()

    def test_summarize_issues_all_good(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        results = [
            {"damage_detected": False, "brand_logo_visible": True},
            {"damage_detected": False, "brand_logo_visible": True},
        ]
        assert agent._summarize_issues(results) == []

    def test_summarize_issues_empty_results(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        assert agent._summarize_issues([]) == []

    def test_summarize_issues_damage_and_no_logo(self):
        from app.agents.ai_vision_agent import AIVisionAgent

        agent = AIVisionAgent()
        results = [{"damage_detected": True, "brand_logo_visible": False}]
        issues = agent._summarize_issues(results)
        assert len(issues) == 2


# ==========================================
# config/protocols
# ==========================================
class TestSiteConfigProviderProtocol:
    def test_protocol_is_runtime_checkable(self):
        from app.config.protocols import SiteConfigProvider

        class MockProvider:
            def get_site_config(self, site_name):
                return {"name": site_name}

            def get_full_config(self):
                return {}

        assert isinstance(MockProvider(), SiteConfigProvider)

    def test_protocol_not_satisfied_by_incomplete_class(self):
        from app.config.protocols import SiteConfigProvider

        class IncompleteProvider:
            def get_site_config(self, site_name):
                return None

        assert not isinstance(IncompleteProvider(), SiteConfigProvider)


class TestDefaultSiteConfigProvider:
    @patch("app.config.loader.get_site_config")
    def test_get_site_config_delegates(self, mock_get):
        from app.config.protocols import DefaultSiteConfigProvider

        mock_get.return_value = {"name": "test_site"}
        provider = DefaultSiteConfigProvider()
        result = provider.get_site_config("test_site")

        mock_get.assert_called_once_with("test_site")
        assert result == {"name": "test_site"}

    @patch("app.config.loader.get_site_config")
    def test_get_site_config_returns_none(self, mock_get):
        from app.config.protocols import DefaultSiteConfigProvider

        mock_get.return_value = None
        provider = DefaultSiteConfigProvider()
        assert provider.get_site_config("nonexistent") is None

    @patch("app.config.loader.load_full_config")
    def test_get_full_config_delegates(self, mock_load):
        from app.config.protocols import DefaultSiteConfigProvider

        mock_load.return_value = {"sites": {"site1": {}}}
        provider = DefaultSiteConfigProvider()
        result = provider.get_full_config()

        mock_load.assert_called_once()
        assert "sites" in result

    def test_default_provider_satisfies_protocol(self):
        from app.config.protocols import DefaultSiteConfigProvider, SiteConfigProvider

        provider = DefaultSiteConfigProvider()
        assert isinstance(provider, SiteConfigProvider)


# ==========================================
# config/paths
# ==========================================
class TestPaths:
    def test_project_root_is_path(self):
        from app.config.paths import PROJECT_ROOT

        assert isinstance(PROJECT_ROOT, Path)

    def test_standard_dirs_are_paths(self):
        from app.config.paths import (
            DATA_GENERATED_DIR,
            DOCS_REPORTS_DIR,
            LOGS_GENERATED_DIR,
            PROJECT_ROOT,
            SCRIPTS_DEV_DIR,
            TMP_GENERATED_DIR,
        )

        for d in [SCRIPTS_DEV_DIR, DOCS_REPORTS_DIR, LOGS_GENERATED_DIR, TMP_GENERATED_DIR, DATA_GENERATED_DIR]:
            assert isinstance(d, Path)
            assert str(d).startswith(str(PROJECT_ROOT))

    def test_compatibility_dirs_exist(self):
        from app.config.paths import INSTANCE_DIR, OUTPUT_DIR, EXPORTS_DIR

        for d in [INSTANCE_DIR, OUTPUT_DIR, EXPORTS_DIR]:
            assert isinstance(d, Path)


class TestEnsureDirs:
    def test_ensure_dirs_creates_directories(self, tmp_path):
        from app.config import paths

        # Override paths to use tmp_path
        test_dirs = [
            tmp_path / "scripts" / "dev",
            tmp_path / "docs" / "reports",
            tmp_path / "logs" / "generated",
            tmp_path / "tmp" / "generated",
            tmp_path / "data" / "generated",
        ]

        with patch("app.config.paths.SCRIPTS_DEV_DIR", test_dirs[0]):
            with patch("app.config.paths.DOCS_REPORTS_DIR", test_dirs[1]):
                with patch("app.config.paths.LOGS_GENERATED_DIR", test_dirs[2]):
                    with patch("app.config.paths.TMP_GENERATED_DIR", test_dirs[3]):
                        with patch("app.config.paths.DATA_GENERATED_DIR", test_dirs[4]):
                            paths.ensure_dirs()

                            for d in test_dirs:
                                assert d.exists()
