"""
包括テスト: config/protocols, config/paths
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch


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
        from app.config.paths import EXPORTS_DIR, INSTANCE_DIR, OUTPUT_DIR

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

        with (
            patch("app.config.paths.SCRIPTS_DEV_DIR", test_dirs[0]),
            patch("app.config.paths.DOCS_REPORTS_DIR", test_dirs[1]),
            patch("app.config.paths.LOGS_GENERATED_DIR", test_dirs[2]),
            patch("app.config.paths.TMP_GENERATED_DIR", test_dirs[3]),
            patch("app.config.paths.DATA_GENERATED_DIR", test_dirs[4]),
        ):
            paths.ensure_dirs()

            for d in test_dirs:
                assert d.exists()
