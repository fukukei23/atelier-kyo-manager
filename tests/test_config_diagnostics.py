"""
包括テスト: config/config_loader, utils/diagnostics
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.utils.diagnostics import BUNDLE_FILES_GLOBS, make_diagnostic_bundle


# ==========================================
# config/config_loader
# ==========================================
class TestConfigGetSecretsPriority:
    """Config.get() secrets > .env > OS env の3段階優先順位"""

    @patch("app.config.config_loader.Secrets")
    def test_secrets_returns_value(self, mock_secrets):
        from app.config.config_loader import Config

        mock_secrets.API_KEY = "from_secrets"
        result = Config.get("API_KEY")
        assert result == "from_secrets"

    @patch("app.config.config_loader.Secrets", None)
    def test_secrets_none_falls_to_dotenv(self, tmp_path):
        from app.config.config_loader import Config

        env_file = tmp_path / ".env"
        env_file.write_text("API_KEY=from_dotenv\n")

        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_resolved = MagicMock()
            mock_resolved.parents.__getitem__ = MagicMock(return_value=tmp_path)
            mock_path_cls.return_value.resolve.return_value = mock_resolved
            mock_path_cls.return_value.exists.return_value = True
            mock_path_cls.return_value.read_text.return_value = "API_KEY=from_dotenv\n"

            result = Config.get("API_KEY")
            assert result == "from_dotenv"

    @patch("app.config.config_loader.Secrets", None)
    def test_dotenv_missing_falls_to_os_environ(self):
        from app.config.config_loader import Config

        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            with patch.dict(os.environ, {"MY_VAR": "from_os"}, clear=False):
                result = Config.get("MY_VAR")
                assert result == "from_os"

    @patch("app.config.config_loader.Secrets", None)
    def test_all_missing_returns_default(self):
        from app.config.config_loader import Config

        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            with patch.dict(os.environ, {}, clear=True):
                result = Config.get("MISSING_KEY", default="fallback")
                assert result == "fallback"

    @patch("app.config.config_loader.Secrets", None)
    def test_all_missing_no_default_returns_none(self):
        from app.config.config_loader import Config

        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            with patch.dict(os.environ, {}, clear=True):
                result = Config.get("MISSING_KEY")
                assert result is None

    @patch("app.config.config_loader.Secrets")
    def test_secrets_empty_string_falls_through(self, mock_secrets):
        from app.config.config_loader import Config

        mock_secrets.EMPTY = ""
        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            with patch.dict(os.environ, {"EMPTY": "from_os"}, clear=False):
                result = Config.get("EMPTY")
                assert result == "from_os"

    @patch("app.config.config_loader.Secrets", None)
    def test_dotenv_skips_comments_and_empty_lines(self):
        from app.config.config_loader import Config

        env_content = "# comment\n\nKEY=val\n  # indented\n"

        # Patch the file path resolution by patching Path.__init__ behavior
        import app.config.config_loader as mod

        # Directly test the .env parsing logic by patching exists/read_text
        with patch.object(mod.Path, "exists", return_value=True):
            with patch.object(mod.Path, "read_text", return_value=env_content):
                result = Config.get("KEY")
                assert result == "val"

    @patch("app.config.config_loader.Secrets")
    def test_secrets_non_string_falls_through(self, mock_secrets):
        from app.config.config_loader import Config

        mock_secrets.COUNT = 42  # int, not str
        with patch("app.config.config_loader.Path") as mock_path_cls:
            mock_path_cls.return_value.exists.return_value = False
            with patch.dict(os.environ, {}, clear=True):
                result = Config.get("COUNT")
                assert result is None

    @patch("app.config.config_loader.Secrets", None)
    def test_dotenv_read_exception_falls_to_os(self):
        """Cover lines 67-68: .env read exception falls through to OS env."""
        from app.config.config_loader import Config

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "read_text", side_effect=PermissionError("denied")):
                with patch.dict(os.environ, {"KEY": "from_os"}, clear=False):
                    result = Config.get("KEY")
                    assert result == "from_os"


class TestConfigLoaderFallback:
    """Fallback import functions when loader.py unavailable."""

    def test_load_full_config_fallback(self):
        from app.config.config_loader import load_full_config

        # In test env, loader.py should be available but test the function directly
        result = load_full_config()
        assert isinstance(result, dict)

    def test_get_site_config_fallback(self):
        from app.config.config_loader import get_site_config

        result = get_site_config("nonexistent")
        # Returns None or dict depending on whether loader is available
        assert result is None or isinstance(result, dict)


# ==========================================
# utils/diagnostics
# ==========================================
class TestMakeDiagnosticBundle:
    """make_diagnostic_bundle async function tests."""

    def _make_mock_context(self, tmp_path):
        ctx = MagicMock()
        ctx.run_id = "test-run-001"
        ctx.run_path = str(tmp_path / "run_test")
        ctx.logger = MagicMock()
        return ctx

    @pytest.mark.asyncio
    async def test_creates_manifest_and_readme(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)

        with patch("app.utils.diagnostics.make_trace_report"):
            result = await make_diagnostic_bundle(ctx, include_trace_report=False)

        run_dir = Path(ctx.run_path)
        assert (run_dir / "MANIFEST.json").exists()
        assert (run_dir / "README.md").exists()

        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert manifest["run_id"] == "test-run-001"
        assert "files" in manifest

    @pytest.mark.asyncio
    async def test_creates_zip_output(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)

        with patch("app.utils.diagnostics.make_trace_report"):
            result = await make_diagnostic_bundle(ctx, include_trace_report=False)

        assert result.endswith(".zip")
        assert Path(result).exists()
        assert zipfile.is_zipfile(result)

    @pytest.mark.asyncio
    async def test_collects_matching_files(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)

        # Create files matching BUNDLE_FILES_GLOBS
        (run_dir / "ai_forensic_report.json").write_text("{}")
        (run_dir / "fail_snapshot.md").write_text("# fail")

        with patch("app.utils.diagnostics.make_trace_report"):
            result = await make_diagnostic_bundle(ctx, include_trace_report=False)

        manifest = json.loads((run_dir / "MANIFEST.json").read_text())
        assert "ai_forensic_report.json" in manifest["files"]
        assert "fail_snapshot.md" in manifest["files"]

    @pytest.mark.asyncio
    async def test_trace_zip_calls_make_trace_report(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.zip").write_text("fake zip")

        with patch("app.utils.diagnostics.make_trace_report") as mock_trace:
            await make_diagnostic_bundle(ctx, include_trace_report=True)
            mock_trace.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_trace_report_when_disabled(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.zip").write_text("fake zip")

        with patch("app.utils.diagnostics.make_trace_report") as mock_trace:
            await make_diagnostic_bundle(ctx, include_trace_report=False)
            mock_trace.assert_not_called()

    @pytest.mark.asyncio
    async def test_trace_report_exception_continues(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.zip").write_text("fake zip")

        with patch("app.utils.diagnostics.make_trace_report", side_effect=RuntimeError("fail")):
            result = await make_diagnostic_bundle(ctx, include_trace_report=True)

        # Should still complete
        assert result.endswith(".zip")
        ctx.logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_overwrites_existing_zip(self, tmp_path):
        ctx = self._make_mock_context(tmp_path)
        run_dir = tmp_path / "run_test"
        run_dir.mkdir(parents=True)

        with patch("app.utils.diagnostics.make_trace_report"):
            result1 = await make_diagnostic_bundle(ctx, include_trace_report=False)
            result2 = await make_diagnostic_bundle(ctx, include_trace_report=False)

        assert result1 == result2
        assert Path(result2).exists()


class TestBundleFilesGlobs:
    """Verify BUNDLE_FILES_GLOBS constant."""

    def test_is_list_of_strings(self):
        assert isinstance(BUNDLE_FILES_GLOBS, list)
        assert all(isinstance(g, str) for g in BUNDLE_FILES_GLOBS)

    def test_contains_key_patterns(self):
        assert "trace.zip" in BUNDLE_FILES_GLOBS
        assert "MANIFEST.json" not in BUNDLE_FILES_GLOBS  # generated separately
