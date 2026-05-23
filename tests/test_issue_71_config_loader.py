"""Issue #71: config/loader.py カバレッジ 77→95%+

Tests for uncovered paths:
- L90-98: overrides with new site (not in base)
- L101-103: JSON decode error / FileNotFoundError
- L118-127: deprecated alias load_and_merge_configs
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config import loader
from app.config.loader import deep_merge, load_and_merge_configs, load_full_config


@pytest.fixture(autouse=True)
def clear_cache():
    loader._cached_config = None
    yield
    loader._cached_config = None


class TestDeepMerge:
    def test_simple_merge(self):
        dest = {"b": 2}
        result = deep_merge({"a": 1}, dest)
        assert result == {"b": 2, "a": 1}

    def test_nested_merge_source_overwrites_dest(self):
        dest = {"outer": {"a": 1, "b": 2}}
        result = deep_merge({"outer": {"b": 3, "c": 4}}, dest)
        assert result["outer"] == {"a": 1, "b": 3, "c": 4}

    def test_overwrite_scalar(self):
        dest = {"key": "old"}
        result = deep_merge({"key": "new"}, dest)
        assert result["key"] == "new"


class TestLoadFullConfigWithOverrides:
    def _make_config_dir(self, tmp_path):
        """Create config structure matching APP_ROOT / app / config / sites"""
        config_dir = tmp_path / "app" / "config" / "sites"
        config_dir.mkdir(parents=True)
        return config_dir

    def test_override_new_site_not_in_base(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)

        base = {"sites": [{"name": "SSENSE", "base_url": "https://ssense.com"}], "default_discovery_settings": {}}
        (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

        overrides = {"NEW_SITE": {"base_url": "https://new.com"}}
        (config_dir / "overrides.local.json").write_text(json.dumps(overrides), encoding="utf-8")

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            result = load_full_config(force_reload=True)

        assert "NEW_SITE" in result
        assert result["NEW_SITE"]["base_url"] == "https://new.com"

    def test_override_existing_site(self, tmp_path):
        config_dir = self._make_config_dir(tmp_path)

        base = {"sites": [{"name": "SSENSE", "base_url": "https://ssense.com"}], "default_discovery_settings": {}}
        (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

        overrides = {"SSENSE": {"extra_key": "extra_val"}}
        (config_dir / "overrides.local.json").write_text(json.dumps(overrides), encoding="utf-8")

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            result = load_full_config(force_reload=True)

        assert result["SSENSE"]["extra_key"] == "extra_val"


class TestLoadFullConfigErrors:
    def test_file_not_found_raises(self, tmp_path):
        config_dir = tmp_path / "app" / "config" / "sites"
        config_dir.mkdir(parents=True)

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            with pytest.raises(FileNotFoundError):
                load_full_config(force_reload=True)

    def test_invalid_json_raises(self, tmp_path):
        config_dir = tmp_path / "app" / "config" / "sites"
        config_dir.mkdir(parents=True)
        (config_dir / "base.json").write_text("not valid json{{{", encoding="utf-8")

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            with pytest.raises(json.JSONDecodeError):
                load_full_config(force_reload=True)


class TestDeprecatedAlias:
    def test_load_and_merge_configs_returns_config(self, tmp_path):
        config_dir = tmp_path / "app" / "config" / "sites"
        config_dir.mkdir(parents=True)

        base = {"sites": [{"name": "TEST", "base_url": "https://test.com"}], "default_discovery_settings": {}}
        (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            result = load_and_merge_configs(force_reload=True)

        assert "TEST" in result


class TestCaching:
    def test_cache_returns_same_object(self, tmp_path):
        config_dir = tmp_path / "app" / "config" / "sites"
        config_dir.mkdir(parents=True)

        base = {"sites": [{"name": "A", "base_url": "a"}], "default_discovery_settings": {}}
        (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

        with patch("app.config.loader.APP_ROOT", tmp_path):
            loader._cached_config = None
            r1 = load_full_config(force_reload=True)
            r2 = load_full_config(force_reload=False)
            assert r1 is r2
