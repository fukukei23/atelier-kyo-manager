"""
config/loader.py テスト (Issue #71)

deep_merge, load_full_config, get_site_config,
load_and_merge_configs をカバー
"""

import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from app.config.loader import (
    deep_merge,
    get_site_config,
    load_and_merge_configs,
    load_full_config,
)


# === deep_merge ===


def test_deep_merge_flat():
    """フラットな辞書のマージ（source の値が destination に書き込まれる）"""
    dst = {"b": 3, "c": 4}
    result = deep_merge({"a": 1, "b": 2}, dst)
    assert result["a"] == 1
    assert result["b"] == 2  # source の値で上書き
    assert result["c"] == 4


def test_deep_merge_nested():
    """入れ子辞書の再帰マージ"""
    src = {"outer": {"inner": "new", "kept": "src"}}
    dst = {"outer": {"existing": "dst", "kept": "dst"}}
    result = deep_merge(src, dst)
    assert result["outer"]["inner"] == "new"
    assert result["outer"]["existing"] == "dst"
    assert result["outer"]["kept"] == "src"


def test_deep_merge_overwrite_non_dict():
    """辞書→非辞書の上書き"""
    result = deep_merge({"key": "value"}, {})
    assert result == {"key": "value"}


# === load_full_config ===


def test_load_full_config_returns_dict():
    """正常系: dict を返す"""
    import app.config.loader as loader
    loader._cached_config = None  # キャッシュクリア
    config = load_full_config()
    assert isinstance(config, dict)
    assert len(config) > 0


def test_load_full_config_caching():
    """キャッシュ: 2回目は同じオブジェクト"""
    import app.config.loader as loader
    loader._cached_config = None
    first = load_full_config()
    second = load_full_config()
    assert first is second


def test_load_full_config_force_reload():
    """force_reload=True でキャッシュ破棄"""
    import app.config.loader as loader
    loader._cached_config = None
    first = load_full_config()
    second = load_full_config(force_reload=True)
    # force_reload でも中身は同じ（ファイルが変わっていなければ）
    assert first.keys() == second.keys()


def test_load_full_config_file_not_found(tmp_path):
    """base.json なし → 例外発生"""
    import app.config.loader as loader
    loader._cached_config = None

    with patch.object(loader, "APP_ROOT", tmp_path):
        with pytest.raises(FileNotFoundError):
            load_full_config()


def test_load_full_config_invalid_json(tmp_path):
    """不正JSON → JSONDecodeError"""
    import app.config.loader as loader
    loader._cached_config = None

    config_dir = tmp_path / "app" / "config" / "sites"
    config_dir.mkdir(parents=True)
    (config_dir / "base.json").write_text("{invalid", encoding="utf-8")

    with patch.object(loader, "APP_ROOT", tmp_path):
        with pytest.raises(json.JSONDecodeError):
            load_full_config()


def test_load_full_config_with_overrides_new_site(tmp_path):
    """overrides に新しいサイト定義が追加されるパス (L90-95)"""
    import app.config.loader as loader
    loader._cached_config = None

    config_dir = tmp_path / "app" / "config" / "sites"
    config_dir.mkdir(parents=True)

    base = {
        "sites": [{"name": "BUYMA", "home_url": "https://buyma.com"}],
        "default_discovery_settings": {"mode": "auto"},
    }
    (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

    # overrides に base にない新しいサイトを追加
    overrides = {"NEW_SITE": {"home_url": "https://new-site.com"}}
    (config_dir / "overrides.local.json").write_text(json.dumps(overrides), encoding="utf-8")

    with patch.object(loader, "APP_ROOT", tmp_path):
        config = load_full_config()

    assert "BUYMA" in config
    assert "NEW_SITE" in config
    assert config["NEW_SITE"]["home_url"] == "https://new-site.com"


def test_load_full_config_with_overrides_merge_existing(tmp_path):
    """overrides で既存サイトの設定をマージ"""
    import app.config.loader as loader
    loader._cached_config = None

    config_dir = tmp_path / "app" / "config" / "sites"
    config_dir.mkdir(parents=True)

    base = {
        "sites": [{"name": "BUYMA", "home_url": "https://buyma.com", "timeout": 30}],
        "default_discovery_settings": {},
    }
    (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")

    overrides = {"BUYMA": {"timeout": 60, "extra": True}}
    (config_dir / "overrides.local.json").write_text(json.dumps(overrides), encoding="utf-8")

    with patch.object(loader, "APP_ROOT", tmp_path):
        config = load_full_config()

    assert config["BUYMA"]["timeout"] == 60
    assert config["BUYMA"]["extra"] is True


def test_load_full_config_no_overrides_file(tmp_path):
    """overrides.local.json なし → baseのみ"""
    import app.config.loader as loader
    loader._cached_config = None

    config_dir = tmp_path / "app" / "config" / "sites"
    config_dir.mkdir(parents=True)

    base = {"sites": [{"name": "TEST"}], "default_discovery_settings": {}}
    (config_dir / "base.json").write_text(json.dumps(base), encoding="utf-8")
    # overrides は作成しない

    with patch.object(loader, "APP_ROOT", tmp_path):
        config = load_full_config()

    assert "TEST" in config


# === get_site_config ===


def test_get_site_config_existing():
    """存在するサイト → 設定dict"""
    import app.config.loader as loader
    loader._cached_config = None
    config = get_site_config("BUYMA")
    assert config is not None


def test_get_site_config_missing():
    """存在しないサイト → None"""
    import app.config.loader as loader
    loader._cached_config = None
    assert get_site_config("NONEXISTENT_SITE") is None


# === load_and_merge_configs (deprecated alias, L118-122) ===


def test_load_and_merge_configs_deprecated_warning(caplog):
    """非推奨エイリアスが警告を出す"""
    import app.config.loader as loader
    loader._cached_config = None
    with caplog.at_level(logging.WARNING, logger="app.config.loader"):
        config = load_and_merge_configs()
    assert isinstance(config, dict)
    assert "deprecated" in caplog.text.lower()
