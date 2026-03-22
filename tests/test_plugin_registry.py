# -*- coding: utf-8 -*-
"""
test_plugin_registry.py
======================================================================
Pluginレジストリのユニットテスト
======================================================================
"""
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


def test_plugin_registry_import():
    """Pluginレジストリのインポートテスト"""
    from app.agents.browser.plugins import get_plugin_registry, get_plugin_for_site

    registry = get_plugin_registry()
    assert isinstance(registry, dict)


def test_moncler_plugin_registered():
    """Monclerプラグインが登録されていることを確認"""
    from app.agents.browser.plugins import get_plugin_for_site

    plugin = get_plugin_for_site("MONCLER_OFFICIAL")
    assert plugin is not None
    assert hasattr(plugin, 'site')
    assert plugin.site == "MONCLER_OFFICIAL"


def test_nonexistent_plugin_returns_none():
    """存在しないプラグインはNoneを返す"""
    from app.agents.browser.plugins import get_plugin_for_site

    plugin = get_plugin_for_site("NONEXISTENT_SITE_XYZ")
    assert plugin is None


def test_plugin_has_required_methods():
    """登録済みプラグインが必要なメソッドを持っていることを確認"""
    from app.agents.browser.plugins import get_plugin_for_site

    plugin = get_plugin_for_site("MONCLER_OFFICIAL")
    assert plugin is not None

    # 必須メソッドの存在確認
    assert hasattr(plugin, 'before_navigate')
    assert hasattr(plugin, 'after_navigate')
    assert hasattr(plugin, 'assert_plp')
    assert hasattr(plugin, 'materialize')


def test_registry_is_singleton():
    """レジストリがシングルトンであることを確認（複数回呼んでも同一オブジェクト）"""
    from app.agents.browser.plugins import get_plugin_registry

    registry1 = get_plugin_registry()
    registry2 = get_plugin_registry()
    assert registry1 is registry2


def test_all_expected_plugins_registered():
    """期待される全てのPluginが登録されていることを確認"""
    from app.agents.browser.plugins import get_plugin_registry

    registry = get_plugin_registry()
    expected = ["MONCLER_OFFICIAL", "GUCCI", "PRADA", "SSENSE"]

    for site in expected:
        assert site in registry, f"{site} should be in registry"
        plugin = registry[site]
        assert hasattr(plugin, 'site')
        assert plugin.site == site
