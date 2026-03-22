# -*- coding: utf-8 -*-
"""
test_new_plugins.py
======================================================================
SSENSE/GUCCI/PRADA Pluginのユニットテスト
======================================================================
"""
import pytest
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class TestSSENSEPlugin:
    """SSENSE Plugin のテスト"""

    def test_plugin_import(self):
        """インポートテスト"""
        from app.agents.plugins.ssense_plp_v1 import SSENSEPLPStrategy
        assert SSENSEPLPStrategy is not None

    def test_plugin_site_name(self):
        """サイト名が正しいか確認"""
        from app.agents.plugins.ssense_plp_v1 import SSENSEPLPStrategy
        assert SSENSEPLPStrategy.site == "SSENSE"

    def test_plugin_has_required_methods(self):
        """必須メソッドの存在確認"""
        from app.agents.plugins.ssense_plp_v1 import SSENSEPLPStrategy
        plugin = SSENSEPLPStrategy()
        assert hasattr(plugin, 'before_navigate')
        assert hasattr(plugin, 'after_navigate')
        assert hasattr(plugin, 'assert_plp')
        assert hasattr(plugin, 'materialize')

    def test_default_locale(self):
        """デフォルトロケールの確認"""
        from app.agents.plugins.ssense_plp_v1 import SSENSEPLPStrategy
        plugin = SSENSEPLPStrategy()
        assert plugin._DEFAULT_LOCALE == "en-US"

    def test_hard_plp_url(self):
        """ハードPLP URLの確認"""
        from app.agents.plugins.ssense_plp_v1 import SSENSEPLPStrategy
        plugin = SSENSEPLPStrategy()
        assert "ssense.com" in plugin._HARD_PLP_URL


class TestGUCCIPlugin:
    """GUCCI Plugin のテスト"""

    def test_plugin_import(self):
        """インポートテスト"""
        from app.agents.plugins.gucci_plp_v1 import GucciPLPStrategy
        assert GucciPLPStrategy is not None

    def test_plugin_site_name(self):
        """サイト名が正しいか確認"""
        from app.agents.plugins.gucci_plp_v1 import GucciPLPStrategy
        assert GucciPLPStrategy.site == "GUCCI"

    def test_plugin_has_required_methods(self):
        """必須メソッドの存在確認"""
        from app.agents.plugins.gucci_plp_v1 import GucciPLPStrategy
        plugin = GucciPLPStrategy()
        assert hasattr(plugin, 'before_navigate')
        assert hasattr(plugin, 'after_navigate')
        assert hasattr(plugin, 'assert_plp')
        assert hasattr(plugin, 'materialize')

    def test_hard_plp_url(self):
        """ハードPLP URLの確認"""
        from app.agents.plugins.gucci_plp_v1 import GucciPLPStrategy
        plugin = GucciPLPStrategy()
        assert "gucci.com" in plugin._HARD_PLP_URL


class TestPRADAPlugin:
    """PRADA Plugin のテスト"""

    def test_plugin_import(self):
        """インポートテスト"""
        from app.agents.plugins.prada_plp_v1 import PradaPLPStrategy
        assert PradaPLPStrategy is not None

    def test_plugin_site_name(self):
        """サイト名が正しいか確認"""
        from app.agents.plugins.prada_plp_v1 import PradaPLPStrategy
        assert PradaPLPStrategy.site == "PRADA"

    def test_plugin_has_required_methods(self):
        """必須メソッドの存在確認"""
        from app.agents.plugins.prada_plp_v1 import PradaPLPStrategy
        plugin = PradaPLPStrategy()
        assert hasattr(plugin, 'before_navigate')
        assert hasattr(plugin, 'after_navigate')
        assert hasattr(plugin, 'assert_plp')
        assert hasattr(plugin, 'materialize')

    def test_hard_plp_url(self):
        """ハードPLP URLの確認"""
        from app.agents.plugins.prada_plp_v1 import PradaPLPStrategy
        plugin = PradaPLPStrategy()
        assert "prada.com" in plugin._HARD_PLP_URL


class TestPluginIntegration:
    """Plugin統合テスト"""

    def test_all_plugins_in_registry(self):
        """全てのPluginがレジストリに登録されているか確認"""
        from app.agents.browser.plugins import get_plugin_registry

        registry = get_plugin_registry()
        assert "SSENSE" in registry
        assert "GUCCI" in registry
        assert "PRADA" in registry

    def test_plugins_are_instances(self):
        """レジストリのPluginがインスタンスか確認"""
        from app.agents.browser.plugins import get_plugin_registry

        registry = get_plugin_registry()
        for name in ["SSENSE", "GUCCI", "PRADA"]:
            plugin = registry[name]
            assert plugin is not None
            assert hasattr(plugin, 'site')
            assert plugin.site == name
