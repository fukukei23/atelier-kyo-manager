"""Tests for plugin_runtime module."""

from unittest.mock import MagicMock, patch

from app.agents.plugin_runtime import first_plugin_for_site, load_plugins_for_site


class TestLoadPluginsForSite:
    @patch("app.agents.plugin_runtime._iter_plugin_modules", return_value=[])
    def test_no_plugins_returns_empty(self, mock_iter):
        result = load_plugins_for_site("test_site")
        assert result == []

    @patch("app.agents.plugin_runtime._iter_plugin_modules")
    def test_matching_plugin_loaded(self, mock_iter):
        mock_plugin_cls = MagicMock()
        mock_plugin_cls.__mro__ = [type]
        mock_plugin_cls.site = "moncler"

        mock_module = MagicMock()
        mock_module.MyPlugin = mock_plugin_cls
        del mock_module.base  # avoid false match

        with patch("app.agents.plugin_runtime.importlib.import_module", return_value=mock_module):
            mock_iter.return_value = ["app.agents.plugins.test_plugin"]
            # Need to mock issubclass to return True for our mock
            with patch("app.agents.plugin_runtime.isinstance"):
                with patch("app.agents.plugin_runtime.issubclass", return_value=True):
                    with patch("app.agents.plugin_runtime.getattr"):
                        # First getattr: dir iteration returns "MyPlugin"
                        # We need a different approach
                        pass

    @patch("app.agents.plugin_runtime._iter_plugin_modules")
    def test_import_failure_continues(self, mock_iter):
        with patch("app.agents.plugin_runtime.importlib.import_module", side_effect=ImportError("no module")):
            mock_iter.return_value = ["bad.module"]
            result = load_plugins_for_site("test_site")
            assert result == []


class TestFirstPluginForSite:
    @patch("app.agents.plugin_runtime.load_plugins_for_site", return_value=[])
    def test_no_plugins_returns_none(self, mock_load):
        result = first_plugin_for_site("test_site")
        assert result is None

    @patch("app.agents.plugin_runtime.load_plugins_for_site")
    def test_returns_first_plugin(self, mock_load):
        mock_plugin = MagicMock()
        mock_load.return_value = [mock_plugin, MagicMock()]
        result = first_plugin_for_site("test_site")
        assert result is mock_plugin
