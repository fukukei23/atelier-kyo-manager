import pytest

from app.agents.browser.session_manager import SessionManager
from app.core.run_context import RunContext


class _FakeTracing:
    async def start(self, **kwargs):
        return None

    async def stop(self, **kwargs):
        return None


class _FakePage:
    def __init__(self):
        self._closed = False
        self.timeout = None

    def is_closed(self):
        return self._closed

    def set_default_timeout(self, value):
        self.timeout = value


class _FakeContext:
    def __init__(self):
        self._pages = []
        self.tracing = _FakeTracing()

    @property
    def pages(self):
        return self._pages

    async def route(self, *args, **kwargs):
        return None

    async def add_init_script(self, script):
        return None

    async def add_cookies(self, cookies):
        return None

    async def new_page(self):
        page = _FakePage()
        self._pages.append(page)
        return page

    async def close(self):
        return None


class _FakeBrowser:
    def __init__(self):
        self.context = _FakeContext()

    async def new_context(self, **kwargs):
        return self.context

    async def close(self):
        return None


class _FakeChromium:
    async def launch(self, **kwargs):
        return _FakeBrowser()


class _FakePlaywright:
    def __init__(self):
        self.chromium = _FakeChromium()

    async def stop(self):
        return None


class _AsyncPlaywrightFactory:
    async def start(self):
        return _FakePlaywright()


@pytest.mark.asyncio
async def test_session_manager_open_and_close(monkeypatch, tmp_path):
    import app.agents.browser.session_manager as sm_mod

    monkeypatch.setattr(sm_mod, "async_playwright", lambda: _AsyncPlaywrightFactory())
    monkeypatch.setattr(sm_mod, "PROXY_POOL_PATH", tmp_path / "proxy_pool.json")
    sessions_dir = tmp_path / "sessions"
    sessions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(sm_mod, "SESSION_DIR", sessions_dir)

    run_ctx = RunContext(base_path=str(tmp_path / "runs"))
    mgr = SessionManager(
        site="MONCLER_OFFICIAL",
        site_config={"discovery_settings": {}},
        run_context=run_ctx,
        settings={"timeout_sec": 1},
        target_url="https://example.com",
        timeout_ms=1000,
    )

    async with mgr as session:
        assert session.page is not None
        assert session.context is not None

    # ensure handles are cleared after close
    assert mgr.page is None
    assert mgr.context is None
