# -*- coding: utf-8 -*-
"""
Task 1-1: PlpDriver のユニットテスト（モック Page/HTML）

PlpDriver の主要な機能をテストする：
- (a) Happy path: PLP → PDP success
- (b) Trap / Legal page detection
- (c) Overlay handling
- (d) Config getter methods (Stage 4)

Related Spec: docs/spec/CR-ATELIER-002_STEP5_SPEC.md
Test ID Prefix: PLP
"""

import pytest
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch
from typing import Dict, Any, List

from app.agents.browser.plp_driver import PlpDriver, PlpNavigationResult
from app.core.run_context import RunContext


@pytest.fixture
def mock_page():
    """モック Page オブジェクト"""
    page = AsyncMock()
    page.url = "https://example.com/products"
    page.is_closed.return_value = False
    page.locator.return_value.count = AsyncMock(return_value=10)
    page.locator.return_value.first = page.locator.return_value
    page.locator.return_value.nth = page.locator.return_value
    page.locator.return_value.scroll_into_view_if_needed = AsyncMock()
    page.locator.return_value.click = AsyncMock()
    page.locator.return_value.get_attribute = AsyncMock(return_value="https://example.com/product/123")
    page.evaluate = AsyncMock()
    page.wait_for_timeout = AsyncMock()
    page.wait_for_load_state = AsyncMock()
    page.wait_for_url = AsyncMock()
    page.wait_for_selector = AsyncMock()
    return page


@pytest.fixture
def mock_context():
    """モック BrowserContext オブジェクト"""
    context = AsyncMock()
    context.wait_for_event = AsyncMock(return_value=AsyncMock())
    return context


@pytest.fixture
def site_config():
    """テスト用 site_config"""
    return {
        "selectors": {
            "pdp": {
                "pdp_link_selectors": ["a.product-link"],
                "plp_container_selectors": [".product-grid"],
            },
            "ui": {
                "cookie_accept": ["#cookie-accept"],
            },
        },
        "navigation": {
            "overlays": {
                "geo_modal_selectors": ["button.close-modal"],
            },
        },
        "discovery_settings": {
            "plp_scroll_rounds": 5,
        },
    }


@pytest.fixture
def run_context():
    """テスト用 RunContext"""
    return RunContext()


@pytest.mark.asyncio
async def test_plp_driver_materialize_tiles(mock_page, mock_context, site_config, run_context):
    """PLP タイルのマテリアライズをテスト"""
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    import time
    start_t = time.time()
    budget_ms = 30000
    
    # タイル数カウントのモック設定（locator() が呼ばれるたびに新しい Locator を返す）
    def locator_side_effect(selector):
        loc = MagicMock()  # AsyncMockではなくMagicMockを使用（属性アクセスが正しく動作するように）
        loc.count = AsyncMock(return_value=12)  # 12個のタイルが見つかる
        loc.first = loc
        loc.nth = MagicMock(return_value=loc)
        return loc
    
    mock_page.locator = MagicMock(side_effect=locator_side_effect)
    mock_page.evaluate = AsyncMock()  # スクロール用
    mock_page.wait_for_timeout = AsyncMock()
    mock_page.wait_for_load_state = AsyncMock()
    mock_page.wait_for_selector = AsyncMock()  # wait_for_selectors 用
    
    tiles_seen = await driver._materialize_plp_tiles(
        start_t=start_t,
        budget_ms=budget_ms,
    )
    
    assert tiles_seen == 12
    assert mock_page.evaluate.called  # スクロールが実行された


@pytest.mark.asyncio
async def test_plp_driver_trap_detection(mock_page, mock_context, site_config, run_context):
    """
    (b) Trap / Legal page detection
    
    Given:
    - A mocked Page whose URL or content matches trap/legal page
    
    When:
    - navigate_to_pdp runs
    
    Then:
    - It should set trap_detected = True
    - trap_reason should be a non-empty string
    - If _recover_from_trap is implemented, recovery_attempted = True
    """
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    # Trap ページの URL を設定
    mock_page.url = "https://example.com/legal"
    
    trap_call_count = [0]  # 呼び出し回数をカウント
    
    def is_trap_page_side_effect(url, trap_config):
        trap_call_count[0] += 1
        # 最初の呼び出し（リカバリ前）では True を返す
        if trap_call_count[0] == 1:
            return True
        # 2回目以降（リカバリ後、URL が変更された後）では False を返す
        if "legal" in url or "terms" in url:
            return True
        return False
    
    # _recover_from_trap が呼ばれたら、URL を変更して成功を返す
    async def recover_side_effect(*args, **kwargs):
        mock_page.url = "https://example.com/products"  # リカバリ後のURL
        return True
    
    with patch.object(driver, '_materialize_tiles', return_value=5), \
         patch.object(driver, '_is_trap_page', side_effect=is_trap_page_side_effect), \
         patch.object(driver, '_recover_from_trap', new_callable=AsyncMock, side_effect=recover_side_effect) as mock_recover, \
         patch.object(driver, '_click_tile_and_navigate', return_value=mock_page):
        
        # リカバリが成功するケース
        result = await driver.navigate_to_pdp(
            start_t=time.time(),
            budget_ms=30000,
            target_url="https://example.com/products",
        )
        
        # アサーション（Stage 5: 早期トラップ検出によりrecovery_attempted=False）
        assert result.trap_detected is True
        assert result.trap_reason is not None
        assert len(result.trap_reason) > 0
        # 早期検出のため、リカバリ処理は呼ばれない
        assert result.recovery_attempted is False
        assert result.recovery_successful is False


@pytest.mark.asyncio
async def test_plp_driver_trap_detection_no_recovery(mock_page, mock_context, site_config, run_context):
    """Trap検出後、早期検出で即座にリターンする（リカバリなし）"""
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    mock_page.url = "https://example.com/legal"
    site_config["navigation"] = {
        "trap_url_patterns": ["/legal"],
    }
    
    # Stage 5: 早期トラップ検出のため、例外ではなく結果オブジェクトが返る
    result = await driver.navigate_to_pdp(
        start_t=time.time(),
        budget_ms=30000,
        target_url="https://example.com/products",
    )
    
    # 早期検出で即座にリターン（リカバリ処理は呼ばれない）
    assert result.trap_detected is True
    assert result.recovery_attempted is False
    assert result.recovery_successful is False
    assert result.tiles_seen == 0


@pytest.mark.asyncio
async def test_plp_driver_click_tile(mock_page, mock_context, site_config, run_context):
    """タイルクリック → PDP 遷移をテスト"""
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    # クリック後のページ遷移をモック
    new_page = AsyncMock()
    new_page.url = "https://example.com/product/123"
    new_page.wait_for_load_state = AsyncMock()
    new_page.wait_for_url = AsyncMock()
    
    # リンク要素のモック
    link_element = MagicMock()
    link_element.scroll_into_view_if_needed = AsyncMock()
    link_element.get_attribute = AsyncMock(return_value="https://example.com/product/123")
    link_element.click = AsyncMock()
    
    # リンクやタイルが見つかるようにモック設定
    def locator_side_effect(selector):
        loc = MagicMock()  # AsyncMockではなくMagicMockを使用
        loc.count = AsyncMock(return_value=1)  # 1つのリンクが見つかる
        loc.nth = MagicMock(return_value=link_element)
        loc.first = MagicMock()
        loc.first.scroll_into_view_if_needed = AsyncMock()
        loc.first.count = AsyncMock(return_value=1)
        loc.first.click = AsyncMock()
        return loc
    
    mock_page.locator = MagicMock(side_effect=locator_side_effect)
    
    # _click_and_wait_for_navigation の戻り値をモック（_click_and_capture_navigation をラップしている）    
    with patch.object(driver, '_click_and_wait_for_navigation', return_value=new_page):
        result_page = await driver._click_tile_and_navigate_to_pdp()
        assert result_page is not None
        assert result_page.url == "https://example.com/product/123"


@pytest.mark.asyncio
async def test_plp_driver_navigate_to_pdp_happy_path(mock_page, mock_context, site_config, run_context):
    """
    (a) Happy path: PLP → PDP success
    
    Given:
    - A mocked Page whose DOM contains at least one product tile/link
    - site_config with minimal required keys
    
    When:
    - await PlpDriver.navigate_to_pdp() is called
    
    Then:
    - It should attempt to materialize tiles
    - Click a tile via _click_tile_and_navigate_to_pdp
    - Wait for navigation (new tab / same-tab / SPA URL change)
    - Return a PlpNavigationResult with correct fields
    """
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    start_t = time.time()
    budget_ms = 30000
    
    # モック設定: タイルが8個見つかる
    mock_page.locator.return_value.count = AsyncMock(return_value=8)
    
    # 新タブが開かれるケース
    new_page = AsyncMock()
    new_page.url = "https://example.com/product/123"
    new_page.wait_for_load_state = AsyncMock()
    new_page.wait_for_url = AsyncMock()
    
    with patch.object(driver, '_materialize_tiles', return_value=8), \
         patch.object(driver, '_is_trap_page', return_value=False), \
         patch.object(driver, '_click_tile_and_navigate', return_value=new_page):
        
        result = await driver.navigate_to_pdp(
            start_t=start_t,
            budget_ms=budget_ms,
        )
        
        # アサーション
        assert isinstance(result, PlpNavigationResult)
        assert result.pdp_url == "https://example.com/product/123"
        assert result.pdp_opened_in_new_tab is True  # new_page != mock_page
        assert result.plp_url == mock_page.url
        assert result.tiles_seen == 8
        assert result.tiles_seen > 0
        assert result.trap_detected is False
        assert result.trap_reason is None
        assert result.recovery_attempted is False
        # Stage 4: 新しいフィールドのアサーション
        assert result.recovery_successful is False
        assert isinstance(result.overlays_handled, list)
        assert result.navigation_method is not None
        assert isinstance(result.errors, list)


@pytest.mark.asyncio
async def test_plp_driver_navigate_to_pdp_same_tab(mock_page, mock_context, site_config, run_context):
    """同タブでのPDP遷移をテスト"""
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    start_t = time.time()
    budget_ms = 30000
    
    # 同タブでの遷移（self.page を返す）
    mock_page.url = "https://example.com/product/123"  # URLが変更された
    
    with patch.object(driver, '_materialize_tiles', return_value=5), \
         patch.object(driver, '_is_trap_page', return_value=False), \
         patch.object(driver, '_click_tile_and_navigate', return_value=mock_page):
        
        result = await driver.navigate_to_pdp(
            start_t=start_t,
            budget_ms=budget_ms,
        )
        
        assert result.pdp_url == "https://example.com/product/123"
        assert result.pdp_opened_in_new_tab is False  # 同タブ
        assert result.tiles_seen == 5
        # Stage 4: 新しいフィールドのアサーション
        assert result.navigation_method in ("same_tab", "spa")


@pytest.mark.asyncio
async def test_plp_driver_handle_overlays(mock_page, mock_context, site_config, run_context):
    """
    (c) Overlay handling
    
    Given:
    - A mocked Page where:
      - A cookie banner element exists for selectors["ui"]["cookie_accept"]
      - A geo modal exists for navigation["overlays"]["geo_modal_selectors"]
    
    When:
    - navigate_to_pdp runs
    
    Then:
    - _handle_overlays() should be invoked and try to click those elements
    """
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    # Cookie バナーのモック
    cookie_locator_first = MagicMock()
    cookie_locator_first.count = AsyncMock(return_value=1)
    cookie_locator_first.click = AsyncMock()
    
    cookie_locator = MagicMock()
    cookie_locator.first = cookie_locator_first
    cookie_locator.count = AsyncMock(return_value=1)
    
    # Geo モーダルのモック
    geo_locator_first = MagicMock()
    geo_locator_first.count = AsyncMock(return_value=1)
    geo_locator_first.click = AsyncMock()
    
    geo_locator = MagicMock()
    geo_locator.first = geo_locator_first
    geo_locator.count = AsyncMock(return_value=1)
    
    # locator の呼び出しに応じて異なる locator を返す
    def locator_side_effect(selector):
        selector_str = str(selector).lower()
        # Cookie バナーのセレクタにマッチ
        if "#cookie-accept" in selector or "cookie" in selector_str:
            return cookie_locator
        # Geo モーダルのセレクタにマッチ
        elif "close-modal" in selector_str or "geo" in selector_str:
            return geo_locator
        else:
            default = MagicMock()
            default.first = MagicMock()
            default.first.count = AsyncMock(return_value=0)
            default.count = AsyncMock(return_value=0)
            return default
    
    mock_page.locator = MagicMock(side_effect=locator_side_effect)
    mock_page.evaluate = AsyncMock()  # オーバーレイ削除用
    mock_page.wait_for_timeout = AsyncMock()
    
    # Stage 4: _handle_overlays() は overlays_handled リストをパラメータとして受け取る
    overlays_handled: List[str] = []
    await driver._handle_overlays(overlays_handled)
    
    # Cookie バナー、Geo モーダル、オーバーレイの処理が呼ばれたことを確認
    assert mock_page.locator.called
    assert cookie_locator_first.click.called  # Cookie バナーがクリックされた
    assert geo_locator_first.click.called  # Geo モーダルがクリックされた
    assert mock_page.evaluate.called  # オーバーレイ削除が実行された
    # Stage 4: 処理したオーバーレイの種類が記録されていることを確認
    assert len(overlays_handled) > 0


# === Task 3: Config getter テスト ===

def test_get_plp_config_with_new_schema(mock_page, mock_context, run_context):
    """Stage 4: 新スキーマ（selectors.plp.*）から設定を取得するテスト"""
    site_config = {
        "selectors": {
            "plp": {
                "product_tiles": [".tile-selector"],
                "product_link": ["a.product-link"],
                "container": [".product-container"],
                "click_strategy": "tile",
                "min_tiles": 10,
                "max_scroll_rounds": 8,
                "scroll_pause_ms": 200,
            },
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_plp_config()
    
    # 新スキーマの値が取得できていることを確認
    assert config.get("product_tiles") == [".tile-selector"]
    assert config.get("product_link") == ["a.product-link"]
    assert config.get("container") == [".product-container"]
    assert config.get("click_strategy") == "tile"
    assert config.get("min_tiles") == 10
    assert config.get("max_scroll_rounds") == 8
    assert config.get("scroll_pause_ms") == 200


def test_get_plp_config_fallback_to_legacy_schema(mock_page, mock_context, run_context):
    """Stage 4: 旧スキーマ（selectors.pdp.*, discovery_settings）からフォールバックするテスト"""
    site_config = {
        "selectors": {
            "pdp": {
                "pdp_link_selectors": ["a.legacy-link"],
                "plp_container_selectors": [".legacy-container"],
            },
        },
        "discovery_settings": {
            "plp_scroll_rounds": 6,
            "plp": {
                "scroll_pause_ms": 150,
            },
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_plp_config()
    
    # 旧スキーマからフォールバックして値を取得できていることを確認
    assert config.get("product_link") == ["a.legacy-link"]
    assert config.get("container") == [".legacy-container"]
    assert config.get("max_scroll_rounds") == 6
    assert config.get("scroll_pause_ms") == 150


def test_get_overlay_config_with_new_schema(mock_page, mock_context, run_context):
    """Stage 4: 新スキーマ（navigation.overlays.*）から設定を取得するテスト"""
    site_config = {
        "navigation": {
            "overlays": {
                "cookie_banner": {
                    "selectors": ["#new-cookie-banner"],
                    "wait_after_click_ms": 800,
                },
                "geo_popup": {
                    "selectors": ["#new-geo-modal"],
                    "wait_after_click_ms": 600,
                },
                "other_overlays": {
                    "some_other": "value",
                },
            },
        },
        "selectors": {
            "ui": {
                "cookie_accept": ["#fallback-cookie"],
            },
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_overlay_config()
    
    # 新スキーマの値が取得できていることを確認
    assert config.get("cookie", {}).get("selectors") == ["#new-cookie-banner"]
    assert config.get("cookie", {}).get("wait_after_click_ms") == 800
    assert config.get("geo", {}).get("selectors") == ["#new-geo-modal"]
    assert config.get("geo", {}).get("wait_after_click_ms") == 600
    assert config.get("other", {}).get("some_other") == "value"


def test_get_overlay_config_fallback_to_legacy_schema(mock_page, mock_context, run_context):
    """Stage 4: 旧スキーマ（selectors.ui.*, navigation.overlays.geo_modal_selectors）からフォールバックするテスト"""
    site_config = {
        "navigation": {
            "overlays": {
                "geo_modal_selectors": ["button.close-geo"],
            },
        },
        "selectors": {
            "ui": {
                "cookie_accept": ["#legacy-cookie"],
            },
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_overlay_config()
    
    # 旧スキーマからフォールバックして値を取得できていることを確認
    assert config.get("cookie", {}).get("selectors") == ["#legacy-cookie"]
    assert config.get("geo", {}).get("selectors") == ["button.close-geo"]


def test_get_trap_config_with_new_schema(mock_page, mock_context, run_context):
    """Stage 4: 新スキーマ（navigation.trap.*）から設定を取得するテスト"""
    site_config = {
        "navigation": {
            "trap": {
                "detect_by_url": {
                    "patterns": ["/privacy", "/terms"],
                    "exact_matches": ["https://example.com/legal"],
                },
                "detect_by_selector": [".trap-page", "#legal-notice"],
                "recovery_actions": [
                    {"action": "go_back", "max_attempts": 2},
                    {"action": "goto_target", "target_url_key": "seed_plp_url", "max_attempts": 1},
                ],
            },
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_trap_config()
    
    # 新スキーマの値が取得できていることを確認
    assert "/privacy" in config.get("detect_by_url", {}).get("patterns", [])
    assert "/terms" in config.get("detect_by_url", {}).get("patterns", [])
    assert "https://example.com/legal" in config.get("detect_by_url", {}).get("exact_matches", [])
    assert ".trap-page" in config.get("detect_by_selector", [])
    assert "#legal-notice" in config.get("detect_by_selector", [])
    recovery_actions = config.get("recovery_actions", [])
    assert len(recovery_actions) == 2
    assert recovery_actions[0].get("action") == "go_back"
    assert recovery_actions[0].get("max_attempts") == 2


def test_get_trap_config_fallback_to_legacy_schema(mock_page, mock_context, run_context):
    """Stage 4: 旧スキーマ（navigation.trap_url_patterns）からフォールバックするテスト"""
    site_config = {
        "navigation": {
            "trap_url_patterns": ["/legal", "/terms"],
        },
    }
    
    driver = PlpDriver(
        page=mock_page,
        context=mock_context,
        site_config=site_config,
        run_context=run_context,
    )
    
    config = driver._get_trap_config()
    
    # 旧スキーマからフォールバックして値を取得できていることを確認
    patterns = config.get("detect_by_url", {}).get("patterns", [])
    assert "/legal" in patterns
    assert "/terms" in patterns
    # デフォルトの recovery_actions が含まれていることを確認
    recovery_actions = config.get("recovery_actions", [])
    assert len(recovery_actions) > 0
