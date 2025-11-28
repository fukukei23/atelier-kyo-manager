# -*- coding: utf-8 -*-
"""
NavigationDriver Stage 3A-2 の動作確認テスト

基本的なインポート、インスタンス化、静的メソッドの動作を確認します。
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from app.agents.browser.navigation_driver import (
    NavigationDriver,
    NavigationContext,
    NavigationOutcome,
)


def test_imports():
    """インポートの確認"""
    assert NavigationDriver is not None
    assert NavigationContext is not None
    assert NavigationOutcome is not None


def test_navigation_context():
    """NavigationContext の作成確認"""
    ctx = NavigationContext(
        site="MONCLER_OFFICIAL",
        query="GUCCI",
        site_config={"home_url": "https://www.moncler.com"},
        settings={"timeout_sec": 30},
        run_context=None,
        start_t=0.0,
        budget_ms=30000,
        entry_url="https://www.moncler.com/en-int/men/",
    )
    
    assert ctx.site == "MONCLER_OFFICIAL"
    assert ctx.query == "GUCCI"
    assert ctx.entry_url == "https://www.moncler.com/en-int/men/"


def test_navigation_outcome():
    """NavigationOutcome の作成確認"""
    outcome = NavigationOutcome(
        entry_url="https://www.moncler.com/en-int/men/",
        plp_materialized=True,
        trap_detected=False,
        trap_reason=None,
        recovered=False,
    )
    
    assert outcome.entry_url == "https://www.moncler.com/en-int/men/"
    assert outcome.plp_materialized is True
    assert outcome.trap_detected is False
    assert outcome.recovered is False


def test_looks_like_trap_or_legal():
    """静的メソッド looks_like_trap_or_legal のテスト"""
    # Trap URL のテスト
    assert NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-jp/") is True
    assert NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-int/") is True
    assert NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/cookie-policy") is True
    assert NavigationDriver.looks_like_trap_or_legal("https://www.monclergroup.com/") is True
    
    # 正常なURLのテスト
    assert NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-int/men/") is False
    assert NavigationDriver.looks_like_trap_or_legal("https://www.moncler.com/en-int/men/jackets/") is False


def test_navigation_driver_init():
    """NavigationDriver の初期化確認"""
    # モック Page オブジェクト
    mock_page = MagicMock()
    mock_page.url = "https://www.moncler.com/en-int/men/"
    mock_page.is_closed = MagicMock(return_value=False)
    
    # モック関数
    async def mock_ensure_plp_materialized(page, site_config, settings, start_t, budget_ms):
        return True
    
    # NavigationDriver のインスタンス化
    driver = NavigationDriver(
        page=mock_page,
        ensure_plp_materialized=mock_ensure_plp_materialized,
        recovery_fn=None,
        telemetry=None,
        strategy=None,
    )
    
    assert driver.page == mock_page
    assert driver.ensure_plp_materialized is not None
    assert driver.recovery_fn is None


@pytest.mark.asyncio
async def test_run_plp_flow_basic():
    """run_plp_flow の基本的な動作確認（モック使用）"""
    # モック Page オブジェクト
    mock_page = MagicMock()
    mock_page.url = "https://www.moncler.com/en-int/men/"
    mock_page.is_closed = MagicMock(return_value=False)
    
    # モック関数
    async def mock_ensure_plp_materialized(page, site_config, settings, start_t, budget_ms):
        return True
    
    # モック RunContext
    mock_run_context = MagicMock()
    mock_run_context.save_json = MagicMock()
    
    # NavigationDriver のインスタンス化
    driver = NavigationDriver(
        page=mock_page,
        ensure_plp_materialized=mock_ensure_plp_materialized,
        recovery_fn=None,
        telemetry=None,
        strategy=None,
    )
    
    # NavigationContext の作成
    ctx = NavigationContext(
        site="MONCLER_OFFICIAL",
        query="GUCCI",
        site_config={"home_url": "https://www.moncler.com"},
        settings={"timeout_sec": 30},
        run_context=mock_run_context,
        start_t=0.0,
        budget_ms=30000,
        entry_url="https://www.moncler.com/en-int/men/",
    )
    
    # run_plp_flow の実行
    outcome = await driver.run_plp_flow(ctx, target_url="https://www.moncler.com/en-int/men/")
    
    # 結果の確認
    assert outcome is not None
    assert outcome.entry_url == "https://www.moncler.com/en-int/men/"
    assert outcome.plp_materialized is True
    assert outcome.trap_detected is False


@pytest.mark.asyncio
async def test_run_plp_flow_with_trap():
    """run_plp_flow の trap 検出テスト"""
    # モック Page オブジェクト（trap URL）
    mock_page = MagicMock()
    mock_page.url = "https://www.moncler.com/en-jp/"
    mock_page.is_closed = MagicMock(return_value=False)
    
    # モック関数
    async def mock_ensure_plp_materialized(page, site_config, settings, start_t, budget_ms):
        return True
    
    # モック RunContext
    mock_run_context = MagicMock()
    mock_run_context.save_json = MagicMock()
    
    # モック recovery_fn
    async def mock_recovery_fn(page, site_config, target_url):
        mock_page.url = "https://www.moncler.com/en-int/men/"
    
    # NavigationDriver のインスタンス化
    driver = NavigationDriver(
        page=mock_page,
        ensure_plp_materialized=mock_ensure_plp_materialized,
        recovery_fn=mock_recovery_fn,
        telemetry=None,
        strategy=None,
    )
    
    # NavigationContext の作成
    ctx = NavigationContext(
        site="MONCLER_OFFICIAL",
        query="GUCCI",
        site_config={"home_url": "https://www.moncler.com"},
        settings={"timeout_sec": 30},
        run_context=mock_run_context,
        start_t=0.0,
        budget_ms=30000,
        entry_url="https://www.moncler.com/en-jp/",
    )
    
    # run_plp_flow の実行
    outcome = await driver.run_plp_flow(ctx, target_url="https://www.moncler.com/en-int/men/")
    
    # 結果の確認（recovery_fn が呼ばれて回復した場合）
    assert outcome is not None
    assert outcome.recovered is True  # recovery_fn が呼ばれた
    # trap が回復された場合、trap_detected は False になる
    assert outcome.trap_detected is False

