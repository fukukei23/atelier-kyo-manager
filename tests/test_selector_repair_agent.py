# -*- coding: utf-8 -*-
"""
test_selector_repair_agent.py
======================================================================
SelectorRepairAgent のユニットテスト
======================================================================
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


class MockRunContext:
    """RunContextのモック"""
    def __init__(self):
        self.run_id = "test_run_001"
        self.output_dir = APP_ROOT / "output"
        self.session_id = "test_session"

    async def take_screenshot(self, page, name):
        pass

    def save_json(self, filename, data):
        pass


@pytest.fixture
def mock_llm_controller():
    """モックLLMコントローラー"""
    mock = MagicMock()
    mock.generate = AsyncMock(return_value=MagicMock(
        text='{"proposed_selectors": ["div.product-item", "a[data-product-link]"], "rationale": "Test rationale"}'
    ))
    return mock


@pytest.fixture
def sample_html():
    """テスト用HTMLスニペット"""
    return """
    <html>
    <body>
        <div class="product-list">
            <div class="product-item" data-testid="product-card">
                <a href="/product/12345">Product 1</a>
                <span class="price">¥50,000</span>
            </div>
            <div class="product-item">
                <a href="/product/12346">Product 2</a>
                <span class="price">¥60,000</span>
            </div>
        </div>
    </body>
    </html>
    """


@pytest.fixture
def sample_site_config():
    """テスト用サイト設定"""
    return {
        "name": "TEST_SITE",
        "home_url": "https://example.com",
        "selectors": {
            "results_item": "div.product-item",
            "first_product_link": "a[href*='/product/']",
            "cookie_consent": "#onetrust-accept-btn-handler"
        }
    }


@pytest.mark.asyncio
async def test_propose_fix_with_mock_llm(mock_llm_controller, sample_html, sample_site_config):
    """LLMモックを使って propose_fix をテスト"""
    from app.agents.selector_repair_agent import SelectorRepairAgent

    agent = SelectorRepairAgent()
    agent.llm_controller = mock_llm_controller

    run_context = MockRunContext()

    result = await agent.propose_fix(
        intent="Find all product cards",
        failed_selectors=["article[data-testid='product-card']", ".product-card"],
        html_content=sample_html,
        site="TEST_SITE",
        site_config=sample_site_config,
        run_context=run_context
    )

    # 結果の検証
    assert result is not None
    assert "proposed_selectors" in result
    assert isinstance(result["proposed_selectors"], list)
    assert len(result["proposed_selectors"]) > 0
    assert "rationale" in result

    # モックが呼ばれたことを確認
    mock_llm_controller.generate.assert_called_once()


@pytest.mark.asyncio
async def test_propose_fix_with_fallback_when_llm_unavailable(sample_html, sample_site_config):
    """LLMコントローラーが利用できない場合、フォールバックを返す"""
    from app.agents.selector_repair_agent import SelectorRepairAgent

    agent = SelectorRepairAgent()
    agent.llm_controller = None  # 明示的にNoneに設定

    run_context = MockRunContext()

    result = await agent.propose_fix(
        intent="Find all product cards",
        failed_selectors=["article[data-testid='product-card']"],
        html_content=sample_html,
        site="TEST_SITE",
        site_config=sample_site_config,
        run_context=run_context
    )

    # フォールバックが返されることを確認
    assert result is not None
    assert result["proposed_selectors"] == []
    assert "unavailable" in result["rationale"].lower()


@pytest.mark.asyncio
async def test_propose_fix_handles_json_parse_error(mock_llm_controller, sample_html, sample_site_config):
    """JSON解析エラー時に適切に処理する"""
    from app.agents.selector_repair_agent import SelectorRepairAgent

    agent = SelectorRepairAgent()
    # 無効なJSONを返すモック
    agent.llm_controller = mock_llm_controller
    mock_llm_controller.generate = AsyncMock(return_value=MagicMock(text="invalid json {"))

    run_context = MockRunContext()

    result = await agent.propose_fix(
        intent="Find all product cards",
        failed_selectors=["article[data-testid='product-card']"],
        html_content=sample_html,
        site="TEST_SITE",
        site_config=sample_site_config,
        run_context=run_context
    )

    # エラー時にはフォールバックが返される
    assert result is not None
    assert "proposed_selectors" in result


@pytest.mark.asyncio
async def test_propose_fix_with_empty_failed_selectors(mock_llm_controller, sample_html, sample_site_config):
    """空のfailed_selectorsでも動作する"""
    from app.agents.selector_repair_agent import SelectorRepairAgent

    agent = SelectorRepairAgent()
    agent.llm_controller = mock_llm_controller

    run_context = MockRunContext()

    result = await agent.propose_fix(
        intent="Find all product cards",
        failed_selectors=[],  # 空リスト
        html_content=sample_html,
        site="TEST_SITE",
        site_config=sample_site_config,
        run_context=run_context
    )

    assert result is not None
    assert "proposed_selectors" in result


@pytest.mark.asyncio
async def test_propose_fix_extracts_selectors_from_response(mock_llm_controller, sample_html, sample_site_config):
    """LLMの応答からセレクタを正しく抽出する"""
    from app.agents.selector_repair_agent import SelectorRepairAgent

    agent = SelectorRepairAgent()
    agent.llm_controller = mock_llm_controller

    run_context = MockRunContext()

    result = await agent.propose_fix(
        intent="Find all product cards",
        failed_selectors=[".broken-selector"],
        html_content=sample_html,
        site="TEST_SITE",
        site_config=sample_site_config,
        run_context=run_context
    )

    # 提案されたセレクタを確認
    selectors = result.get("proposed_selectors", [])
    assert len(selectors) > 0
    # セレクタが文字列であることを確認
    for sel in selectors:
        assert isinstance(sel, str)
        assert len(sel) > 0
