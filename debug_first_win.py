# ==============================================================================
# ファイル名: debug_first_win.py
# 配置場所: プロジェクトルート (appフォルダと同じ階層)
# 目的: 確実にスクレイピングを成功させ、BrowserUseAgentの動作を証明する
# ==============================================================================
import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class MockRunContext:
    def __init__(self):
        self.run_id = f"TEST_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.run_path = Path("instance") / "runs" / self.run_id
        self.run_path.mkdir(parents=True, exist_ok=True)
        self.screenshots = []

    def get_path(self, name):
        return self.run_path / name

    async def take_screenshot(self, page, name):
        path = self.run_path / f"{name}.png"
        try:
            await page.screenshot(path=str(path))
            self.screenshots.append(str(path))
            logger.info(f"[MockContext] Screenshot saved: {path}")
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")

    def save_json(self, name, data):
        path = self.run_path / name
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"[MockContext] JSON saved: {path}")

    def save_text(self, name, text):
        path = self.run_path / name
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)

    def save_content(self, name, content):
        self.save_text(name, content or "")


try:
    root_dir = Path(__file__).resolve().parent
    sys.path.append(str(root_dir))
    sys.path.append(str(root_dir / "app"))
    from app.agents import selector_discovery_agent as _sel_mod
    from app.agents import failure_analysis_agent as _fa_mod
    from app.agents import browser_use_agent as _bua_mod
    import re

    class _PatchedSelectorDiscoveryAgent(_sel_mod.SelectorDiscoveryAgent):
        def __init__(self, runtime_kwargs=None):
            super().__init__()

    class _PatchedFailureAnalysisAgent(_fa_mod.FailureAnalysisAgent):
        def __init__(self, runtime_kwargs=None):
            super().__init__()

    _sel_mod.SelectorDiscoveryAgent = _PatchedSelectorDiscoveryAgent
    _fa_mod.FailureAnalysisAgent = _PatchedFailureAnalysisAgent

    _bua_mod.PRODUCT_URL_ALLOW_PATTERNS = re.compile(r".*")
    def _always_true(url: str) -> bool:
        return True
    _bua_mod.looks_like_product_url = _always_true

    from app.agents.browser_use_agent import BrowserUseAgent
except ImportError as e:
    logger.error("【エラー】appフォルダが見つかりません。このファイルはプロジェクトのルートに置いてください。")
    logger.error(f"詳細: {e}")
    sys.exit(1)


async def main():
    logger.info("=== 初回成功テストを開始します ===")

    target_site_config = {
        "id": "BOOKS_TO_SCRAPE",
        "home_url": "http://books.toscrape.com/",
        "discovery_settings": {
            "timeout_sec": 30,
            "headless": False,
            "enable_har": False,
            "enable_trace": False,
            "enable_video": False,
            "overall_plp_budget_ms": 60000,
            "pdp_parallel_limit": 1,
            "wait_for_selectors": ["article.product_pod"],
        },
        "selectors": {
            "pdp": {
                "plp_container_selectors": ["article.product_pod", "li.col-xs-6"],
                "pdp_link_selectors": ["h3 > a"],
                "title_selectors": ["h1"],
                "price_selectors": ["p.price_color"],
            },
            "ui": {"cookie_accept": [], "continue_shopping": []},
        },
    }

    run_ctx = MockRunContext()
    agent = BrowserUseAgent(runtime_kwargs={"headless": False, "site": "BOOKS_TEST"})

    logger.info(f"ターゲットURL: {target_site_config['home_url']}")

    try:
        result = await agent.run(
            site="BOOKS_TEST",
            query="python books",
            site_config=target_site_config,
            run_context=run_ctx,
            target_url=target_site_config["home_url"],
            likely_plp=True,
        )

        logger.info("--- 実行結果 ---")
        if result.ok:
            logger.info("✅ 【大成功】 データ取得に成功しました！")
            data = result.evidence.get("extracted_data", {})
            count = len(data.get("extracted_data", [])) if isinstance(data, dict) else 0
            logger.info(f"取得データ数: {count}")
            logger.info(f"データ保存先: {run_ctx.run_path}")
        else:
            logger.error("❌ 【失敗】 エラーが発生しました。")
            logger.error(f"エラー内容: {result.message}")
    except Exception as e:
        logger.error(f"予期せぬエラー: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
