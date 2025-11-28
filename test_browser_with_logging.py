#!/usr/bin/env python3
"""
実ブラウザテスト実行スクリプト（ログファイル出力付き）
"""
import asyncio
import sys
import logging
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# ログファイルの設定
log_file = project_root / f"browser_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def run_test():
    """BrowserUseAgent の実ブラウザテストを実行"""
    try:
        logger.info("=" * 80)
        logger.info("実ブラウザテスト開始: MONCLER_OFFICIAL")
        logger.info(f"ログファイル: {log_file}")
        logger.info("=" * 80)
        
        from app.agents.browser_use_agent import BrowserUseAgent
        from app.core.run_context import RunContext
        
        site = "MONCLER_OFFICIAL"
        url = "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB"
        query = "down jacket"
        
        # RunContext を作成
        run_context = RunContext(
            site=site,
            query=query,
            run_id=f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # BrowserUseAgent を初期化
        agent = BrowserUseAgent(
            site=site,
            query=query,
            runtime_kwargs={
                "headful": True,
                "timeout": 120,
                "use_proxy": True
            }
        )
        
        logger.info(f"サイト: {site}")
        logger.info(f"URL: {url}")
        logger.info(f"クエリ: {query}")
        logger.info("BrowserUseAgent を実行中...")
        
        # 実行
        result = await agent.run(
            target_url=url,
            run_context=run_context
        )
        
        logger.info("=" * 80)
        logger.info("テスト完了")
        logger.info(f"結果: ok={result.ok if hasattr(result, 'ok') else 'N/A'}")
        if hasattr(result, 'proposal') and result.proposal:
            logger.info(f"提案数: {len(result.proposal) if isinstance(result.proposal, (list, dict)) else 'N/A'}")
        logger.info(f"ログファイル: {log_file}")
        logger.info("=" * 80)
        
        return result
        
    except KeyboardInterrupt:
        logger.warning("\n[中断] ユーザーによる中断")
        raise
    except Exception as e:
        logger.error(f"\n[エラー] {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        result = asyncio.run(run_test())
        print(f"\n✓ テスト完了。ログファイル: {log_file}")
        sys.exit(0)
    except KeyboardInterrupt:
        print(f"\n[中断] ログファイル: {log_file}")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ エラー: {e}")
        print(f"ログファイル: {log_file}")
        sys.exit(1)

