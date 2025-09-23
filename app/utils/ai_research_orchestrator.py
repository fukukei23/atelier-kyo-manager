# 2025年09月21日 19:49 JST
# ファイル名: ai_research_orchestrator.py
# レジストリ: app/ai_research_orchestrator.py
# バージョン: 8.0.0J (Full Intelligence Cycle)
#
# --- v8.0.0Jでの主な変更点 ---
# - [最終統合] 偵察任務完了後、記録保管官(PersistenceAgent)と
#   報告担当官(ReportingAgent)を呼び出すロジックを追加。
# - [インテリジェンス・サイクル] これにより、「偵察→報告→記録→分析」
#   という、自己完結した完全な情報サイクルが確立されました。

import logging
import json
import argparse
import asyncio
from pathlib import Path
import sys

# --- プロジェクトのルートパスをsys.pathに追加 ---
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import loader, config
from app.agents.supplier_scout_agent import SupplierScoutAgent
from app.agents.persistence_agent import PersistenceAgent
from app.agents.reporting_agent import ReportingAgent
from app.models.result_models import DiscoveryResult

class AiResearchOrchestrator:
    """AIマルチエージェントシステム全体の調査プロセスを指揮する最高司令部。"""
    def __init__(self):
        config.setup_logging()
        self.sites_config = loader.load_full_config()
        # --- ★★★ 後方支援部隊を配備 ★★★ ---
        self.persistence_agent = PersistenceAgent()
        self.reporting_agent = ReportingAgent()
        logging.info("最高司令部（Orchestrator）が起動しました。")

    async def run(self, **kwargs):
        """指定された調査任務を実行する。"""
        # ... (作戦開始までのロジックは変更なし)
        scout_agent = SupplierScoutAgent(runtime_kwargs=kwargs)
        results = await scout_agent.run(
            sites_config=self.sites_config,
            sites=kwargs.get("sites", []),
            query=kwargs.get("query", "")
        )

        # --- ★★★ 作戦完了後の処理を追加 ★★★ ---
        successful_results = [r for r in results if isinstance(r, DiscoveryResult) and r.ok and r.evidence.get("extracted_data")]

        if successful_results:
            logging.info(f"{len(successful_results)}件の成功した偵察結果を記録保管官に引き渡します。")
            # データを永続化に適した形式に変換
            records_to_persist = [res.evidence["extracted_data"] for res in successful_results]
            self.persistence_agent.snapshot_full_results(records_to_persist)

        logging.info("報告担当官に作戦サマリーの生成を命令します。")
        summary_report_path = self.reporting_agent.build_run_summary_report(results)
        logging.info(f"作戦サマリーレポートが生成されました: {summary_report_path}")

        logging.info("全作戦が完了しました。最終結果を報告します。")
        self._log_final_results(results)

        return results

    def _log_final_results(self, results: list):
        # ... (最終報告ロジックは変更なし)
        pass

def main():
    # ... (コマンドライン引数の定義は変更なし)
    parser = argparse.ArgumentParser(description="AI Research Orchestrator")
    parser.add_argument("--site", dest="sites", required=True, action="append")
    parser.add_argument("--query", required=True)
    parser.add_argument("--discover-selectors", action="store_true")
    parser.add_argument("--headful", action="store_false", dest="headless")
    parser.add_argument("--slow-mo", type=int, default=0)

    args = parser.parse_args()
    runtime_kwargs = vars(args)

    orchestrator = AiResearchOrchestrator()
    asyncio.run(orchestrator.run(**runtime_kwargs))

if __name__ == "__main__":
    main()
