# 2025年09月21日 19:49 JST
# ファイル名: ai_research_orchestrator.py
# レジストリ: app/ai_research_orchestrator.py
# バージョン: 9.0.0J (Enhanced Error Handling + Retry Logic)
#
# --- v9.0.0Jでの主な変更点 ---
# - [エラー処理強化] 各エージェント呼び出しにtry-exceptを追加
# - [リトライロジック] 一時的な失敗に対する自動リトライ
# - [サマリー改善] エラー詳細を含む実行サマリー
# - [Graceful Degradation] 部分的な失敗でも結果は返す

import logging
import json
import argparse
import asyncio
from pathlib import Path
import sys
from datetime import datetime
from typing import List, Optional, Dict, Any

# --- プロジェクトのルートパスをsys.pathに追加 ---
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

from app.config import loader, config
from app.agents.supplier_scout_agent import SupplierScoutAgent
from app.agents.persistence_agent import PersistenceAgent
from app.agents.reporting_agent import ReportingAgent
from app.models.result_models import DiscoveryResult


class OrchestratorError(Exception):
    """Orchestrator関連のカスタム例外"""
    pass


class AiResearchOrchestrator:
    """AIマルチエージェントシステム全体の調査プロセスを指揮する最高司令部。"""

    MAX_RETRIES = 2
    RETRY_DELAY_SECONDS = 3

    def __init__(self):
        config.setup_logging()
        self.sites_config = loader.load_full_config()
        # --- 後方支援部隊を配備 ---
        self.persistence_agent = PersistenceAgent()
        self.reporting_agent = ReportingAgent()
        self.run_stats = {
            "total_runs": 0,
            "successful": 0,
            "failed": 0,
            "retried": 0,
            "start_time": None,
            "end_time": None,
        }
        logging.info("最高司令部（Orchestrator）が起動しました。")

    async def run(self, **kwargs):
        """指定された調査任務を実行する。"""
        self.run_stats["total_runs"] += 1
        self.run_stats["start_time"] = datetime.now().isoformat()

        sites = kwargs.get("sites", [])
        query = kwargs.get("query", "")

        logging.info(f"===== 偵察任務開始: sites={sites}, query={query} =====")

        results = []
        last_error = None

        # リトライループ
        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                scout_agent = SupplierScoutAgent(runtime_kwargs=kwargs)
                results = await scout_agent.run(
                    sites_config=self.sites_config,
                    sites=sites,
                    query=query
                )
                # 成功
                if results:
                    break
            except Exception as e:
                last_error = e
                logging.error(f"偵察任務中にエラーが発生 (attempt {attempt}/{self.MAX_RETRIES}): {e}")
                if attempt < self.MAX_RETRIES:
                    self.run_stats["retried"] += 1
                    logging.info(f"{self.RETRY_DELAY_SECONDS}秒後にリトライします...")
                    await asyncio.sleep(self.RETRY_DELAY_SECONDS)
                else:
                    logging.error(f"最大リトライ回数に達しました。偵察任務を終了します。")

        # --- 作戦完了後の処理 ---
        if results:
            await self._process_successful_results(results)
        else:
            logging.warning("偵察任務から結果が得られませんでした。")

        # 最終サマリー生成
        self._generate_run_summary(results, last_error)

        self.run_stats["end_time"] = datetime.now().isoformat()
        self._log_final_results(results)

        return results

    async def _process_successful_results(self, results: List[Any]) -> None:
        """成功した結果を処理して永続化する"""
        try:
            successful_results = [
                r for r in results
                if isinstance(r, DiscoveryResult) and r.ok and r.evidence.get("extracted_data")
            ]

            if successful_results:
                logging.info(f"{len(successful_results)}件の成功した偵察結果を記録保管官に引き渡します。")
                records_to_persist = [res.evidence["extracted_data"] for res in successful_results]
                self.persistence_agent.snapshot_full_results(records_to_persist)
                self.run_stats["successful"] += len(successful_results)
            else:
                logging.info("成功した偵察結果はありませんでした。")

            logging.info("報告担当官に作戦サマリーの生成を命令します。")
            summary_report_path = self.reporting_agent.build_run_summary_report(results)
            logging.info(f"作戦サマリーレポートが生成されました: {summary_report_path}")

        except Exception as e:
            logging.error(f"結果処理中にエラーが発生: {e}")
            self.run_stats["failed"] += 1

    def _generate_run_summary(self, results: List[Any], last_error: Optional[Exception]) -> None:
        """実行サマリーを生成してログ出力"""
        try:
            total = len(results)
            successful = len([r for r in results if isinstance(r, DiscoveryResult) and r.ok])
            failed = total - successful

            summary = {
                "run_stats": self.run_stats,
                "total_results": total,
                "successful": successful,
                "failed": failed,
                "error": str(last_error) if last_error else None,
                "timestamp": datetime.now().isoformat()
            }

            logging.info("===== 偵察任務実行サマリー =====")
            logging.info(f"  総実行数: {total}")
            logging.info(f"  成功: {successful}")
            logging.info(f"  失敗: {failed}")
            logging.info(f"  リトライ回数: {self.run_stats.get('retried', 0)}")
            if last_error:
                logging.info(f"  最終エラー: {last_error}")
            logging.info("================================")

            # ファイルにも保存
            summary_path = APP_ROOT / "instance" / "runs" / f"run_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            logging.info(f"実行サマリーを保存しました: {summary_path}")

        except Exception as e:
            logging.error(f"サマリー生成中にエラー: {e}")

    def _log_final_results(self, results: list):
        """最終結果をログ出力"""
        if not results:
            logging.info("最終結果: 空の結果セット")
            return

        successful = sum(1 for r in results if isinstance(r, DiscoveryResult) and r.ok)
        logging.info(f"===== 偵察任務完了: {successful}/{len(results)}件成功 =====")

        for i, r in enumerate(results[:5]):  # 最初の5件だけ表示
            if isinstance(r, DiscoveryResult):
                status = "OK" if r.ok else "FAIL"
                name = r.evidence.get("product_name", "N/A") if r.evidence else "N/A"
                logging.info(f"  [{i+1}] {status}: {name}")

    def get_run_stats(self) -> Dict[str, Any]:
        """実行統計を返す"""
        return self.run_stats.copy()


def main():
    parser = argparse.ArgumentParser(description="AI Research Orchestrator")
    parser.add_argument("--site", dest="sites", required=True, action="append")
    parser.add_argument("--query", required=True)
    parser.add_argument("--discover-selectors", action="store_true")
    parser.add_argument("--headful", action="store_false", dest="headless")
    parser.add_argument("--slow-mo", type=int, default=0)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    # 最高司令部を初期化して作戦開始
    orchestrator = AiResearchOrchestrator()
    orchestrator.MAX_RETRIES = args.max_retries

    results = asyncio.run(orchestrator.run(
        sites=args.sites,
        query=args.query,
        discover_selectors=args.discover_selectors,
        headless=args.headless,
        slow_mo=args.slow_mo,
    ))

    # 統計を表示
    stats = orchestrator.get_run_stats()
    print(f"\n===== 作戦終了 =====")
    print(f"  総実行: {stats['total_runs']}")
    print(f"  成功: {stats['successful']}")
    print(f"  失敗: {stats['failed']}")
    print(f"  リトライ: {stats['retried']}")


if __name__ == "__main__":
    main()
