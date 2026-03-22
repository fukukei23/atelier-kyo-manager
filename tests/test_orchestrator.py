# ----------------------------------------------------------------
#  ファイル名: test_orchestrator.py (シングルセッション・最終版)
#  役割:
#    司令塔の全機能を、単一のブラウザセッションでテストする。
# ----------------------------------------------------------------
import os
import sys
import json
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.utils.ai_research_orchestrator import AiResearchOrchestrator

def run_full_test():
    """シングルセッションで全ワークフローをテストする"""
    print("--- AI Research Orchestrator 統合テスト開始 ---")

    app = create_app()
    with app.app_context():
        try:
            # --- ★重要★ ---
            # 常に headless=False で実行し、人間による介入を可能にする
            print("1. ResearchOrchestratorを初期化します... (ブラウザ表示モード)")
            orchestrator = AiResearchOrchestrator(headless=True)
            print("   -> 初期化完了。")

            brand_to_test = "PRADA"
            print(f"2. テスト対象ブランド: {brand_to_test}")

            print("3. run()メソッドを実行し、リサーチワークフローを開始します...")
            results = orchestrator.run(brand_name=brand_to_test)

            print("\n" + "="*50)
            print("--- テスト完了 ---")
            if results:
                print(f"✅ リサーチ成功！ 利益が見込める商品が {len(results)} 件見つかりました！\n")
                print(json.dumps(results, indent=2, ensure_ascii=False))
            else:
                print("✅ リサーチは正常に完了しましたが、利益基準を満たす商品はありませんでした。")
            print("="*50)
        except Exception as e:
            print(f"\n❌ テスト中にエラー: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    run_full_test()
