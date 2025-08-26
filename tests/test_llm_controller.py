# ----------------------------------------------------------------
#  ファイル名: test_llm_controller.py (最終修正版)
#  役割:
#    AILlmControllerがDeepSeekと正常に通信できるかをテストする。
# ----------------------------------------------------------------
import os
import sys
from dotenv import load_dotenv

# --- ↓↓ ここが最重要修正点 ↓↓ ---
# テストスクリプトが起動した時点で、.envファイルを読み込む
load_dotenv()
# --- ↑↑ ここまでが最重要修正点 ↑↑ ---

# プロジェクトのルートディレクトリをPythonのパスに追加
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import create_app
from app.utils.ai_llm_controller import AILlmController

def test_deepseek_connection():
    """DeepSeekへの接続と応答をテストする関数"""
    print("--- DeepSeek 接続テスト開始 ---")

    app = create_app()
    with app.app_context():
        try:
            print("1. AILlmControllerを初期化します...")
            controller = AILlmController()
            # 初期化時の診断メッセージは、成功すれば不要になるので削除済み
            print("   -> 初期化完了。")

            prompt = "自己紹介をしてください。あなたは誰で、何ができますか？"
            print(f"2. DeepSeekに送信するプロンプト:\n   「{prompt}」")

            print("3. DeepSeekにリクエストを送信中...")
            response = controller.generate(prompt, model_name='deepseek')

            if response.startswith("エラー:"):
                print("\n--- テスト結果 ---")
                print("❌ テストに失敗しました。コントローラーからのエラーメッセージ:")
                print(response)
            else:
                print("\n--- テスト結果 ---")
                print("✅ DeepSeekからの応答受信に成功しました！\n")
                print("応答内容:")
                print("-" * 20)
                print(response)
                print("-" * 20)

        except Exception as e:
            print("\n--- テスト結果 ---")
            print(f"❌ 予期せぬエラーが発生しました: {e}")
            print("   APIキーの有効期限やネットワーク設定を確認してください。")

if __name__ == '__main__':
    test_deepseek_connection()
