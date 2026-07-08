# ----------------------------------------------------------------
#  ファイル名: test_llm_controller.py (最終修正版)
#  役割:
#    AILlmControllerがDeepSeekと正常に通信できるかをテストする。
# ----------------------------------------------------------------

import pytest
from dotenv import load_dotenv

# --- ↓↓ ここが最重要修正点 ↓↓ ---
# テストスクリプトが起動した時点で、.envファイルを読み込む
load_dotenv()
# --- ↑↑ ここまでが最重要修正点 ↑↑ ---

from app import create_app
from app.utils.ai_llm_controller import AILlmController


@pytest.mark.skip(reason="product版ではWeb機能(create_app())は除外されています")
def test_deepseek_connection():
    """DeepSeekへの接続と応答をテストする関数"""
    # 実際のLLMクライアントが1つも利用可能でない場合はスキップ
    try:
        app = create_app()
        with app.app_context():
            controller = AILlmController()
            has_client = bool(
                controller.gemini_model
                or getattr(controller, "deepseek_client", None)
                or getattr(controller, "openai_client", None)
            )
            if not has_client:
                pytest.skip("No LLM client configured (GEMINI_API_KEY, DEEPSEEK_API_KEY, or OPENAI_API_KEY required)")
    except Exception:
        pytest.skip("LLM controller initialization failed - API keys may be missing")

    print("--- DeepSeek 接続テスト開始 ---")

    app = create_app()
    with app.app_context():
        controller = AILlmController()
        print("   -> 初期化完了。")

        prompt = "自己紹介をしてください。あなたは誰で、何ができますか？"
        print(f"2. DeepSeekに送信するプロンプト:\n   「{prompt}」")

        print("3. DeepSeekにリクエストを送信中...")
        response = controller.generate(prompt, task_type="deepseek")

        # DeepSeekからの応答を検証
        assert not response.startswith("エラー:"), f"DeepSeek connection failed: {response}"
        assert len(response) > 0, "DeepSeek returned empty response"
        assert "DeepSeek" in response or "deepseek" in response.lower() or len(response) > 10, (
            f"DeepSeek returned unexpected response: {response[:100]}"
        )

        print("\n--- テスト結果 ---")
        print("✅ DeepSeekからの応答受信に成功しました！\n")
        print("応答内容:")
        print("-" * 20)
        print(response[:500] + "..." if len(response) > 500 else response)
        print("-" * 20)


if __name__ == "__main__":
    test_deepseek_connection()
