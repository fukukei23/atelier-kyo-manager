# ================================================================
#  フォルダ名: tests/
#  ファイル名: test_crawler_service.py
#  使い方:
#    この内容で既存のファイルを完全に上書きしてください。
#    withステートメントを活用し、より安全なテストを実行します。
# ================================================================
import json
import logging

from dotenv import load_dotenv

# --- ロギングの基本設定 ---
# INFOレベル以上のログをコンソールに出力
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# .envファイルから環境変数を読み込む
load_dotenv()

from app import create_app
from app.utils.ai_image_crawler import CrawlerService


def run_crawler_test():
    """CrawlerServiceの多段式AI検証機能の最終検証を行う"""
    print("\n" + "=" * 60)
    print("--- CrawlerService (Final Validation Mission) Test ---")
    print("=" * 60)

    app = create_app()
    with app.app_context():
        try:
            # --- ★★★ 最終検証シナリオ ★★★ ---
            # 現在Farfetchで実際に販売されている商品をターゲットとします。
            # エージェントが全ての検証をパスし、成功するのが期待される正しい動作です。
            test_url = "https://www.farfetch.com/jp/shopping/women/apc--grace-item-16960103.aspx"

            # 上記商品の「正解」の画像URL
            master_image_url = "https://cdn-images.farfetch-contents.com/16/96/01/03/16960103_34049399_1000.jpg"
            # ---------------------------

            print(
                f"\n1. 以下の情報を基に、多段式AI検証を開始します:\n   URL: {test_url}\n   Master Image: {master_image_url}\n"
            )

            # withステートメントでCrawlerServiceを安全に利用
            with CrawlerService() as crawler:
                print("   -> CrawlerServiceが正常に初期化されました。")
                product_info = crawler.get_product_info(test_url, master_image_url=master_image_url)

            print("\n" + "=" * 60)
            print("--- テスト完了 ---")
            if product_info and not product_info.get("error"):
                print("✅ [SUCCESS] 情報の取得に成功しました！\n")
                print("取得データ:")
                # 構造化ログ（メトリクス）と最終結果を分けて表示
                metrics = {}
                final_result = {}
                if "event" in product_info and product_info["event"] == "crawler_metrics":
                    metrics = product_info
                    # 本来の戻り値は別途ハンドリングが必要だが、このテストではログ出力で確認
                else:
                    final_result = product_info

                print(json.dumps(final_result or metrics, indent=2, ensure_ascii=False))

            else:
                print("❌ [FAILURE] 情報の取得に失敗しました。")
                print(f"   最終結果: {product_info.get('error', 'Unknown error')}")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"\n❌ [CRITICAL] テスト中に予期せぬクリティカルなエラーが発生しました: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # withブロックが終了すると自動的にcrawler.quit()が呼ばれるため、
            # ここでの明示的な終了処理は不要です。
            print("テストプロセスが正常に終了しました。")


if __name__ == "__main__":
    run_crawler_test()
