#!/usr/bin/env python3
"""
MONCLER Drissionルートの自動診断スクリプト

このスクリプトは、MonclerDrissionHandler を単体で実行し、
成功・失敗それぞれについてスクリーンショット、HTML、JSON、ログを保存します。

使用方法:
    python scripts/run_moncler_drission_diagnostics.py \
      --query "down jacket" \
      --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \
      --headless

出力先:
    artifacts/moncler_drission/YYYYMMDD_HHMMSS/
      - success_plp.html / .png / .json (成功時)
      - error_plp.html / .png / .json (失敗時)
      - run.log (ログファイル)
"""

import argparse
import logging
import sys
import time
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 既存のインポートパスに合わせる
try:
    from app.config.loader import get_site_config
    from app.core.run_context import RunContext
    from app.specialized.moncler_handler import MonclerDrissionHandler
except ImportError as e:
    print(f"❌ インポートエラー: {e}")
    print("プロジェクトルートから実行してください。")
    sys.exit(1)


def setup_logger(out_dir: Path) -> logging.Logger:
    """
    ロガーを設定

    Args:
        out_dir: ログファイルを保存するディレクトリ

    Returns:
        logging.Logger: 設定済みロガー
    """
    log_path = out_dir / "run.log"

    # ロガーを設定
    logger = logging.getLogger("moncler_drission_diagnostics")
    logger.setLevel(logging.DEBUG)

    # 既存のハンドラをクリア
    logger.handlers.clear()

    # ファイルハンドラ
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # コンソールハンドラ
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def load_site_config() -> dict:
    """
    MONCLER_OFFICIAL のサイト設定を読み込む

    Returns:
        dict: サイト設定
    """
    site_config = get_site_config("MONCLER_OFFICIAL")
    if not site_config:
        raise ValueError("MONCLER_OFFICIAL のサイト設定が見つかりません")
    return site_config


def main() -> int:
    """メイン処理"""
    parser = argparse.ArgumentParser(
        description="Run MONCLER Drission diagnostics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  python scripts/run_moncler_drission_diagnostics.py \\
    --query "down jacket" \\
    --target_url "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB" \\
    --headless

  python scripts/run_moncler_drission_diagnostics.py \\
    --query "jacket" \\
    --runs 3 \\
    --out_base "artifacts/moncler_test"
        """,
    )

    parser.add_argument("--query", default="down jacket", help="検索クエリ（デフォルト: 'down jacket'）")
    parser.add_argument("--target_url", default=None, help="直接指定する PLP URL（オプション）")
    parser.add_argument("--headless", action="store_true", help="ヘッドレスモードで実行（デフォルト: False）")
    parser.add_argument("--runs", type=int, default=1, help="実行回数（デフォルト: 1）")
    parser.add_argument(
        "--out_base",
        default="artifacts/moncler_drission",
        help="出力ベースディレクトリ（デフォルト: 'artifacts/moncler_drission'）",
    )

    args = parser.parse_args()

    # 出力ベースディレクトリを作成
    base_dir = Path(args.out_base)
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MONCLER Drission 診断スクリプト")
    print("=" * 80)
    print(f"クエリ: {args.query}")
    print(f"ターゲットURL: {args.target_url or '(検索を使用)'}")
    print(f"ヘッドレス: {args.headless}")
    print(f"実行回数: {args.runs}")
    print(f"出力先: {base_dir}")
    print("=" * 80)
    print()

    # サイト設定を読み込む
    try:
        site_config = load_site_config()
        print("✅ サイト設定を読み込みました")
    except Exception as e:
        print(f"❌ サイト設定の読み込みに失敗: {e}")
        return 1

    # 各実行を処理
    success_count = 0
    failure_count = 0

    for i in range(args.runs):
        run_id = time.strftime("%Y%m%d_%H%M%S")
        if args.runs > 1:
            run_id = f"{run_id}_{i + 1:03d}"

        out_dir = base_dir / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # ロガーを設定
        logger = setup_logger(out_dir)

        print(f"\n[実行 {i + 1}/{args.runs}] Run ID: {run_id}")
        print(f"出力先: {out_dir}")

        # RunContext を初期化
        try:
            run_context = RunContext()
            # 診断用に run_path を設定
            if hasattr(run_context, "run_path"):
                run_context.run_path = str(out_dir)
        except Exception as e:
            logger.error(f"RunContext の初期化に失敗: {e}")
            failure_count += 1
            continue

        # MonclerDrissionHandler を初期化
        try:
            handler = MonclerDrissionHandler(
                runtime_kwargs={
                    "headless": args.headless,
                    "diagnostics_dir": str(base_dir),
                    "run_id": run_id,
                },
                debug=True,  # 診断モードを有効化
            )
        except Exception as e:
            logger.error(f"MonclerDrissionHandler の初期化に失敗: {e}")
            failure_count += 1
            continue

        # 実行
        try:
            logger.info(f"処理開始: query={args.query}, target_url={args.target_url}")

            result = handler.run(
                query=args.query,
                site_config=site_config,
                run_context=run_context,
                target_url=args.target_url,
            )

            if result.ok:
                logger.info(f"✅ 成功: {result.message}")
                print(f"  ✅ 成功: {result.message}")
                success_count += 1
            else:
                logger.warning(f"⚠️  失敗: {result.message}")
                print(f"  ⚠️  失敗: {result.message}")
                failure_count += 1

            # 結果の詳細をログに記録
            if hasattr(result, "evidence") and result.evidence:
                logger.debug(f"取得データ: {result.evidence}")

        except Exception as e:
            logger.exception(f"❌ 未処理例外: {e}")
            print(f"  ❌ 未処理例外: {e}")
            failure_count += 1

        print(f"  診断ファイル: {out_dir}")

    # サマリー
    print()
    print("=" * 80)
    print("実行サマリー")
    print("=" * 80)
    print(f"成功: {success_count} 回")
    print(f"失敗: {failure_count} 回")
    print(f"合計: {args.runs} 回")
    print("=" * 80)

    return 0 if failure_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
