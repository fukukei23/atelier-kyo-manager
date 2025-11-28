#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonclerDrissionHandler の実行とレポート生成スクリプト

テストを実行し、結果を適切なフォルダに保存してレポートを生成します。
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def main():
    """MonclerDrissionHandler を実行してレポート生成"""
    
    # 結果保存ディレクトリ
    results_dir = project_root / "docs" / "test_results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = results_dir / f"moncler_drission_test_{timestamp}.log"
    json_file = results_dir / f"moncler_drission_test_{timestamp}.json"
    
    output_lines = []
    result_data = {
        "timestamp": datetime.now().isoformat(),
        "test_type": "MonclerDrissionHandler",
        "success": False,
        "error": None,
        "result": None,
    }
    
    def log(msg: str):
        """ログを出力とリストに追加"""
        print(msg)
        output_lines.append(msg)
    
    try:
        log("=" * 80)
        log("MonclerDrissionHandler 実行")
        log("=" * 80)
        log(f"実行日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        log("")
        
        # 1. DrissionPage が利用可能かチェック
        log("ステップ 1: DrissionPage のチェック")
        try:
            from DrissionPage import ChromiumPage
            log("  ✅ DrissionPage が利用可能です")
            result_data["drissionpage_available"] = True
        except ImportError as e:
            log(f"  ❌ DrissionPage がインストールされていません: {e}")
            log("  以下のコマンドでインストールしてください:")
            log("  pip install DrissionPage")
            log("")
            log("注意: Windows環境で実行する必要があります")
            result_data["drissionpage_available"] = False
            result_data["error"] = f"DrissionPage not installed: {e}"
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            return 1
        
        log("")
        
        # 2. RunContext を作成
        log("ステップ 2: RunContext の作成")
        try:
            from app.core.run_context import RunContext
            run_context = RunContext()
            log(f"  ✅ RunContext 作成完了 (run_id: {run_context.run_id})")
            result_data["run_id"] = run_context.run_id
        except Exception as e:
            log(f"  ❌ RunContext 作成エラー: {e}")
            result_data["error"] = f"RunContext creation failed: {e}"
            traceback.print_exc()
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            return 1
        
        log("")
        
        # 3. サイト設定を読み込む
        log("ステップ 3: サイト設定の読み込み")
        try:
            from app.config.loader import load_and_merge_configs
            sites_config = load_and_merge_configs()
            moncler_config = sites_config.get("MONCLER_OFFICIAL")
            
            if not moncler_config:
                log("  ⚠️  MONCLER_OFFICIAL の設定が見つかりません")
                log("  デフォルト設定を使用します")
                moncler_config = {
                    "base_url": "https://www.moncler.com",
                    "navigation": {
                        "header_search": {
                            "search_input_selector": "input[name='q'], input[type='search']",
                        }
                    },
                    "selectors": {
                        "plp": {
                            "pdp_link_selectors": [
                                "a[href*='/products/']",
                            ]
                        }
                    }
                }
                result_data["config_source"] = "default"
            else:
                log("  ✅ MONCLER_OFFICIAL の設定を読み込みました")
                result_data["config_source"] = "file"
        except Exception as e:
            log(f"  ⚠️  設定読み込みエラー: {e}")
            log("  デフォルト設定を使用します")
            moncler_config = {
                "base_url": "https://www.moncler.com",
                "navigation": {
                    "header_search": {
                        "search_input_selector": "input[name='q'], input[type='search']",
                    }
                },
                "selectors": {
                    "plp": {
                        "pdp_link_selectors": [
                            "a[href*='/products/']",
                        ]
                    }
                }
            }
            result_data["config_source"] = "default"
            result_data["config_error"] = str(e)
        
        log("")
        
        # 4. MonclerDrissionHandler を作成
        log("ステップ 4: MonclerDrissionHandler の作成")
        try:
            from app.specialized.moncler_handler import MonclerDrissionHandler
            
            handler = MonclerDrissionHandler(
                runtime_kwargs={
                    "headless": True,  # ヘッドレスモードで実行
                },
                user_data_path="user_data/moncler_profile_test",
            )
            log("  ✅ MonclerDrissionHandler 作成完了")
            result_data["handler_created"] = True
        except Exception as e:
            log(f"  ❌ MonclerDrissionHandler 作成エラー: {e}")
            result_data["error"] = f"Handler creation failed: {e}"
            traceback.print_exc()
            with open(log_file, "w", encoding="utf-8") as f:
                f.write("\n".join(output_lines))
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
            return 1
        
        log("")
        
        # 5. テスト実行
        log("ステップ 5: テスト実行")
        query = "down jacket"
        target_url = None  # 検索を使用
        
        log(f"  クエリ: {query}")
        log(f"  target_url: {target_url or '(検索を使用)'}")
        log("  テスト実行中...")
        log("")
        
        try:
            # 実行（同期関数として実行）
            result = handler.run(
                query=query,
                site_config=moncler_config,
                run_context=run_context,
                target_url=target_url,
            )
            
            log("")
            log("=" * 80)
            log("実行結果")
            log("=" * 80)
            log(f"成功: {result.ok}")
            log(f"メッセージ: {result.message}")
            log(f"サイト: {result.site}")
            log(f"クエリ: {result.query}")
            
            result_data["success"] = result.ok
            result_data["message"] = result.message
            result_data["result"] = {
                "ok": result.ok,
                "site": result.site,
                "query": result.query,
                "message": result.message,
            }
            
            if result.evidence and "extracted_data" in result.evidence:
                items = result.evidence["extracted_data"]
                log(f"取得した商品数: {len(items)}")
                result_data["result"]["items_count"] = len(items)
                result_data["result"]["items"] = items[:5]  # 最初の5件のみ保存
                
                if items:
                    log("")
                    log("取得した商品（最初の3件）:")
                    for i, item in enumerate(items[:3], 1):
                        log(f"  {i}. {item.get('title', 'N/A')}")
                        log(f"     価格: {item.get('price', 'N/A')}")
                        log(f"     URL: {item.get('url', 'N/A')}")
            
            log("")
            log("=" * 80)
            
            if result.ok:
                log("✅ テスト成功")
                return_code = 0
            else:
                log("❌ テスト失敗")
                return_code = 1
                
        except Exception as e:
            log("")
            log("=" * 80)
            log("❌ テスト実行エラー")
            log("=" * 80)
            log(f"エラー: {e}")
            result_data["error"] = str(e)
            result_data["error_type"] = type(e).__name__
            traceback.print_exc()
            return_code = 1
        
        # ログファイルに保存
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            if 'traceback' in locals():
                f.write("\n\n")
                f.write("=" * 80)
                f.write("\n詳細なトレースバック:\n")
                f.write("=" * 80)
                f.write("\n")
                traceback.print_exc(file=f)
        
        # JSON結果を保存
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        # Markdownレポートを生成
        report_file = results_dir / f"moncler_drission_test_{timestamp}.md"
        generate_markdown_report(result_data, report_file, json_file, log_file)
        
        log("")
        log("=" * 80)
        log("ファイル保存完了")
        log("=" * 80)
        log(f"ログファイル: {log_file.relative_to(project_root)}")
        log(f"JSON結果: {json_file.relative_to(project_root)}")
        log(f"レポート: {report_file.relative_to(project_root)}")
        log("=" * 80)
        
        return return_code
        
    except Exception as e:
        error_msg = f"予期しないエラー: {e}"
        log(error_msg)
        result_data["error"] = error_msg
        traceback.print_exc()
        
        with open(log_file, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            f.write("\n\n")
            f.write("=" * 80)
            f.write("\n詳細なトレースバック:\n")
            f.write("=" * 80)
            f.write("\n")
            traceback.print_exc(file=f)
        
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        return 1


def generate_markdown_report(result_data: dict, report_file: Path, json_file: Path, log_file: Path) -> None:
    """Markdown形式のレポートを生成"""
    lines = []
    
    lines.append("# MonclerDrissionHandler テスト実行レポート")
    lines.append("")
    lines.append(f"**実行日時**: {result_data.get('timestamp', 'N/A')}")
    lines.append("")
    
    # ステータス
    if result_data.get("success"):
        lines.append("## ✅ テスト成功")
    else:
        lines.append("## ❌ テスト失敗")
    lines.append("")
    
    # 実行結果
    lines.append("## 実行結果")
    lines.append("")
    
    if result_data.get("result"):
        result = result_data["result"]
        lines.append(f"- **サイト**: {result.get('site', 'N/A')}")
        lines.append(f"- **クエリ**: {result.get('query', 'N/A')}")
        lines.append(f"- **メッセージ**: {result.get('message', 'N/A')}")
        lines.append(f"- **商品数**: {result.get('items_count', 0)}")
        lines.append("")
        
        items = result.get("items", [])
        if items:
            lines.append("### 取得した商品")
            lines.append("")
            for i, item in enumerate(items, 1):
                lines.append(f"{i}. **{item.get('title', 'N/A')}**")
                lines.append(f"   - 価格: {item.get('price', 'N/A')}")
                lines.append(f"   - URL: {item.get('url', 'N/A')}")
                lines.append("")
    
    # エラー情報
    if result_data.get("error"):
        lines.append("## エラー情報")
        lines.append("")
        lines.append(f"```")
        lines.append(result_data["error"])
        lines.append("```")
        lines.append("")
    
    # 環境情報
    lines.append("## 環境情報")
    lines.append("")
    lines.append(f"- **DrissionPage**: {'利用可能' if result_data.get('drissionpage_available') else '未インストール'}")
    lines.append(f"- **設定ソース**: {result_data.get('config_source', 'N/A')}")
    lines.append(f"- **Run ID**: {result_data.get('run_id', 'N/A')}")
    lines.append("")
    
    # 詳細情報へのリンク
    lines.append("## 詳細情報")
    lines.append("")
    lines.append(f"- **ログファイル**: `{log_file.name}`")
    lines.append(f"- **JSON結果**: `{json_file.name}`")
    lines.append("")
    
    # 完了レポートへのリンク
    lines.append("## 関連ファイル")
    lines.append("")
    lines.append(f"詳細なログは `{log_file.name}` を確認してください。")
    lines.append(f"構造化された結果は `{json_file.name}` を確認してください。")
    lines.append("")
    
    # レポートを保存
    report_file.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())

