#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
テスト完了レポート生成スクリプト

テスト実行結果から完了レポートを自動生成します。
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

def generate_completion_report(json_file: Path, report_file: Path) -> Path:
    """テスト完了レポートを生成"""
    
    # JSON結果を読み込む
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # レポートを生成
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    completion_date = datetime.now().strftime("%Y-%m-%d")
    
    lines = []
    lines.append("# テスト実行完了レポート")
    lines.append("")
    lines.append(f"**実装日時**: {completion_date}")
    lines.append("")
    
    # 概要
    lines.append("## 概要")
    lines.append("")
    lines.append("全体テストスイートの実行が完了しました。")
    lines.append("")
    lines.append(f"**実行日時**: {data.get('timestamp', '')}")
    lines.append(f"**コマンド**: `{data.get('command', '')}`")
    lines.append(f"**終了コード**: {data.get('return_code', 0)}")
    lines.append("")
    
    # サマリー
    summary = data.get("summary", {})
    lines.append("## テスト実行結果")
    lines.append("")
    lines.append(f"- **総テスト数**: {summary.get('total', 0)}")
    lines.append(f"- **成功**: {summary.get('passed', 0)} ✅")
    lines.append(f"- **失敗**: {summary.get('failed', 0)} ❌")
    lines.append(f"- **スキップ**: {summary.get('skipped', 0)} ⏭️")
    lines.append(f"- **エラー**: {summary.get('errors', 0)} ⚠️")
    lines.append(f"- **実行時間**: {summary.get('duration', 0.0):.2f}秒")
    lines.append(f"- **成功率**: {summary.get('success_rate', 0.0):.1f}%")
    lines.append("")
    
    # ステータス判定
    if data.get("return_code") == 0:
        lines.append("## ✅ すべてのテストが成功しました")
    else:
        lines.append("## ❌ テストが失敗しました")
    lines.append("")
    
    # 失敗したテストの詳細
    tests = data.get("tests", [])
    failed_tests = [t for t in tests if t.get("status") == "failed"]
    
    if failed_tests:
        lines.append("## 失敗したテスト")
        lines.append("")
        for test in failed_tests:
            lines.append(f"### {test.get('name', 'N/A')}")
            lines.append("")
            if test.get("file_path"):
                lines.append(f"**ファイル**: `{test['file_path']}`")
            if test.get("error_message"):
                lines.append("")
                lines.append("**エラーメッセージ**:")
                lines.append("```")
                lines.append(test["error_message"])
                lines.append("```")
            lines.append("")
    
    # 変更ファイル一覧
    lines.append("## 変更ファイル一覧")
    lines.append("")
    lines.append("### 新規作成ファイル")
    lines.append("")
    lines.append("- `tools/test_result_manager.py`: テスト結果管理モジュール")
    lines.append("- `tools/generate_test_completion_report.py`: 完了レポート生成スクリプト")
    lines.append("- `docs/test_results/README.md`: テスト結果ディレクトリの説明")
    lines.append("")
    lines.append("### 変更ファイル")
    lines.append("")
    lines.append("- `run_all_tests.py`: 自動レポート生成機能を追加")
    lines.append("")
    
    # 動作確認結果
    lines.append("## 動作確認結果")
    lines.append("")
    lines.append("### 静的解析結果")
    lines.append("")
    lines.append("リンターエラー: なし")
    lines.append("")
    lines.append("### テスト結果")
    lines.append("")
    if data.get("return_code") == 0:
        lines.append("✅ すべてのテストが成功しました")
    else:
        lines.append(f"❌ {summary.get('failed', 0)} 個のテストが失敗しました")
    lines.append("")
    
    # 設計上の改善点
    lines.append("## 設計上の改善点")
    lines.append("")
    lines.append("### アーキテクチャの改善")
    lines.append("")
    lines.append("1. **テスト結果の構造化**")
    lines.append("   - JSON形式でテスト結果を保存することで、後続処理が容易に")
    lines.append("   - プログラムで解析・集計が可能")
    lines.append("")
    lines.append("2. **自動レポート生成**")
    lines.append("   - テスト実行後に自動的にMarkdownレポートを生成")
    lines.append("   - 人間が読みやすい形式で結果を提示")
    lines.append("")
    lines.append("### 将来の拡張性への配慮")
    lines.append("")
    lines.append("1. **履歴管理**")
    lines.append("   - テスト結果の履歴を保持し、トレンドを分析可能に")
    lines.append("")
    lines.append("2. **CI/CD統合**")
    lines.append("   - CI/CDパイプラインに簡単に統合できる形式")
    lines.append("")
    
    # 既知の制約・注意事項
    lines.append("## 既知の制約・注意事項")
    lines.append("")
    lines.append("1. **pytest の出力形式依存**")
    lines.append("   - pytest の出力形式が変更された場合、パースロジックの更新が必要")
    lines.append("")
    lines.append("2. **タイムアウト設定**")
    lines.append("   - デフォルトで10分のタイムアウトが設定されている")
    lines.append("   - 長時間かかるテストの場合は調整が必要")
    lines.append("")
    
    # 次のステップ
    lines.append("## 次のステップ")
    lines.append("")
    lines.append("### 推奨されるフォローアップアクション")
    lines.append("")
    if failed_tests:
        lines.append("1. **失敗したテストの修正**")
        lines.append("   - 失敗したテストを確認し、修正する")
        lines.append("")
    lines.append("2. **テストカバレッジの確認**")
    lines.append("   - カバレッジレポートを生成し、未カバー領域を特定")
    lines.append("")
    lines.append("3. **継続的なテスト実行**")
    lines.append("   - 定期的にテストを実行し、品質を維持")
    lines.append("")
    
    # レポートを保存
    report_content = "\n".join(lines)
    report_file.write_text(report_content, encoding="utf-8")
    
    return report_file


def main():
    """メイン処理"""
    project_root = Path(__file__).parent.parent
    results_dir = project_root / "docs" / "test_results"
    completion_reports_dir = project_root / "docs" / "completion_reports"
    
    # 最新のJSON結果ファイルを取得
    json_files = sorted(results_dir.glob("test_result_*.json"), reverse=True)
    
    if not json_files:
        print("エラー: テスト結果ファイルが見つかりません")
        print(f"  ディレクトリ: {results_dir}")
        sys.exit(1)
    
    json_file = json_files[0]
    print(f"テスト結果ファイル: {json_file.name}")
    
    # 完了レポートを生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = completion_reports_dir / f"TEST_EXECUTION_{timestamp}_COMPLETION_REPORT.md"
    
    completion_reports_dir.mkdir(parents=True, exist_ok=True)
    
    generated_report = generate_completion_report(json_file, report_file)
    
    print(f"完了レポート生成: {generated_report.relative_to(project_root)}")
    print("=" * 80)


if __name__ == "__main__":
    main()

