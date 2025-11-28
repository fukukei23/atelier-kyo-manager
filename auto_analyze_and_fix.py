#!/usr/bin/env python3
"""
自動ログ解析と修正提案システム
WSL環境でのログファイルを自動的に読み取り、問題を分析し、修正案を提案
"""
import re
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class LogAnalyzer:
    """ログファイルを解析して問題を特定"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    def find_latest_log(self) -> Optional[Path]:
        """最新のログファイルを検索"""
        log_files = list(self.project_root.glob("browser_test_*.log"))
        if not log_files:
            return None
        return max(log_files, key=lambda p: p.stat().st_mtime)
    
    def analyze(self, log_file: Path) -> Dict:
        """ログファイルを解析"""
        analysis = {
            "log_file": str(log_file),
            "timestamp": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
            "issues": [],
            "suggestions": [],
            "errors": []
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                content = ''.join(lines)
            
            # 問題1: PDP リンクが0件
            if re.search(r'Collected 0 PDP links', content):
                analysis["issues"].append({
                    "type": "pdp_links_zero",
                    "severity": "high",
                    "description": "PDP リンクが0件です",
                    "location": "NavigationDriver.collect_pdp_links"
                })
                
                # MonclerPLPStrategy ではタイルが見つかっているか確認
                if re.search(r'\[MonclerPLPStrategy\] Tile counts \(total=\d+\)', content):
                    analysis["suggestions"].append({
                        "type": "selector_mismatch",
                        "description": "MonclerPLPStrategy ではタイルが見つかっているが、collect_pdp_links で見つからない",
                        "action": "pdp_link_selectors に 'a[href*=\"/products/\"]' が含まれているか確認",
                        "file": "app/config/sites/overrides.local.json",
                        "priority": "high"
                    })
            
            # 問題2: Materialization 失敗
            if re.search(r'PLP materialization failed', content):
                analysis["issues"].append({
                    "type": "materialization_failed",
                    "severity": "high",
                    "description": "PLP materialization が失敗しています",
                    "location": "NavigationDriver.ensure_plp_materialized"
                })
                
                # タイル数が0の場合は、セレクタの問題の可能性
                if re.search(r'found 0 tiles', content):
                    analysis["suggestions"].append({
                        "type": "materialization_selector",
                        "description": "タイルが見つからないため、materialization のセレクタを確認",
                        "action": "ensure_plp_materialized の tile_selectors を確認",
                        "file": "app/agents/browser/navigation_driver.py",
                        "priority": "high"
                    })
            
            # 問題3: セレクタが見つからない
            selector_errors = re.findall(r"waiting for locator\(['\"]([^'\"]+)['\"]\)", content)
            if selector_errors:
                for selector in selector_errors:
                    analysis["issues"].append({
                        "type": "selector_not_found",
                        "severity": "medium",
                        "description": f"セレクタが見つかりません: {selector}",
                        "location": "NavigationDriver.click_first_card_or_link"
                    })
                    
                    # セレクタの修正案
                    if "data-qa='product-tile'" in selector:
                        analysis["suggestions"].append({
                            "type": "selector_fix",
                            "description": f"セレクタ '{selector}' が見つかりません",
                            "action": "click_first_card_or_link のセレクタを修正（タイムアウト処理を追加済み）",
                            "file": "app/agents/browser/navigation_driver.py",
                            "priority": "medium",
                            "fixed": True  # 既に修正済み
                        })
            
            # 問題4: タイムアウト
            if re.search(r'Timeout after \d+s', content):
                analysis["issues"].append({
                    "type": "timeout",
                    "severity": "high",
                    "description": "テストがタイムアウトしました",
                    "location": "tools/run_browser_use.py"
                })
            
            # エラーメッセージ
            error_lines = [line.strip() for line in lines if 'ERROR' in line]
            if error_lines:
                analysis["errors"] = error_lines[-10:]  # 最後の10行
            
        except Exception as e:
            analysis["error"] = f"ログ解析エラー: {e}"
        
        return analysis
    
    def generate_report(self, analysis: Dict) -> str:
        """解析結果をレポート形式で生成"""
        report = []
        report.append("=" * 80)
        report.append("ログ解析レポート")
        report.append("=" * 80)
        report.append(f"ログファイル: {analysis['log_file']}")
        report.append(f"タイムスタンプ: {analysis['timestamp']}")
        report.append()
        
        if analysis.get("issues"):
            report.append("--- 検出された問題 ---")
            for i, issue in enumerate(analysis["issues"], 1):
                report.append(f"{i}. [{issue['severity'].upper()}] {issue['type']}")
                report.append(f"   説明: {issue['description']}")
                report.append(f"   場所: {issue['location']}")
                report.append()
        
        if analysis.get("suggestions"):
            report.append("--- 修正提案 ---")
            for i, suggestion in enumerate(analysis["suggestions"], 1):
                status = "✓ 修正済み" if suggestion.get("fixed") else "要対応"
                report.append(f"{i}. [{status}] {suggestion['type']}")
                report.append(f"   説明: {suggestion['description']}")
                report.append(f"   アクション: {suggestion['action']}")
                report.append(f"   ファイル: {suggestion['file']}")
                report.append(f"   優先度: {suggestion['priority']}")
                report.append()
        
        if analysis.get("errors"):
            report.append("--- エラーメッセージ（最新10件） ---")
            for error in analysis["errors"]:
                report.append(f"  {error}")
            report.append()
        
        report.append("=" * 80)
        return "\n".join(report)

def main():
    """メイン処理"""
    project_root = Path(__file__).parent
    analyzer = LogAnalyzer(project_root)
    
    # 最新のログファイルを検索
    log_file = analyzer.find_latest_log()
    if not log_file:
        print("ログファイルが見つかりませんでした")
        return
    
    print(f"ログファイルを解析中: {log_file.name}")
    print()
    
    # 解析
    analysis = analyzer.analyze(log_file)
    
    # レポート生成
    report = analyzer.generate_report(analysis)
    print(report)
    
    # レポートをファイルに保存
    report_file = project_root / f"log_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_file.write_text(report, encoding='utf-8')
    print(f"\nレポートを保存しました: {report_file.name}")

if __name__ == "__main__":
    main()

