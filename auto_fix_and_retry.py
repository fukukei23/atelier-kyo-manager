#!/usr/bin/env python3
"""
完全自動修正と再実行ループ
WSL環境でのログファイルを自動的に読み取り、問題を分析し、修正を適用して再実行
"""
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

class AutoFixAndRetry:
    """自動修正と再実行のループ"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.max_retries = 3
        self.retry_count = 0
        self.fixes_applied = []
        
    def find_latest_log(self) -> Optional[Path]:
        """最新のログファイルを検索"""
        log_files = list(self.project_root.glob("browser_test_*.log"))
        if not log_files:
            return None
        return max(log_files, key=lambda p: p.stat().st_mtime)
    
    def parse_log_errors(self, log_file: Path) -> Dict[str, any]:
        """ログファイルからエラーを解析"""
        errors = {
            "pdp_links_zero": False,
            "materialization_failed": False,
            "timeout": False,
            "selector_not_found": [],
            "error_messages": [],
            "moncler_tiles_found": False,
            "tile_count": 0
        }
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = f.readlines() if hasattr(f, 'readlines') else content.split('\n')
            
            # PDP リンクが0件
            if re.search(r'Collected 0 PDP links', content):
                errors["pdp_links_zero"] = True
                
            # Materialization 失敗
            if re.search(r'PLP materialization failed', content):
                errors["materialization_failed"] = True
                
            # タイムアウト
            if re.search(r'Timeout after \d+s', content):
                errors["timeout"] = True
                
            # セレクタが見つからない
            selector_errors = re.findall(r"waiting for locator\(['\"]([^'\"]+)['\"]\)", content)
            if selector_errors:
                errors["selector_not_found"] = selector_errors
                
            # MonclerPLPStrategy でタイルが見つかっているか
            # 最後のマッチを取得（または total > 0 の最初のマッチを優先）
            tile_matches = re.findall(r'\[MonclerPLPStrategy\] Tile counts \(total=(\d+)\)', content)
            if tile_matches:
                # 最後のマッチを取得（通常、最後のマッチが最新の状態）
                last_tile_count = int(tile_matches[-1])
                # または total > 0 の最初のマッチを優先
                positive_tile_counts = [int(tc) for tc in tile_matches if int(tc) > 0]
                if positive_tile_counts:
                    errors["tile_count"] = positive_tile_counts[0]  # 最初の positive マッチ
                    errors["moncler_tiles_found"] = True
                else:
                    errors["tile_count"] = last_tile_count
                    errors["moncler_tiles_found"] = last_tile_count > 0
                
            # エラーメッセージ
            error_lines = [line.strip() for line in lines if 'ERROR' in line]
            errors["error_messages"] = error_lines[-10:] if error_lines else []
            
        except Exception as e:
            print(f"[警告] ログ解析エラー: {e}", file=sys.stderr)
            
        return errors
    
    def analyze_issues(self, errors: Dict) -> List[Dict]:
        """エラーを分析して修正案を生成"""
        fixes = []
        
        # 問題1: PDP リンクが0件だが、MonclerPLPStrategy ではタイルが見つかっている
        if errors["pdp_links_zero"] and errors["moncler_tiles_found"]:
            fixes.append({
                "type": "selector_mismatch",
                "description": "MonclerPLPStrategy ではタイルが見つかっているが、collect_pdp_links で見つからない",
                "suggestion": "pdp_link_selectors に 'a[href*=\"/products/\"]' が含まれているか確認",
                "file": "app/config/sites/overrides.local.json",
                "action": "check_and_add_selector",
                "auto_fixable": True,
                "priority": "high"
            })
        
        # 問題2: セレクタが見つからない（既に修正済みの可能性）
        for selector in errors.get("selector_not_found", []):
            if "data-qa='product-tile'" in selector or "data-qa=" in selector:
                fixes.append({
                    "type": "selector_timeout",
                    "description": f"セレクタが見つからない: {selector}",
                    "suggestion": "click_first_card_or_link のタイムアウト処理を改善（既に修正済み）",
                    "file": "app/agents/browser/navigation_driver.py",
                    "action": "already_fixed",
                    "auto_fixable": False,
                    "priority": "medium"
                })
        
        # 問題3: Materialization 失敗
        if errors["materialization_failed"]:
            if errors["moncler_tiles_found"] and errors["tile_count"] > 0:
                # MonclerPLPStrategy でタイルが見つかっているが、materialization で見つからない
                fixes.append({
                    "type": "materialization_selector_mismatch",
                    "description": f"MonclerPLPStrategy で {errors['tile_count']} 個のタイルが見つかっているが、materialization で見つからない",
                    "suggestion": "selectors.plp.tile_selectors に MonclerPLPStrategy で見つかっているセレクタを追加",
                    "file": "app/config/sites/overrides.local.json",
                    "action": "add_tile_selectors",
                    "auto_fixable": True,
                    "priority": "high",
                    "tile_count": errors["tile_count"]
                })
            elif errors["tile_count"] == 0:
                fixes.append({
                    "type": "materialization_selector",
                    "description": "タイルが見つからないため、materialization のセレクタを確認",
                    "suggestion": "ensure_plp_materialized の tile_selectors を確認",
                    "file": "app/agents/browser/navigation_driver.py",
                    "action": "check_materialization",
                    "auto_fixable": False,
                    "priority": "high"
                })
        
        return fixes
    
    def apply_fix(self, fix: Dict) -> bool:
        """修正を適用"""
        try:
            if fix["action"] == "check_and_add_selector":
                # pdp_link_selectors に 'a[href*="/products/"]' が含まれているか確認
                config_file = self.project_root / fix["file"]
                if not config_file.exists():
                    print(f"[エラー] ファイルが見つかりません: {fix['file']}")
                    return False
                
                try:
                    import json
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # MONCLER_OFFICIAL の selectors.plp.pdp_link_selectors を確認
                    moncler = config.get("MONCLER_OFFICIAL", {})
                    selectors = moncler.get("selectors", {})
                    plp = selectors.get("plp", {})
                    pdp_link_selectors = plp.get("pdp_link_selectors", [])
                    
                    # 'a[href*="/products/"]' が含まれているか確認
                    target_selector = 'a[href*="/products/"]'
                    if target_selector not in pdp_link_selectors:
                        # 追加
                        if not pdp_link_selectors:
                            pdp_link_selectors = []
                        pdp_link_selectors.insert(0, target_selector)  # 先頭に追加
                        plp["pdp_link_selectors"] = pdp_link_selectors
                        selectors["plp"] = plp
                        moncler["selectors"] = selectors
                        config["MONCLER_OFFICIAL"] = moncler
                        
                        # 保存
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                        
                        print(f"[修正] {fix['file']} に '{target_selector}' を追加しました")
                        self.fixes_applied.append(fix)
                        return True
                    else:
                        print(f"[確認] {fix['file']} に '{target_selector}' は既に含まれています")
                        return False
                        
                except Exception as e:
                    print(f"[エラー] JSON 読み込み/書き込みエラー: {e}")
                    return False
                    
            elif fix["action"] == "already_fixed":
                print(f"[情報] {fix['description']} - 既に修正済みです")
                return False
                
            elif fix["action"] == "add_tile_selectors":
                # selectors.plp.tile_selectors に MonclerPLPStrategy で見つかっているセレクタを追加
                config_file = self.project_root / fix["file"]
                if not config_file.exists():
                    print(f"[エラー] ファイルが見つかりません: {fix['file']}")
                    return False
                
                try:
                    import json
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                    
                    # MONCLER_OFFICIAL の selectors.plp.tile_selectors を確認
                    moncler = config.get("MONCLER_OFFICIAL", {})
                    selectors = moncler.get("selectors", {})
                    plp = selectors.get("plp", {})
                    tile_selectors = plp.get("tile_selectors", [])
                    
                    # MonclerPLPStrategy で見つかっているセレクタを追加
                    # ログから見つかっているセレクタ: "a[href*='/products/']", "div:has(a[href*='/products/'])"
                    new_selectors = [
                        "a[href*='/products/']",
                        "div:has(a[href*='/products/'])",
                        "a[href*='/product/']",
                        "a[href*='/p-']"
                    ]
                    
                    added = False
                    for sel in new_selectors:
                        if sel not in tile_selectors:
                            tile_selectors.append(sel)
                            added = True
                    
                    if added:
                        plp["tile_selectors"] = tile_selectors
                        selectors["plp"] = plp
                        moncler["selectors"] = selectors
                        config["MONCLER_OFFICIAL"] = moncler
                        
                        # 保存
                        with open(config_file, 'w', encoding='utf-8') as f:
                            json.dump(config, f, indent=2, ensure_ascii=False)
                        
                        print(f"[修正] {fix['file']} に tile_selectors を追加しました: {len(tile_selectors)} 個")
                        self.fixes_applied.append(fix)
                        return True
                    else:
                        print(f"[確認] {fix['file']} の tile_selectors は既に更新済みです")
                        return False
                        
                except Exception as e:
                    print(f"[エラー] JSON 読み込み/書き込みエラー: {e}")
                    return False
                    
            elif fix["action"] == "check_materialization":
                print(f"[提案] {fix['description']} - 手動確認が必要です")
                return False
                
        except Exception as e:
            print(f"[エラー] 修正適用エラー: {e}", file=sys.stderr)
            return False
            
        return False
    
    def run_test(self) -> Tuple[bool, Optional[Path]]:
        """テストを実行"""
        cmd = [
            sys.executable, "-u",
            "tools/run_browser_use.py",
            "--site", "MONCLER_OFFICIAL",
            "--url", "https://www.moncler.com/en-int/women/outerwear/all-down-jackets/?forceLocale=en-int&shipToCountry=GB",
            "--query", "down jacket",
            "--headful",
            "--timeout", "120"
        ]
        
        log_file = self.project_root / f"browser_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        print(f"[実行] テストを開始します...")
        print(f"[ログ] {log_file.name}")
        print()
        
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    timeout=180
                )
            
            success = result.returncode == 0
            return success, log_file
            
        except subprocess.TimeoutExpired:
            print("[エラー] テストがタイムアウトしました")
            return False, log_file
        except Exception as e:
            print(f"[エラー] テスト実行エラー: {e}", file=sys.stderr)
            return False, None
    
    def run_loop(self):
        """自動修正と再実行のループ"""
        print("=" * 80)
        print("完全自動修正と再実行ループ開始")
        print("=" * 80)
        print()
        
        while self.retry_count < self.max_retries:
            self.retry_count += 1
            print(f"[試行 {self.retry_count}/{self.max_retries}]")
            print()
            
            # テスト実行
            success, log_file = self.run_test()
            
            if success:
                print()
                print("=" * 80)
                print("✓ テストが成功しました！")
                print("=" * 80)
                if self.fixes_applied:
                    print()
                    print("適用された修正:")
                    for fix in self.fixes_applied:
                        print(f"  - {fix['description']}")
                return True
            
            if not log_file:
                print("[エラー] ログファイルが生成されませんでした")
                break
            
            # ログ解析
            print()
            print("[解析] ログファイルを解析中...")
            errors = self.parse_log_errors(log_file)
            
            # エラー表示
            print()
            print("--- 検出された問題 ---")
            if errors["pdp_links_zero"]:
                print("  ✗ PDP リンクが0件")
            if errors["materialization_failed"]:
                print("  ✗ PLP materialization が失敗")
            if errors["timeout"]:
                print("  ✗ タイムアウトが発生")
            if errors["selector_not_found"]:
                print(f"  ✗ セレクタが見つからない: {errors['selector_not_found'][:3]}")
            if errors["moncler_tiles_found"]:
                print(f"  ℹ MonclerPLPStrategy で {errors['tile_count']} 個のタイルが見つかりました")
            print()
            
            # 修正案の生成
            fixes = self.analyze_issues(errors)
            
            if not fixes:
                print("[情報] 自動修正可能な問題が見つかりませんでした")
                print("手動での確認が必要です")
                break
            
            # 修正案の表示
            print("--- 修正案 ---")
            auto_fixable_fixes = [f for f in fixes if f.get("auto_fixable", False)]
            manual_fixes = [f for f in fixes if not f.get("auto_fixable", False)]
            
            if auto_fixable_fixes:
                print("自動修正可能:")
                for i, fix in enumerate(auto_fixable_fixes, 1):
                    print(f"  {i}. [{fix['type']}] {fix['description']}")
                    print(f"     アクション: {fix['action']}")
                    print(f"     ファイル: {fix['file']}")
                    print()
            
            if manual_fixes:
                print("手動確認が必要:")
                for i, fix in enumerate(manual_fixes, 1):
                    print(f"  {i}. [{fix['type']}] {fix['description']}")
                    print(f"     提案: {fix['suggestion']}")
                    print()
            
            # 自動修正の適用
            if auto_fixable_fixes:
                print("[適用] 自動修正を適用中...")
                applied = False
                for fix in auto_fixable_fixes:
                    if self.apply_fix(fix):
                        applied = True
                        print(f"  ✓ {fix['description']} を修正しました")
                
                if applied:
                    print()
                    print("[再実行] 修正後に再実行します...")
                    print()
                    time.sleep(2)  # 少し待機
                    continue
                else:
                    print("[停止] 自動修正を適用できませんでした")
                    break
            else:
                print("[停止] 手動確認が必要な修正案のみです")
                print("上記の修正案を確認して、手動で修正してください")
                break
        
        print()
        print("=" * 80)
        print("自動修正ループを終了しました")
        print("=" * 80)
        if self.fixes_applied:
            print()
            print("適用された修正:")
            for fix in self.fixes_applied:
                print(f"  - {fix['description']}")
        return False

if __name__ == "__main__":
    project_root = Path(__file__).parent
    auto_fix = AutoFixAndRetry(project_root)
    success = auto_fix.run_loop()
    sys.exit(0 if success else 1)
