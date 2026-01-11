#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursorのチャット履歴データベースファイルを検索するスクリプト

使い方:
    python tools/find_cursor_chat_history.py

Windows側の C:\Users\USER\.cursor\projects\ 配下を検索して、
チャット履歴の可能性があるデータベースファイルをリストアップします。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Windowsパス（WSL経由とWindows直接の両方に対応）
import platform
import os

def find_windows_user_home():
    """Windowsのユーザーディレクトリを探す"""
    if platform.system() == "Linux":
        # WSL経由の場合
        # /mnt/c/Users/ 配下から探す
        users_dir = Path("/mnt/c/Users")
        if users_dir.exists():
            # 最初に見つかったディレクトリを使用（通常は1つだけ）
            user_dirs = [d for d in users_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if user_dirs:
                # USERという名前があれば優先
                user_dir = next((d for d in user_dirs if d.name == "USER"), None)
                if user_dir:
                    return user_dir
                return user_dirs[0]
            # デフォルトとしてUSERを使用
            return Path("/mnt/c/Users/USER")
        return Path("/mnt/c/Users/USER")
    else:
        # Windows直接の場合
        # 環境変数から取得
        user_profile = os.getenv("USERPROFILE") or os.getenv("HOME")
        if user_profile:
            return Path(user_profile)
        return Path("C:/Users/USER")

WINDOWS_USER_HOME = find_windows_user_home()
CURSOR_PROJECTS_DIR = WINDOWS_USER_HOME / ".cursor" / "projects"


def format_size(size_bytes: int) -> str:
    """ファイルサイズを読みやすい形式に変換"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def get_file_info(file_path: Path) -> Dict[str, any]:
    """ファイル情報を取得"""
    try:
        stat = file_path.stat()
        return {
            "path": str(file_path),
            "size": stat.st_size,
            "modified": datetime.fromtimestamp(stat.st_mtime),
            "size_str": format_size(stat.st_size),
        }
    except Exception as e:
        return {
            "path": str(file_path),
            "size": 0,
            "modified": None,
            "size_str": "0 B",
            "error": str(e),
        }


def find_db_files(base_dir: Path) -> List[Dict[str, any]]:
    """データベースファイルを再帰的に検索"""
    db_files = []
    
    if not base_dir.exists():
        print(f"⚠️  ディレクトリが存在しません: {base_dir}")
        return db_files
    
    print(f"🔍 検索中: {base_dir}")
    print()
    
    # 検索パターン
    db_patterns = [
        "*.db",
        "*chat*.db",
        "*history*.db",
        "state.vscdb",
        "*.sqlite",
        "*.sqlite3",
    ]
    
    for pattern in db_patterns:
        for db_file in base_dir.rglob(pattern):
            if db_file.is_file():
                info = get_file_info(db_file)
                info["pattern"] = pattern
                db_files.append(info)
    
    return db_files


def find_workspace_storage_dirs(base_dir: Path) -> List[Path]:
    """workspaceStorage ディレクトリを探す"""
    storage_dirs = []
    
    if not base_dir.exists():
        return storage_dirs
    
    for ws_dir in base_dir.rglob("workspaceStorage"):
        if ws_dir.is_dir():
            storage_dirs.append(ws_dir)
    
    return storage_dirs


def analyze_workspace(base_dir: Path) -> Dict[str, any]:
    """ワークスペースディレクトリを分析"""
    result = {
        "path": str(base_dir),
        "exists": base_dir.exists(),
        "db_files": [],
        "storage_dirs": [],
        "project_name": base_dir.name,
    }
    
    if not base_dir.exists():
        return result
    
    # データベースファイルを検索
    result["db_files"] = find_db_files(base_dir)
    
    # workspaceStorage ディレクトリを検索
    result["storage_dirs"] = [
        str(d) for d in find_workspace_storage_dirs(base_dir)
    ]
    
    return result


def main():
    """メイン処理"""
    from datetime import timedelta
    
    print("=" * 80)
    print("Cursor チャット履歴検索ツール")
    print("=" * 80)
    print()
    
    # 昨日の0時以降を検索対象
    cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    print(f"⏰ 検索条件: 更新日時が {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} 以降のファイルを優先表示")
    print()
    
    # プロジェクトディレクトリを検索
    if not CURSOR_PROJECTS_DIR.exists():
        print(f"❌ Cursorプロジェクトディレクトリが見つかりません: {CURSOR_PROJECTS_DIR}")
        print()
        print("手動で確認してください:")
        print("  1. Windowsのエクスプローラーを開く")
        print("  2. アドレスバーに以下を入力:")
        print(f"     {CURSOR_PROJECTS_DIR}")
        return
    
    print(f"📁 プロジェクトディレクトリ: {CURSOR_PROJECTS_DIR}")
    print()
    
    # すべてのプロジェクトをリストアップ
    projects = []
    if CURSOR_PROJECTS_DIR.exists():
        for project_dir in CURSOR_PROJECTS_DIR.iterdir():
            if project_dir.is_dir():
                projects.append(project_dir)
    
    if not projects:
        print("⚠️  プロジェクトディレクトリが見つかりませんでした")
        return
    
    print(f"📦 {len(projects)} 個のプロジェクトが見つかりました:")
    for i, proj in enumerate(projects, 1):
        print(f"   {i}. {proj.name}")
    print()
    print("=" * 80)
    print()
    
    # 各プロジェクトを分析
    all_results = []
    all_recent_files = []  # 最近更新されたファイルを収集
    
    for project_dir in projects:
        print(f"🔍 分析中: {project_dir.name}")
        result = analyze_workspace(project_dir)
        all_results.append(result)
        
        if result["db_files"]:
            recent_count = len([f for f in result["db_files"] if f.get("modified") and f["modified"] >= cutoff_time])
            print(f"   ✅ {len(result['db_files'])} 個のデータベースファイルが見つかりました")
            if recent_count > 0:
                print(f"   🆕 そのうち {recent_count} 個が最近（昨日以降）更新されています")
        else:
            print(f"   ⚠️  データベースファイルが見つかりませんでした")
        
        if result["storage_dirs"]:
            print(f"   📂 workspaceStorage: {len(result['storage_dirs'])} 個")
        print()
    
    # 結果をまとめて表示
    print("=" * 80)
    print("検索結果（更新日時順・最近更新されたものが先頭）")
    print("=" * 80)
    print()
    
    for result in all_results:
        print(f"📦 プロジェクト: {result['project_name']}")
        print(f"   パス: {result['path']}")
        print()
        
        if result["db_files"]:
            # 更新日時でソート（最近更新されたものから）
            sorted_files = sorted(result["db_files"], key=lambda x: x.get("modified", datetime.min), reverse=True)
            
            # 最近更新されたファイルを分離
            recent_files = [f for f in sorted_files if f.get("modified") and f["modified"] >= cutoff_time]
            old_files = [f for f in sorted_files if not f.get("modified") or f["modified"] < cutoff_time]
            
            if recent_files:
                print(f"   🆕 最近更新されたファイル ({len(recent_files)} 個):")
                for db in recent_files:
                    is_very_recent = db.get("modified") and (datetime.now() - db["modified"]).total_seconds() < 12 * 3600
                    marker = "⭐" if is_very_recent else "•"
                    time_label = "(今朝以降)" if is_very_recent else "(昨日以降)"
                    
                    print(f"      {marker} {Path(db['path']).name}")
                    print(f"         サイズ: {db['size_str']}")
                    if db.get("modified"):
                        print(f"         更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')} {time_label}")
                    print(f"         パス: {db['path']}")
                    print()
                    
                    all_recent_files.append({**db, "project": result['project_name']})
            
            if old_files:
                print(f"   📄 それ以前のファイル ({len(old_files)} 個):")
                for db in old_files[:5]:  # 最大5個まで表示
                    print(f"      • {Path(db['path']).name}")
                    print(f"        サイズ: {db['size_str']}")
                    if db.get("modified"):
                        print(f"        更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
                if len(old_files) > 5:
                    print(f"      ... 他 {len(old_files) - 5} 個のファイル")
                print()
        
        if result["storage_dirs"]:
            print(f"   📂 workspaceStorage ディレクトリ:")
            for storage_dir in result["storage_dirs"]:
                print(f"      • {storage_dir}")
                # workspaceStorage内のデータベースファイルも検索
                storage_path = Path(storage_dir)
                if storage_path.exists():
                    for db_file in storage_path.rglob("*.db"):
                        if db_file.is_file():
                            info = get_file_info(db_file)
                            print(f"         - {db_file.name} ({info['size_str']})")
            print()
        
        print("-" * 80)
        print()
    
    # 最近更新されたファイルを優先表示
    if all_recent_files:
        print("=" * 80)
        print("🎯 最近更新されたデータベースファイル（再起動後のチャット履歴の可能性が高い）")
        print("=" * 80)
        print()
        
        # 更新日時順にソート
        sorted_recent = sorted(all_recent_files, key=lambda x: x.get("modified", datetime.min), reverse=True)[:10]
        
        for db in sorted_recent:
            print(f"   ⭐ プロジェクト: {db['project']}")
            print(f"      ファイル: {Path(db['path']).name}")
            print(f"      サイズ: {db['size_str']}")
            if db.get("modified"):
                print(f"      更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      パス: {db['path']}")
            print()
    
    # 最大サイズのファイル（参考）
    all_db_files = []
    for result in all_results:
        all_db_files.extend(result["db_files"])
    
    if all_db_files:
        largest = max(all_db_files, key=lambda x: x.get("size", 0))
        print("💡 最も大きなデータベースファイル（参考）:")
        print(f"   ファイル: {Path(largest['path']).name}")
        print(f"   サイズ: {largest['size_str']}")
        if largest.get("modified"):
            print(f"   更新日時: {largest['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   パス: {largest['path']}")
        print()
    
    print("=" * 80)
    print("完了")
    print("=" * 80)
    print()
    print("💡 ヒント:")
    print("   - 大きいデータベースファイルがチャット履歴の可能性が高いです")
    print("   - workspaceStorage ディレクトリ内にも履歴がある可能性があります")
    print("   - プロジェクトを開く際は、常に同じパスから開くようにしてください")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

