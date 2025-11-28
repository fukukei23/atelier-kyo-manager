#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cursorチャット履歴を検索して実行（直接実行版）
ターミナルが動作しない場合でも実行できるように、すべてのロジックを内包
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Windowsパス（WSL経由とWindows直接の両方に対応）
import platform

def find_windows_user_home():
    """Windowsのユーザーディレクトリを探す"""
    if platform.system() == "Linux":
        # WSL経由の場合
        users_dir = Path("/mnt/c/Users")
        if users_dir.exists():
            user_dirs = [d for d in users_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
            if user_dirs:
                user_dir = next((d for d in user_dirs if d.name == "USER"), None)
                if user_dir:
                    return user_dir
                return user_dirs[0]
            return Path("/mnt/c/Users/USER")
        return Path("/mnt/c/Users/USER")
    else:
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

def find_db_files(base_dir: Path) -> List[Dict]:
    """データベースファイルを再帰的に検索"""
    db_files = []
    if not base_dir.exists():
        return db_files
    
    for pattern in ["*.db", "*chat*.db", "*history*.db", "state.vscdb", "*.sqlite", "*.sqlite3"]:
        try:
            for db_file in base_dir.rglob(pattern):
                if db_file.is_file():
                    try:
                        stat = db_file.stat()
                        db_files.append({
                            "path": str(db_file),
                            "size": stat.st_size,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                            "size_str": format_size(stat.st_size),
                            "name": db_file.name,
                        })
                    except Exception:
                        pass
        except Exception:
            pass
    
    return db_files

def main():
    """メイン処理"""
    cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
    
    print("=" * 80)
    print("Cursor チャット履歴検索ツール")
    print("=" * 80)
    print()
    print(f"⏰ 検索条件: 更新日時が {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} 以降のファイルを優先表示")
    print()
    print(f"🔍 Windowsユーザーディレクトリ: {WINDOWS_USER_HOME}")
    print(f"🔍 Cursorプロジェクトディレクトリ: {CURSOR_PROJECTS_DIR}")
    print()
    
    if not CURSOR_PROJECTS_DIR.exists():
        print(f"❌ Cursorプロジェクトディレクトリが見つかりません: {CURSOR_PROJECTS_DIR}")
        print()
        print("💡 代替パスを検索中...")
        alternative_paths = [
            Path("/mnt/c/Users") / d / ".cursor" / "projects"
            for d in ["USER", os.getenv("USER", "USER")]
        ]
        
        found = False
        for alt_path in alternative_paths:
            if alt_path.exists():
                print(f"✅ 見つかりました: {alt_path}")
                global CURSOR_PROJECTS_DIR
                CURSOR_PROJECTS_DIR = alt_path
                found = True
                break
        
        if not found:
            print("手動で確認してください:")
            print(f"  1. Windowsのエクスプローラーを開く")
            print(f"  2. アドレスバーに以下を入力: {CURSOR_PROJECTS_DIR}")
            return
    
    # すべてのプロジェクトをリストアップ
    projects = []
    if CURSOR_PROJECTS_DIR.exists():
        try:
            projects = [d for d in CURSOR_PROJECTS_DIR.iterdir() if d.is_dir()]
        except Exception as e:
            print(f"❌ プロジェクトディレクトリの読み込みに失敗: {e}")
            return
    
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
    all_recent_files = []
    
    for project_dir in projects:
        print(f"🔍 分析中: {project_dir.name}")
        db_files = find_db_files(project_dir)
        
        if db_files:
            recent_count = len([f for f in db_files if f.get("modified") and f["modified"] >= cutoff_time])
            print(f"   ✅ {len(db_files)} 個のデータベースファイルが見つかりました")
            if recent_count > 0:
                print(f"   🆕 そのうち {recent_count} 個が最近（昨日以降）更新されています")
        else:
            print(f"   ⚠️  データベースファイルが見つかりませんでした")
        print()
        
        all_results.append({
            "project_name": project_dir.name,
            "path": str(project_dir),
            "db_files": db_files,
        })
    
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
            sorted_files = sorted(result["db_files"], key=lambda x: x.get("modified", datetime.min), reverse=True)
            recent_files = [f for f in sorted_files if f.get("modified") and f["modified"] >= cutoff_time]
            old_files = [f for f in sorted_files if not f.get("modified") or f["modified"] < cutoff_time]
            
            if recent_files:
                print(f"   🆕 最近更新されたファイル ({len(recent_files)} 個):")
                for db in recent_files:
                    is_very_recent = db.get("modified") and (datetime.now() - db["modified"]).total_seconds() < 12 * 3600
                    marker = "⭐" if is_very_recent else "•"
                    time_label = "(今朝以降)" if is_very_recent else "(昨日以降)"
                    
                    print(f"      {marker} {db['name']}")
                    print(f"         サイズ: {db['size_str']}")
                    if db.get("modified"):
                        print(f"         更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')} {time_label}")
                    print(f"         パス: {db['path']}")
                    print()
                    
                    all_recent_files.append({**db, "project": result['project_name']})
            
            if old_files:
                print(f"   📄 それ以前のファイル ({len(old_files)} 個):")
                for db in old_files[:5]:
                    print(f"      • {db['name']}")
                    print(f"        サイズ: {db['size_str']}")
                    if db.get("modified"):
                        print(f"        更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
                    print()
                if len(old_files) > 5:
                    print(f"      ... 他 {len(old_files) - 5} 個のファイル")
                print()
        
        print("-" * 80)
        print()
    
    # 最近更新されたファイルを優先表示
    if all_recent_files:
        print("=" * 80)
        print("🎯 最近更新されたデータベースファイル（再起動後のチャット履歴の可能性が高い）")
        print("=" * 80)
        print()
        
        sorted_recent = sorted(all_recent_files, key=lambda x: x.get("modified", datetime.min), reverse=True)[:10]
        
        for db in sorted_recent:
            print(f"   ⭐ プロジェクト: {db['project']}")
            print(f"      ファイル: {db['name']}")
            print(f"      サイズ: {db['size_str']}")
            if db.get("modified"):
                print(f"      更新日時: {db['modified'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      パス: {db['path']}")
            print()
    
    print("=" * 80)
    print("完了")
    print("=" * 80)

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

