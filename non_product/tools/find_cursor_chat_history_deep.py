#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Cursorのチャット履歴を深く検索するスクリプト
User/globalStorage や User/workspaceStorage も検索対象に含める
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict
import platform

def find_windows_user_home():
    """Windowsのユーザーディレクトリを探す"""
    if platform.system() == "Linux":
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
CURSOR_DIR = WINDOWS_USER_HOME / ".cursor"

def format_size(size_bytes: int) -> str:
    """ファイルサイズを読みやすい形式に変換"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"

def find_db_files_in_directory(base_dir: Path, max_depth: int = 5) -> List[Dict]:
    """データベースファイルを再帰的に検索（深度制限付き）"""
    db_files = []
    if not base_dir.exists():
        return db_files
    
    patterns = ["*.db", "*chat*.db", "*history*.db", "state.vscdb", "*.sqlite", "*.sqlite3"]
    
    for pattern in patterns:
        try:
            # rglobを使うが、深度制限のために手動で探索
            for root, dirs, files in os.walk(str(base_dir)):
                # 深度チェック
                depth = Path(root).relative_to(base_dir).parts
                if len(depth) > max_depth:
                    dirs.clear()  # これ以上深く探索しない
                    continue
                
                for file in files:
                    if any(file.endswith(ext) for ext in ['.db', '.sqlite', '.sqlite3']):
                        file_path = Path(root) / file
                        try:
                            stat = file_path.stat()
                            db_files.append({
                                "path": str(file_path),
                                "name": file_path.name,
                                "size": stat.st_size,
                                "size_str": format_size(stat.st_size),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "relative_path": str(file_path.relative_to(base_dir)),
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
    print("Cursor チャット履歴 深度検索ツール")
    print("=" * 80)
    print()
    print(f"⏰ 検索条件: 更新日時が {cutoff_time.strftime('%Y-%m-%d %H:%M:%S')} 以降のファイルを優先表示")
    print()
    print(f"🔍 Cursorディレクトリ: {CURSOR_DIR}")
    print()
    
    if not CURSOR_DIR.exists():
        print(f"❌ Cursorディレクトリが見つかりません: {CURSOR_DIR}")
        return
    
    all_found_files = []
    
    # 1. User ディレクトリ配下を検索
    print("=" * 80)
    print("1. User ディレクトリ配下を検索")
    print("=" * 80)
    user_dir = CURSOR_DIR / "User"
    if user_dir.exists():
        print(f"パス: {user_dir}")
        print("検索中...")
        
        # globalStorage を検索
        global_storage = user_dir / "globalStorage"
        if global_storage.exists():
            print(f"\n   📂 globalStorage を検索中...")
            files = find_db_files_in_directory(global_storage, max_depth=3)
            if files:
                print(f"      ✅ {len(files)} 個のファイルが見つかりました")
                all_found_files.extend(files)
            else:
                print(f"      ⚠️  ファイルが見つかりませんでした")
        
        # workspaceStorage を検索
        workspace_storage = user_dir / "workspaceStorage"
        if workspace_storage.exists():
            print(f"\n   📂 workspaceStorage を検索中...")
            files = find_db_files_in_directory(workspace_storage, max_depth=4)
            if files:
                print(f"      ✅ {len(files)} 個のファイルが見つかりました")
                all_found_files.extend(files)
            else:
                print(f"      ⚠️  ファイルが見つかりませんでした")
        
        # User ディレクトリ直下も検索
        print(f"\n   📂 User ディレクトリ直下を検索中...")
        files = find_db_files_in_directory(user_dir, max_depth=2)
        if files:
            print(f"      ✅ {len(files)} 個のファイルが見つかりました")
            all_found_files.extend(files)
    else:
        print(f"❌ Userディレクトリが見つかりません: {user_dir}")
    print()
    
    # 2. projects ディレクトリ配下をさらに深く検索
    print("=" * 80)
    print("2. projects ディレクトリ配下を深度検索")
    print("=" * 80)
    projects_dir = CURSOR_DIR / "projects"
    if projects_dir.exists():
        print(f"パス: {projects_dir}")
        for project_dir in projects_dir.iterdir():
            if project_dir.is_dir():
                print(f"\n   📁 {project_dir.name} を検索中...")
                files = find_db_files_in_directory(project_dir, max_depth=5)
                if files:
                    print(f"      ✅ {len(files)} 個のファイルが見つかりました")
                    all_found_files.extend(files)
    
    # 3. Cursor ディレクトリ全体を検索
    print("=" * 80)
    print("3. Cursor ディレクトリ全体を検索")
    print("=" * 80)
    print("検索中...")
    files = find_db_files_in_directory(CURSOR_DIR, max_depth=6)
    if files:
        print(f"✅ {len(files)} 個のファイルが見つかりました")
        all_found_files.extend(files)
    print()
    
    # 重複を除去
    seen_paths = set()
    unique_files = []
    for f in all_found_files:
        if f["path"] not in seen_paths:
            seen_paths.add(f["path"])
            unique_files.append(f)
    
    all_found_files = unique_files
    
    # 結果を表示
    print("=" * 80)
    print("検索結果（更新日時順）")
    print("=" * 80)
    print()
    
    if all_found_files:
        # 更新日時でソート
        sorted_files = sorted(all_found_files, key=lambda x: x.get("modified", datetime.min), reverse=True)
        recent_files = [f for f in sorted_files if f.get("modified") and f["modified"] >= cutoff_time]
        old_files = [f for f in sorted_files if not f.get("modified") or f["modified"] < cutoff_time]
        
        if recent_files:
            print(f"🆕 最近更新されたファイル ({len(recent_files)} 個):")
            print()
            for f in recent_files:
                is_very_recent = f.get("modified") and (datetime.now() - f["modified"]).total_seconds() < 12 * 3600
                marker = "⭐" if is_very_recent else "•"
                time_label = "(今朝以降)" if is_very_recent else "(昨日以降)"
                
                print(f"   {marker} {f['name']}")
                print(f"      サイズ: {f['size_str']}")
                if f.get("modified"):
                    print(f"      更新日時: {f['modified'].strftime('%Y-%m-%d %H:%M:%S')} {time_label}")
                print(f"      パス: {f['path']}")
                if f.get("relative_path"):
                    print(f"      相対パス: {f['relative_path']}")
                print()
        
        if old_files:
            print(f"📄 それ以前のファイル ({len(old_files)} 個):")
            for f in old_files[:10]:
                print(f"   • {f['name']} ({f['size_str']}, {f['modified'].strftime('%Y-%m-%d %H:%M:%S') if f.get('modified') else 'N/A'})")
            if len(old_files) > 10:
                print(f"   ... 他 {len(old_files) - 10} 個")
    else:
        print("⚠️  データベースファイルが見つかりませんでした")
    
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

