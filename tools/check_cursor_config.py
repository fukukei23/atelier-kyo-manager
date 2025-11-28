#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Cursorの設定ファイルとディレクトリの内容を確認するスクリプト（読み取り専用）

確認対象:
- C:\Users\USER\.cursor\extensions
- C:\Users\USER\.cursor\projects
- C:\Users\USER\.cursor\argv.json
- C:\Users\USER\.cursor\ide_state.json
"""
from __future__ import annotations

import os
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
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

def read_json_file(file_path: Path) -> Optional[Dict]:
    """JSONファイルを読み込む（安全に）"""
    if not file_path.exists():
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}

def list_directory(dir_path: Path, max_items: int = 20) -> List[Dict]:
    """ディレクトリの内容をリストアップ"""
    if not dir_path.exists() or not dir_path.is_dir():
        return []
    
    items = []
    try:
        for item in dir_path.iterdir():
            try:
                if item.is_file():
                    stat = item.stat()
                    items.append({
                        "name": item.name,
                        "type": "file",
                        "size": stat.st_size,
                        "size_str": format_size(stat.st_size),
                        "modified": datetime.fromtimestamp(stat.st_mtime),
                    })
                elif item.is_dir():
                    try:
                        stat = item.stat()
                        # ディレクトリ内のファイル数を数える（限定的に）
                        file_count = sum(1 for _ in item.iterdir()) if item.exists() else 0
                        items.append({
                            "name": item.name,
                            "type": "directory",
                            "file_count": file_count,
                            "modified": datetime.fromtimestamp(stat.st_mtime),
                        })
                    except:
                        items.append({
                            "name": item.name,
                            "type": "directory",
                            "file_count": "?",
                        })
            except Exception as e:
                items.append({
                    "name": item.name,
                    "type": "unknown",
                    "error": str(e),
                })
            
            if len(items) >= max_items:
                break
    except Exception as e:
        return [{"error": str(e)}]
    
    return items

def main():
    """メイン処理"""
    print("=" * 80)
    print("Cursor 設定ファイル確認ツール（読み取り専用）")
    print("=" * 80)
    print()
    print(f"🔍 Cursorディレクトリ: {CURSOR_DIR}")
    print()
    
    if not CURSOR_DIR.exists():
        print(f"❌ Cursorディレクトリが見つかりません: {CURSOR_DIR}")
        return
    
    # 1. extensions ディレクトリ
    print("=" * 80)
    print("1. extensions ディレクトリ")
    print("=" * 80)
    extensions_dir = CURSOR_DIR / "extensions"
    print(f"パス: {extensions_dir}")
    print()
    
    if extensions_dir.exists():
        items = list_directory(extensions_dir, max_items=30)
        if items:
            print(f"📦 {len(items)} 個のアイテムが見つかりました:")
            for item in items[:10]:
                if item.get("type") == "directory":
                    print(f"   📁 {item['name']} (ディレクトリ, {item.get('file_count', '?')} 個のアイテム)")
                elif item.get("type") == "file":
                    print(f"   📄 {item['name']} ({item.get('size_str', '?')})")
                else:
                    print(f"   ❓ {item.get('name', 'unknown')}")
            if len(items) > 10:
                print(f"   ... 他 {len(items) - 10} 個")
        else:
            print("⚠️  ディレクトリは空です")
    else:
        print("❌ ディレクトリが存在しません")
    print()
    
    # 2. projects ディレクトリ
    print("=" * 80)
    print("2. projects ディレクトリ")
    print("=" * 80)
    projects_dir = CURSOR_DIR / "projects"
    print(f"パス: {projects_dir}")
    print()
    
    if projects_dir.exists():
        items = list_directory(projects_dir, max_items=20)
        if items:
            print(f"📦 {len(items)} 個のプロジェクトが見つかりました:")
            for item in items:
                if item.get("type") == "directory":
                    print(f"   📁 {item['name']}")
                    # 各プロジェクトディレクトリの中身も確認
                    project_path = projects_dir / item['name']
                    try:
                        db_files = list(project_path.rglob("*.db"))
                        if db_files:
                            print(f"      ✅ {len(db_files)} 個の .db ファイルがあります")
                            # 最近更新されたものを表示
                            recent_dbs = sorted(db_files, key=lambda p: p.stat().st_mtime, reverse=True)[:3]
                            for db in recent_dbs:
                                try:
                                    stat = db.stat()
                                    print(f"         • {db.name} ({format_size(stat.st_size)}, {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')})")
                                except:
                                    pass
                        else:
                            print(f"      ⚠️  .db ファイルが見つかりません")
                    except Exception as e:
                        print(f"      ❌ エラー: {e}")
        else:
            print("⚠️  ディレクトリは空です")
    else:
        print("❌ ディレクトリが存在しません")
    print()
    
    # 3. argv.json ファイル
    print("=" * 80)
    print("3. argv.json ファイル")
    print("=" * 80)
    argv_json = CURSOR_DIR / "argv.json"
    print(f"パス: {argv_json}")
    print()
    
    if argv_json.exists():
        data = read_json_file(argv_json)
        if data and "error" not in data:
            print("📄 内容:")
            print(json.dumps(data, indent=2, ensure_ascii=False))
        elif data and "error" in data:
            print(f"❌ 読み込みエラー: {data['error']}")
        else:
            print("⚠️  ファイルが空です")
    else:
        print("❌ ファイルが存在しません")
    print()
    
    # 4. ide_state.json ファイル
    print("=" * 80)
    print("4. ide_state.json ファイル")
    print("=" * 80)
    ide_state_json = CURSOR_DIR / "ide_state.json"
    print(f"パス: {ide_state_json}")
    print()
    
    if ide_state_json.exists():
        data = read_json_file(ide_state_json)
        if data and "error" not in data:
            print("📄 内容:")
            # 大きいファイルの場合は要約のみ表示
            json_str = json.dumps(data, indent=2, ensure_ascii=False)
            if len(json_str) > 2000:
                print(json_str[:2000])
                print("\n... (内容が長いため省略)")
            else:
                print(json_str)
        elif data and "error" in data:
            print(f"❌ 読み込みエラー: {data['error']}")
        else:
            print("⚠️  ファイルが空です")
    else:
        print("❌ ファイルが存在しません")
    print()
    
    # 追加: チャット履歴に関連しそうなファイルを検索（より深く）
    print("=" * 80)
    print("5. チャット履歴に関連しそうなファイルを検索（深度検索）")
    print("=" * 80)
    print()
    
    # User/globalStorage と User/workspaceStorage も検索
    search_dirs = [
        CURSOR_DIR,
        CURSOR_DIR / "User" / "globalStorage",
        CURSOR_DIR / "User" / "workspaceStorage",
    ]
    
    search_patterns = ["*chat*.db", "*history*.db", "*state*.db", "*.sqlite", "*.sqlite3", "*.db"]
    found_files = []
    
    for search_dir in search_dirs:
        if not search_dir.exists():
            continue
        
        print(f"🔍 検索中: {search_dir}")
        for pattern in search_patterns:
            try:
                for file_path in search_dir.rglob(pattern):
                    if file_path.is_file():
                        try:
                            stat = file_path.stat()
                            found_files.append({
                                "path": str(file_path),
                                "name": file_path.name,
                                "size": stat.st_size,
                                "size_str": format_size(stat.st_size),
                                "modified": datetime.fromtimestamp(stat.st_mtime),
                                "relative_path": str(file_path.relative_to(CURSOR_DIR)),
                            })
                        except:
                            pass
            except Exception as e:
                print(f"   ⚠️  エラー: {e}")
    
    # 重複を除去
    seen_paths = set()
    unique_files = []
    for f in found_files:
        if f["path"] not in seen_paths:
            seen_paths.add(f["path"])
            unique_files.append(f)
    found_files = unique_files
    
    if found_files:
        print(f"\n✅ {len(found_files)} 個の関連ファイルが見つかりました:")
        # 更新日時順にソート
        found_files.sort(key=lambda x: x.get("modified", datetime.min), reverse=True)
        
        # 最近更新されたファイルを優先表示
        cutoff_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        recent_files = [f for f in found_files if f.get("modified") and f["modified"] >= cutoff_time]
        old_files = [f for f in found_files if not f.get("modified") or f["modified"] < cutoff_time]
        
        if recent_files:
            print(f"\n🆕 最近更新されたファイル ({len(recent_files)} 個):")
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
            print(f"\n📄 それ以前のファイル ({len(old_files)} 個):")
            for f in old_files[:5]:
                print(f"   • {f['name']} ({f['size_str']}, {f['modified'].strftime('%Y-%m-%d %H:%M:%S') if f.get('modified') else 'N/A'})")
                if f.get("relative_path"):
                    print(f"     相対パス: {f['relative_path']}")
            if len(old_files) > 5:
                print(f"   ... 他 {len(old_files) - 5} 個")
    else:
        print("⚠️  関連ファイルが見つかりませんでした")
        print("\n💡 ヒント:")
        print("   - User/globalStorage や User/workspaceStorage にチャット履歴が保存されている可能性があります")
        print("   - より深く検索する場合は、find_cursor_chat_history_deep.py を実行してください")
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

