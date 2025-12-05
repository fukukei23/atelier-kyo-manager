#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
__PRELOADED_STATE__ JSON データの解析スクリプト

Moncler の DOM snapshot に埋め込まれた __PRELOADED_STATE__ から商品データを抽出する。
"""
from __future__ import annotations
import sys
import re
import json
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def find_product_data(data: Any, path: str = "", max_depth: int = 5) -> List[tuple]:
    """再帰的に商品データを探す"""
    results = []
    if max_depth <= 0:
        return results
    
    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{path}.{key}" if path else key
            # product 関連のキーを探す
            if 'product' in key.lower():
                results.append((current_path, type(value).__name__, len(value) if isinstance(value, (list, dict)) else None))
            # 再帰的に探索
            results.extend(find_product_data(value, current_path, max_depth - 1))
    elif isinstance(data, list):
        for i, item in enumerate(data[:10]):  # 最初の10個のみ
            current_path = f"{path}[{i}]"
            results.extend(find_product_data(item, current_path, max_depth - 1))
    
    return results


def extract_product_urls(data: Any, urls: List[str] = None) -> List[str]:
    """商品 URL を抽出"""
    if urls is None:
        urls = []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                if '/products/' in value or '/product/' in value:
                    if 'moncler.com' in value or value.startswith('/'):
                        urls.append(value)
            else:
                extract_product_urls(value, urls)
    elif isinstance(data, list):
        for item in data:
            extract_product_urls(item, urls)
    
    return urls


def main(run_id: str) -> None:
    """メイン処理"""
    run_dir = ROOT / "instance" / "runs" / run_id
    html_file = run_dir / "failure_dom.html"
    
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        return
    
    print(f"📁 Analyzing: {html_file}")
    
    html = html_file.read_text(encoding='utf-8', errors='ignore')
    
    # application/json スクリプトを抽出
    json_scripts = re.findall(
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.+?)</script>',
        html, re.I | re.DOTALL
    )
    
    if not json_scripts:
        print("❌ No application/json scripts found")
        return
    
    print(f"\nFound {len(json_scripts)} application/json scripts")
    
    # 最初の大きな JSON を解析
    script = json_scripts[0]
    print(f"\n=== Script 1 (Length: {len(script)} chars) ===")
    
    try:
        data = json.loads(script)
        print(f"✅ Valid JSON (type: {type(data).__name__})")
        
        if isinstance(data, dict):
            print(f"\nTop-level keys: {list(data.keys())[:20]}")
            
            # __PRELOADED_STATE__ を探す
            if '__PRELOADED_STATE__' in data:
                print("\n" + "=" * 80)
                print("📊 __PRELOADED_STATE__ Analysis:")
                print("=" * 80)
                
                preloaded = data['__PRELOADED_STATE__']
                print(f"Type: {type(preloaded).__name__}")
                
                if isinstance(preloaded, dict):
                    print(f"\nKeys: {list(preloaded.keys())[:30]}")
                    
                    # 商品関連のキーを探す
                    print("\n🔍 Product-related keys:")
                    product_keys = [k for k in preloaded.keys() if 'product' in k.lower()]
                    for key in product_keys:
                        value = preloaded[key]
                        print(f"  - {key}: {type(value).__name__}")
                        if isinstance(value, (list, dict)):
                            print(f"    Size: {len(value)}")
                            if isinstance(value, list) and value:
                                print(f"    First item type: {type(value[0]).__name__}")
                                if isinstance(value[0], dict):
                                    print(f"    First item keys: {list(value[0].keys())[:10]}")
                    
                    # 商品データを再帰的に探す
                    print("\n🔍 Recursive product data search:")
                    product_data = find_product_data(preloaded, max_depth=3)
                    for path, data_type, size in product_data[:20]:
                        print(f"  - {path}: {data_type}" + (f" (size: {size})" if size else ""))
                    
                    # 商品 URL を抽出
                    print("\n🔍 Product URLs extraction:")
                    product_urls = extract_product_urls(preloaded)
                    unique_urls = list(set(product_urls))[:20]
                    print(f"Found {len(product_urls)} product URLs ({len(unique_urls)} unique)")
                    for url in unique_urls:
                        print(f"  - {url}")
                
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "20251203_112454_918"
    main(run_id)

