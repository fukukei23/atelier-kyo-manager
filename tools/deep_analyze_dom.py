#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOM snapshot の詳細解析スクリプト

Moncler の DOM snapshot から商品リンクの構造を詳細に解析する。
"""
from __future__ import annotations
import sys
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def analyze_json_scripts(html: str) -> None:
    """application/json スクリプトタグを解析"""
    print("\n" + "=" * 80)
    print("📊 Embedded JSON Scripts Analysis:")
    print("=" * 80)
    
    json_scripts = re.findall(
        r'<script[^>]*type=["\']application/json["\'][^>]*>(.+?)</script>',
        html, re.I | re.DOTALL
    )
    print(f"\nFound {len(json_scripts)} application/json scripts")
    
    for i, script in enumerate(json_scripts[:3], 1):
        print(f"\n=== Script {i} ===")
        print(f"Length: {len(script)} chars")
        print(f"First 500 chars: {script[:500]}")
        
        # 商品関連のキーワードを探す
        if 'product' in script.lower():
            print("  -> Contains 'product' keyword")
        if '/products/' in script or '/product/' in script:
            print("  -> Contains product URL pattern")
        
        # JSON としてパース可能か確認
        try:
            data = json.loads(script)
            print(f"  -> Valid JSON (type: {type(data).__name__})")
            if isinstance(data, dict):
                print(f"  -> Keys: {list(data.keys())[:10]}")
        except:
            print("  -> Not valid JSON or too complex")


def analyze_product_cards(html: str) -> None:
    """div[data-test='product-card'] を詳細に解析"""
    print("\n" + "=" * 80)
    print("🃏 Product Card Elements Analysis:")
    print("=" * 80)
    
    # 正確なパターンで検索
    pattern = r'<div[^>]*data-test=["\']product-card["\'][^>]*>'
    matches = list(re.finditer(pattern, html, re.I))
    print(f"\nFound {len(matches)} div[data-test='product-card'] elements")
    
    if not matches:
        # 別のパターンも試す
        alt_patterns = [
            r'data-test=["\']product-card["\']',
            r'data-test=["\']product-grid["\']',
        ]
        for alt_pattern in alt_patterns:
            alt_matches = list(re.finditer(alt_pattern, html, re.I))
            if alt_matches:
                print(f"  -> Found {len(alt_matches)} matches for pattern: {alt_pattern}")
                # 最初のマッチの周辺を表示
                match = alt_matches[0]
                start = max(0, match.start() - 200)
                end = min(len(html), match.end() + 500)
                snippet = html[start:end]
                print(f"  -> Sample context:\n{snippet[:600]}")
        return
    
    # 最初の3つのカードを詳細に解析
    for i, match in enumerate(matches[:3], 1):
        print(f"\n=== Card {i} ===")
        start = match.start()
        end = min(len(html), start + 2000)  # 次の2000文字を取得
        
        # 次の </div> までを取得（簡易版）
        card_html = html[start:end]
        print(f"Position: {start}-{end}")
        print(f"HTML snippet (first 400 chars):\n{card_html[:400]}")
        
        # このカード内の a タグを探す
        a_tags = re.findall(r'<a[^>]*href=["\']([^"\']*)["\']', card_html, re.I)
        if a_tags:
            print(f"\n  -> Found {len(a_tags)} <a> tags with href:")
            for href in a_tags[:5]:
                print(f"     - {href}")
        else:
            print("\n  -> No <a> tags found in this card")
        
        # data-* 属性を探す
        data_attrs = re.findall(r'data-([^=]+)=["\']([^"\']*)["\']', card_html, re.I)
        if data_attrs:
            print(f"\n  -> Found {len(data_attrs)} data-* attributes:")
            for attr_name, attr_value in data_attrs[:5]:
                if len(attr_value) < 100:
                    print(f"     - data-{attr_name}={attr_value}")
                else:
                    print(f"     - data-{attr_name}=... ({len(attr_value)} chars)")


def analyze_all_links(html: str) -> None:
    """すべてのリンクを解析"""
    print("\n" + "=" * 80)
    print("🔗 All Links Analysis:")
    print("=" * 80)
    
    # すべての a タグを取得
    a_tags = re.findall(r'<a[^>]*href=["\']([^"\']*)["\']', html, re.I)
    print(f"\nFound {len(a_tags)} total <a> tags with href")
    
    # Moncler ドメインのリンク
    moncler_links = [link for link in a_tags if 'moncler.com' in link]
    print(f"  -> Moncler domain links: {len(moncler_links)}")
    
    # /products/ または /product/ を含むリンク
    product_links = [link for link in a_tags if '/products/' in link or '/product/' in link]
    print(f"  -> Product URL links: {len(product_links)}")
    
    if product_links:
        print("\n  Product links found:")
        for link in product_links[:10]:
            print(f"    - {link}")
    
    # 相対パスのリンク
    relative_links = [link for link in a_tags if link.startswith('/')]
    print(f"\n  -> Relative path links: {len(relative_links)}")
    if relative_links:
        # /products/ を含む相対パス
        rel_product_links = [link for link in relative_links if '/products/' in link or '/product/' in link]
        if rel_product_links:
            print(f"    -> Product relative links: {len(rel_product_links)}")
            for link in rel_product_links[:10]:
                print(f"       - {link}")


def main(run_id: str) -> None:
    """メイン処理"""
    run_dir = ROOT / "instance" / "runs" / run_id
    html_file = run_dir / "failure_dom.html"
    
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        return
    
    print(f"📁 Analyzing: {html_file}")
    print(f"📊 File size: {html_file.stat().st_size / 1024 / 1024:.2f} MB")
    
    html = html_file.read_text(encoding='utf-8', errors='ignore')
    
    analyze_json_scripts(html)
    analyze_product_cards(html)
    analyze_all_links(html)
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "20251203_112454_918"
    main(run_id)

