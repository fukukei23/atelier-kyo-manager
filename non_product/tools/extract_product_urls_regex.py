#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
正規表現で商品 URL を抽出（JSON パース不要）

巨大な JSON をパースせず、正規表現で直接商品 URL を抽出する。
"""
from __future__ import annotations
import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def extract_product_urls_regex(run_id: str) -> None:
    """正規表現で商品 URL を抽出"""
    run_dir = ROOT / "instance" / "runs" / run_id
    html_file = run_dir / "failure_dom.html"
    
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        return
    
    print(f"📁 Analyzing: {html_file}")
    html = html_file.read_text(encoding='utf-8', errors='ignore')
    
    # JSON をパースせず、正規表現で直接商品 URL を抽出
    product_url_patterns = [
        # 完全な URL
        r'https?://[^"\'<>]*moncler\.com[^"\'<>]*/products/[^"\'<>]*',
        r'https?://[^"\'<>]*moncler\.com[^"\'<>]*/product/[^"\'<>]*',
        # 相対パス（JSON 内の可能性）
        r'"/[^"\'<>]*/products/[^"\'<>]*"',
        r'"/[^"\'<>]*/product/[^"\'<>]*"',
        r"'/[^\"'<>]*/products/[^\"'<>]*'",
        r"'/[^\"'<>]*/product/[^\"'<>]*'",
    ]
    
    print("\n🔍 Extracting product URLs using regex (no JSON parsing)...")
    all_urls = set()
    
    for pattern in product_url_patterns:
        matches = re.findall(pattern, html, re.I)
        all_urls.update(matches)
    
    # URL を正規化（相対パスを絶対パスに変換）
    normalized_urls = set()
    for url in all_urls:
        # クォートを除去
        url = url.strip('"\'')
        # 相対パスの場合は絶対パスに変換
        if url.startswith('/'):
            url = f'https://www.moncler.com{url}'
        normalized_urls.add(url)
    
    print(f"\n✅ Found {len(normalized_urls)} potential product URLs")
    for url in sorted(list(normalized_urls))[:30]:
        print(f"  - {url}")
    
    # Moncler ドメインの商品 URL のみをフィルタ
    moncler_product_urls = [url for url in normalized_urls if 'moncler.com' in url and ('/products/' in url or '/product/' in url)]
    print(f"\n🎯 Moncler product URLs: {len(moncler_product_urls)}")
    for url in sorted(moncler_product_urls)[:20]:
        print(f"  - {url}")


if __name__ == "__main__":
    run_id = sys.argv[1] if len(sys.argv) > 1 else "20251203_112454_918"
    extract_product_urls_regex(run_id)

