#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DOM snapshot 解析スクリプト

Moncler Phase 1.5 の dry-run で生成された DOM snapshot を解析し、
PLP → PDP リンクの抽出が失敗している原因を特定する。

Usage:
    python tools/analyze_dom_snapshot.py <run_id>
    python tools/analyze_dom_snapshot.py 20251203_020753_296
"""
from __future__ import annotations
import sys
import re
from pathlib import Path
from html.parser import HTMLParser
from collections import defaultdict
from typing import List, Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class ProductLinkParser(HTMLParser):
    """PDP リンクを抽出する HTML パーサー"""
    
    def __init__(self):
        super().__init__()
        self.links: List[Dict[str, str]] = []
        self.current_tag = None
        self.current_attrs = {}
        
    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        self.current_attrs = dict(attrs)
        
        # href 属性を持つ要素をチェック
        href = self.current_attrs.get('href', '')
        if href and '/products/' in href:
            self.links.append({
                'tag': tag,
                'href': href,
                'attrs': self.current_attrs,
                'text': '',
            })
        
        # data-* 属性をチェック
        for attr_name, attr_value in attrs:
            if attr_name.startswith('data-') and attr_value and '/products/' in attr_value:
                self.links.append({
                    'tag': tag,
                    'href': None,
                    'data_attr': attr_name,
                    'data_value': attr_value,
                    'attrs': self.current_attrs,
                    'text': '',
                })
    
    def handle_data(self, data):
        if self.links and self.current_tag:
            # 最後に追加したリンクにテキストを追加
            if self.links[-1]['tag'] == self.current_tag:
                self.links[-1]['text'] += data.strip()


def analyze_dom_snapshot(run_id: str) -> None:
    """DOM snapshot を解析して PDP リンクの情報を出力"""
    
    run_dir = ROOT / "instance" / "runs" / run_id
    if not run_dir.exists():
        print(f"❌ Run directory not found: {run_dir}")
        return
    
    html_file = run_dir / "failure_dom.html"
    if not html_file.exists():
        print(f"❌ HTML file not found: {html_file}")
        return
    
    print(f"📁 Analyzing DOM snapshot: {html_file}")
    print(f"📊 File size: {html_file.stat().st_size / 1024 / 1024:.2f} MB")
    print()
    
    # HTML ファイルを読み込み
    try:
        html_content = html_file.read_text(encoding='utf-8', errors='ignore')
    except Exception as e:
        print(f"❌ Failed to read HTML file: {e}")
        return
    
    # パーサーで解析
    parser = ProductLinkParser()
    parser.feed(html_content)
    
    print(f"🔍 Found {len(parser.links)} potential PDP links")
    print()
    
    # リンクの種類別に分類
    href_links = [l for l in parser.links if l.get('href')]
    data_links = [l for l in parser.links if l.get('data_attr')]
    
    print(f"📌 Links with href attribute: {len(href_links)}")
    print(f"📌 Links with data-* attribute: {len(data_links)}")
    print()
    
    # href リンクの詳細
    if href_links:
        print("=" * 80)
        print("🔗 Links with href attribute:")
        print("=" * 80)
        for i, link in enumerate(href_links[:10], 1):
            print(f"\n[{i}] Tag: {link['tag']}")
            print(f"    href: {link['href']}")
            print(f"    Text: {link['text'][:50] if link['text'] else '(empty)'}")
            # 重要な属性を表示
            important_attrs = ['class', 'data-test', 'data-testid', 'id', 'aria-label']
            for attr in important_attrs:
                if attr in link['attrs']:
                    print(f"    {attr}: {link['attrs'][attr]}")
    
    # data-* リンクの詳細
    if data_links:
        print("\n" + "=" * 80)
        print("🔗 Links with data-* attribute:")
        print("=" * 80)
        for i, link in enumerate(data_links[:10], 1):
            print(f"\n[{i}] Tag: {link['tag']}")
            print(f"    {link['data_attr']}: {link['data_value']}")
            print(f"    Text: {link['text'][:50] if link['text'] else '(empty)'}")
            # 重要な属性を表示
            important_attrs = ['class', 'data-test', 'data-testid', 'id', 'aria-label']
            for attr in important_attrs:
                if attr in link['attrs']:
                    print(f"    {attr}: {link['attrs'][attr]}")
    
    # セレクタパターンの分析
    print("\n" + "=" * 80)
    print("🎯 Selector Pattern Analysis:")
    print("=" * 80)
    
    # Moncler の site_config から実際のセレクタを確認
    moncler_selectors = [
        "a[href*='/products/']",
        "a[href*='/product/']",
        "a[href*='/p-']",
        "div[data-test='product-card']",
        "li.product-grid__item",
        ".product-cell",
        "[data-qa='product-tile']",
        "div[data-test='product-grid']",
        "ul.products-grid",
        "div.plp-grid",
    ]
    
    print("\n📋 Moncler site_config selectors matching status:")
    
    # HTML 全体から要素を検索
    for selector in moncler_selectors:
        count = 0
        if selector.startswith('a[href'):
            # href パターン
            pattern = selector.split("'")[1] if "'" in selector else selector.split('"')[1]
            count = len(re.findall(rf'href=["\']([^"\']*{re.escape(pattern)}[^"\']*)["\']', html_content, re.I))
        elif selector.startswith('['):
            # 属性セレクタ
            if 'data-test=' in selector:
                pattern = selector.split("'")[1] if "'" in selector else selector.split('"')[1]
                count = len(re.findall(rf'data-test=["\']{re.escape(pattern)}["\']', html_content, re.I))
            elif 'data-qa=' in selector:
                pattern = selector.split("'")[1] if "'" in selector else selector.split('"')[1]
                count = len(re.findall(rf'data-qa=["\']{re.escape(pattern)}["\']', html_content, re.I))
        elif selector.startswith('.'):
            # クラスセレクタ
            class_name = selector[1:]
            count = len(re.findall(rf'class=["\'][^"\']*{re.escape(class_name)}[^"\']*["\']', html_content, re.I))
        elif ' ' in selector:
            # 複合セレクタ（簡易版）
            parts = selector.split()
            if len(parts) == 2:
                # 親要素と子要素
                parent, child = parts
                if parent.startswith('div[data-test='):
                    pattern = parent.split("'")[1] if "'" in parent else parent.split('"')[1]
                    # 親要素内に子要素が存在するか（簡易チェック）
                    parent_pattern = rf'<div[^>]*data-test=["\']{re.escape(pattern)}["\'][^>]*>'
                    matches = re.findall(parent_pattern, html_content, re.I)
                    if matches:
                        # 子要素のパターンもチェック
                        if child.startswith('a'):
                            count = len(re.findall(rf'{parent_pattern}.*?<a[^>]*>', html_content, re.I | re.DOTALL))
                        else:
                            count = len(matches)  # 簡易的に親要素の数
        else:
            # タグセレクタ
            tag = selector.split('[')[0] if '[' in selector else selector
            count = len(re.findall(rf'<{re.escape(tag)}[^>]*>', html_content, re.I))
        
        print(f"  {selector}: {count} matches")
    
    # HTML 全体から商品関連の要素を探す
    print("\n" + "=" * 80)
    print("🔍 Deep Analysis:")
    print("=" * 80)
    
    # 1. コンテナ要素の存在確認
    print("\n📦 Container elements:")
    containers = [
        ("div[data-test='product-grid']", r'<div[^>]*data-test=["\']product-grid["\'][^>]*>'),
        ("ul.products-grid", r'<ul[^>]*class=["\'][^"\']*products-grid[^"\']*["\']'),
        ("div.plp-grid", r'<div[^>]*class=["\'][^"\']*plp-grid[^"\']*["\']'),
    ]
    for name, pattern in containers:
        matches = re.findall(pattern, html_content, re.I)
        print(f"  {name}: {len(matches)} found")
        if matches:
            # 最初のマッチの周辺を表示
            idx = html_content.find(matches[0])
            snippet = html_content[max(0, idx-50):min(len(html_content), idx+200)]
            print(f"    Sample: {snippet[:150]}...")
    
    # 2. カード要素の存在確認
    print("\n🃏 Card elements:")
    cards = [
        ("div[data-test='product-card']", r'<div[^>]*data-test=["\']product-card["\'][^>]*>'),
        ("li.product-grid__item", r'<li[^>]*class=["\'][^"\']*product-grid__item[^"\']*["\']'),
        (".product-cell", r'<[^>]*class=["\'][^"\']*product-cell[^"\']*["\']'),
        ("[data-qa='product-tile']", r'<[^>]*data-qa=["\']product-tile["\'][^>]*>'),
    ]
    for name, pattern in cards:
        matches = re.findall(pattern, html_content, re.I)
        print(f"  {name}: {len(matches)} found")
        if matches and len(matches) <= 5:
            for i, match in enumerate(matches[:3], 1):
                idx = html_content.find(match)
                snippet = html_content[max(0, idx):min(len(html_content), idx+200)]
                print(f"    [{i}] {snippet[:150]}...")
    
    # 3. JavaScript で埋め込まれた JSON データを探す
    print("\n📊 Embedded JSON data:")
    json_patterns = [
        (r'window\.__INITIAL_STATE__\s*=\s*({.+?});', '__INITIAL_STATE__'),
        (r'window\.__PRELOADED_STATE__\s*=\s*({.+?});', '__PRELOADED_STATE__'),
        (r'<script[^>]*type=["\']application/json["\'][^>]*>(.+?)</script>', 'application/json'),
        (r'data-product-id=["\']([^"\']+)["\']', 'data-product-id'),
    ]
    for pattern, name in json_patterns:
        matches = re.findall(pattern, html_content, re.I | re.DOTALL)
        if matches:
            print(f"  {name}: {len(matches)} found")
            if name == 'data-product-id' and len(matches) > 0:
                print(f"    Sample IDs: {', '.join(matches[:5])}")
    
    # 4. すべての a タグの href を確認（Moncler ドメインのみ）
    print("\n🔗 All <a> tags with Moncler domain:")
    moncler_links = re.findall(r'<a[^>]*href=["\'](https?://[^"\']*moncler\.com[^"\']*)["\']', html_content, re.I)
    if moncler_links:
        print(f"  Found {len(moncler_links)} links")
        unique_links = list(set(moncler_links))[:10]
        for link in unique_links:
            print(f"    - {link}")
    else:
        print("  No Moncler domain links found")
    
    # 5. 推奨セレクタの提案
    print("\n" + "=" * 80)
    print("💡 Recommended selectors:")
    print("=" * 80)
    
    # class 属性の分析
    class_counts = defaultdict(int)
    all_classes = re.findall(r'class=["\']([^"\']+)["\']', html_content, re.I)
    for class_str in all_classes:
        for cls in class_str.split():
            if 'product' in cls.lower() or 'tile' in cls.lower() or 'card' in cls.lower():
                class_counts[cls] += 1
    
    if class_counts:
        print("\n📌 Product/tile/card-related classes found:")
        for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            print(f"  .{cls}: {count} occurrences")
    
    # data-* 属性の分析
    data_attrs = defaultdict(int)
    all_data_attrs = re.findall(r'data-([^=]+)=["\']([^"\']+)["\']', html_content, re.I)
    for attr_name, attr_value in all_data_attrs:
        if 'product' in attr_name.lower() or 'tile' in attr_name.lower() or 'card' in attr_name.lower():
            data_attrs[attr_name] += 1
    
    if data_attrs:
        print("\n📌 Product/tile/card-related data attributes:")
        for attr, count in sorted(data_attrs.items(), key=lambda x: x[1], reverse=True):
            print(f"  [data-{attr}]: {count} occurrences")
    
    print("\n" + "=" * 80)
    print("✅ Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 最新の run_id を自動検出
        runs_dir = ROOT / "instance" / "runs"
        if runs_dir.exists():
            run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
            if run_dirs:
                run_id = run_dirs[0].name
                print(f"🔍 Auto-detected latest run_id: {run_id}")
                print()
            else:
                print("❌ No run directories found")
                sys.exit(1)
        else:
            print("❌ Runs directory not found")
            sys.exit(1)
    else:
        run_id = sys.argv[1]
    
    analyze_dom_snapshot(run_id)

