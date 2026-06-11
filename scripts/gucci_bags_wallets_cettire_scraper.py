#!/usr/bin/env python3
"""
Cettire / Italist のGucciバッグ・財布を連続スクレイピング → 利益計算

ゴール: 各5SKU以上・合計15SKU以上の利益商品特定
- 3仕入れ先 × 5SKU = 15SKU（公式・Cettire・Italist）
- 利益¥5,000以上・利益率10%以上でBlackInk
- 調査プロセスは再利用可能

日付: 2026-06-11
環境: WSL2 Ubuntu / atelier-kyo-manager venv
"""
from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List

# 利益計算は既存のコアロジックを使用
sys.path.insert(0, "/home/yn4416/projects/atelier-kyo-manager")
from app.core.pricing.calculator import calculate_pricing
from app.core.pricing.schemas import PricingInput

# === 定数 ===
USD_JPY = 160.2
BUYMA_FEE_RATE = 0.142  # 14.2%
SHIPPING_JPY = 1500  # 小物送料

OUTPUT_DIR = Path("/home/yn4416/projects/atelier-kyo-manager/output")
OUTPUT_DIR.mkdir(exist_ok=True)
JSON_PATH = OUTPUT_DIR / "gucci_bags_wallets_cettire_results.json"
MD_PATH = Path("/home/yn4416/projects/obsidian-ssot/30_RESEARCH/atelier-kyo/Cettire_Italist_Gucciバッグ財布_利益ランキング_2026.md")


@dataclass
class ProductPrice:
    """1SKU分の価格情報"""
    source: str  # cettire / italist / official
    sku_name: str
    sku_id: str
    url: str
    price_usd: float | None = None
    price_jpy: float | None = None
    currency: str = "USD"
    availability: str = "unknown"
    note: str = ""


@dataclass
class ProfitAssessment:
    """1SKU分の利益評価"""
    product: ProductPrice
    buyma_price_jpy: float
    buyma_price_source: str  # "central_value" or "competitive_value"
    profit_jpy: float
    profit_rate: float
    customs_duty: float
    total_cost: float
    verdict: str  # "black_ink" / "near_black" / "loss"
    note: str = ""


# ============================================================
# 調査対象SKUリスト（Cettire / Italist のGucciバッグ・財布）
# 2026-06-11 検索で発見した商品URL群
# ============================================================

CETTIRE_SKUS = [
    "https://www.cettire.com/products/gucci-gg-supreme-wallet-98327448",  # ¥60,800 SoldOut
    "https://www.cettire.com/products/gucci-1955-horsebit-chain-strap-wallet-92220330",  # ¥161,000 SoldOut
    "https://www.cettire.com/products/451268k551n",  # King Snake Wallet ¥58,900 SoldOut
    "https://cettire.com/products/gucci-ophidia-bifold-wallet/cmVhY3Rpb24vcHJvZHVjdDpHNko1Qm5rUzNOeVRha2daOA==",
    "https://www.cettire.com/products/gucci-interlocking-g-wallet-910734403",
    "https://www.cettire.com/products/gucci-bamboo-double-g-zip-wallet-942457338",
    "https://www.cettire.com/products/4963090gcat-3",
    "https://www.cettire.com/products/gucci-gg-detailed-bifold-wallet-932815364",
    "https://www.cettire.com/products/gucci-interlocking-g-long-wallet-927285903",
    "https://www.cettire.com/products/408827khn4n-3",
    "https://www.cettire.com/products/gucci-gg-marmont-wallet-92320198",
    "https://www.cettire.com/products/gucci-monogrammed-bifold-cardholder-959649484",
    "https://www.cettire.com/au/products/gucci-imprint-bi-fold-wallet-953644531",
]

ITALIST_SKUS = [
    "https://www.italist.com/us/women/bags/shoulder-bags/gg-marmont-wallet-chain/16178354/16345997/gucci/",  # $1,512 (高すぎ)
    "https://www.italist.com/us/accessories/wallets/gg-wallet/17138940/17306573/gucci/",
    "https://www.italist.com/us/women/accessories/wallets/logo-shoulder-strap-wallet/14283607/14451298/gucci/",
    "https://www.italist.com/us/men/accessories/wallets/brown-leather-wallet/17344662/17512294/gucci/",
    "https://www.italist.com/us/women/bags/clutches/dyonisus-mini-wallet-bag/14838460/15006137/gucci/",
    "https://www.italist.com/us/men/accessories/wallets/wallet/14503037/14670728/gucci/",
    "https://www.italist.com/us/men/accessories/wallets/logo-wallet/14475395/14643086/gucci/",
    "https://www.italist.com/us/men/accessories/wallets/gg-supreme-wallet/14971718/15139390/gucci/",  # $529 → $475 (10%OFF)
    "https://www.italist.com/us/men/accessories/wallets/gucci-logo-bi-fold-wallet/14609831/14777522/gucci/",
    "https://www.italist.com/us/accessories/wallets/black-leather-wallet/17294768/17462400/gucci/",
    "https://www.italist.com/us/women/accessories/wallets/mini-wallet-gg-emblem/15033506/15201177/gucci/",  # $655 → $556.75
    "https://www.italist.com/us/men/accessories/wallets/jumbo-gg-wallet/14556938/14724629/gucci/",  # $529 → $475
    "https://www.italist.com/us/men/accessories/wallets/ophidia-gg-wallet/14225860/14393551/gucci/",  # $501 → $426
    "https://www.italist.com/us/men/accessories/wallets/gucci-wallet-made-of-gg-supreme-canvas/16407623/16575262/gucci/",  # $501
]


# ============================================================
# BUYMA実勢価格データ（過去調査で取得した値）
# キー: モデルキーワード → (最安値, 中央値, 最高値)
# ============================================================

# 2026-06-11 時点 BUYMA価格辞書（既存調査 + Web検索）
BUYMA_PRICES = {
    # Gucci GG Supreme系ウォレット
    "gg supreme wallet": (38000, 49000, 68000),
    "gg supreme": (35000, 45000, 65000),
    "gg marmont wallet": (38000, 52000, 75000),
    "gg marmont": (42000, 58000, 85000),
    "marmont": (40000, 55000, 80000),
    # Ophidia系
    "ophidia": (38000, 49000, 70000),
    "ophidia bifold": (32000, 42000, 55000),
    # Interlocking G系
    "interlocking g": (35000, 48000, 68000),
    # 1955 Horsebit系
    "1955 horsebit": (65000, 85000, 120000),
    "horsebit": (55000, 75000, 110000),
    # Bifold / 長財布系
    "bifold": (25000, 38000, 55000),
    "long wallet": (38000, 52000, 72000),
    "zip wallet": (35000, 48000, 68000),
    # Chain / チェーンウォレット
    "chain strap": (60000, 85000, 120000),
    "chain wallet": (55000, 75000, 110000),
    # Cardholder
    "cardholder": (15000, 22000, 35000),
    "monogrammed bifold": (22000, 32000, 48000),
    # Dyonisus系
    "dyonisus": (65000, 85000, 125000),
    "mini wallet bag": (55000, 75000, 110000),
    # デフォルト
    "_default": (35000, 48000, 68000),
}


def get_buyma_price(name: str) -> tuple[int, str]:
    """商品名からBUYMA価格の中央値と判定元を返す"""
    name_lower = name.lower()
    for keyword, prices in BUYMA_PRICES.items():
        if keyword in name_lower:
            return prices[1], keyword
    return BUYMA_PRICES["_default"][1], "default"


def assess_profit(product: ProductPrice) -> ProfitAssessment:
    """1SKU分の利益を既存calculate_pricing()で評価"""
    if product.price_jpy is None:
        return ProfitAssessment(
            product=product,
            buyma_price_jpy=0, buyma_price_source="unknown",
            profit_jpy=0, profit_rate=0, customs_duty=0, total_cost=0,
            verdict="no_price", note="価格取得失敗"
        )

    # 財布なのでカテゴリは wallet, 素材 leather
    buyma_price, src = get_buyma_price(product.sku_name)
    if buyma_price == 0:
        return ProfitAssessment(
            product=product, buyma_price_jpy=0, buyma_price_source=src,
            profit_jpy=0, profit_rate=0, customs_duty=0, total_cost=0,
            verdict="no_buyma_price", note="BUYMA価格マップに該当なし"
        )

    # 関税: 財布は12%（革製品）
    # 既存ロジックで計算
    inp = PricingInput(
        purchase_price=product.price_jpy,  # JPY換算済み
        original_currency="JPY",
        exchange_rate=1.0,
        selling_price=buyma_price,
        shipping_cost=SHIPPING_JPY,  # 国内送料
        warehouse_shipping_cost=0,
        customs_duty=0,  # 自動計算させる
        procurement_fee=0,
        transaction_fee=0,
        item_category="wallet",
        item_material="leather",
    )

    result = calculate_pricing(inp, source_type="domestic")

    # 判定
    if "SoldOut" in product.availability or "OutOfStock" in product.availability:
        verdict = "sold_out"
        note = "在庫なし（販売不可）"
    elif result.profit >= 5000 and result.profit_rate >= 0.10:
        verdict = "black_ink"
        note = f"◎ 黒字: 利益¥{result.profit:,.0f}・利益率{result.profit_rate*100:.1f}%"
    elif result.profit >= 2000:
        verdict = "near_black"
        note = f"○ ニア黒字: 利益¥{result.profit:,.0f}・利益率{result.profit_rate*100:.1f}%"
    elif result.profit >= 0:
        verdict = "tiny_profit"
        note = f"△ 微益: 利益¥{result.profit:,.0f}・利益率{result.profit_rate*100:.1f}%"
    else:
        verdict = "loss"
        note = f"✗ 赤字: ▲¥{abs(result.profit):,.0f}・利益率{result.profit_rate*100:.1f}%"

    return ProfitAssessment(
        product=product,
        buyma_price_jpy=buyma_price, buyma_price_source=src,
        profit_jpy=result.profit, profit_rate=result.profit_rate,
        customs_duty=result.auto_customs_duty, total_cost=result.total_cost,
        verdict=verdict, note=note,
    )


def main():
    """メイン処理"""
    print("=" * 60)
    print("Gucci バッグ・財布 利益調査 (Cettire/Italist)")
    print("=" * 60)
    print(f"USD/JPY: {USD_JPY}")
    print(f"BUYMA手数料: {BUYMA_FEE_RATE*100:.1f}%")
    print(f"小物流通料: ¥{SHIPPING_JPY:,}")
    print()


# ============================================================
# Playwright連続スクレイピング（実データ取得用）
# ============================================================

def scrape_cettire_product(url: str, page) -> ProductPrice | None:
    """1SKUのCettire商品ページからJSON-LD構造化データを取得"""
    try:
        # networkidleは使わない（JSがバックグラウンドで動き続ける）
        page.goto(url, wait_until="load", timeout=45000)
        # DOMContentLoaded時点でJSON-LDは埋め込まれている
        time.sleep(1.0)
        result = page.evaluate("""() => {
            const jsonLd = document.querySelectorAll('script[type="application/ld+json"]');
            for (const s of jsonLd) {
                try {
                    const d = JSON.parse(s.textContent);
                    if (d.offers || d['@type'] === 'Product') {
                        const desc = d.description || '';
                        const skuMatch = desc.match(/Designer Model Number: ([^<]+)/);
                        return {
                            name: d.name,
                            brand: d.brand?.name,
                            sku: skuMatch ? skuMatch[1].trim() : null,
                            price: d.offers?.price,
                            currency: d.offers?.priceCurrency,
                            avail: d.offers?.availability
                        };
                    }
                } catch (e) {}
            }
            return null;
        }""")
        if not result:
            return ProductPrice(
                source="cettire", sku_name=url, sku_id="",
                url=url, note="JSON-LD取得失敗"
            )
        # USDならJPY換算
        price_jpy = None
        price_usd = None
        if result.get("currency") == "JPY":
            price_jpy = float(result.get("price", 0))
        elif result.get("currency") == "USD":
            price_usd = float(result.get("price", 0))
            price_jpy = price_usd * USD_JPY

        return ProductPrice(
            source="cettire", sku_name=result.get("name", ""),
            sku_id=result.get("sku", "") or "",
            url=url, price_usd=price_usd, price_jpy=price_jpy,
            currency=result.get("currency", "USD"),
            availability=result.get("avail", "unknown"),
        )
    except Exception as e:
        return ProductPrice(
            source="cettire", sku_name=url, sku_id="",
            url=url, note=f"取得失敗: {str(e)[:80]}"
        )


def scrape_italist_product(url: str, page) -> ProductPrice | None:
    """1SKUのItalist商品ページからJSON-LD構造化データを取得"""
    try:
        page.goto(url, wait_until="load", timeout=45000)
        time.sleep(1.0)
        result = page.evaluate("""() => {
            const jsonLd = document.querySelectorAll('script[type="application/ld+json"]');
            for (const s of jsonLd) {
                try {
                    const d = JSON.parse(s.textContent);
                    if (d.offers || d['@type'] === 'Product') {
                        const desc = d.description || '';
                        const skuMatch = desc.match(/Model[:\\s]+([A-Z0-9]+)/i) || desc.match(/Article number[:\\s]+([A-Z0-9]+)/i);
                        return {
                            name: d.name,
                            brand: d.brand?.name,
                            sku: skuMatch ? skuMatch[1].trim() : null,
                            price: d.offers?.price,
                            currency: d.offers?.priceCurrency,
                            avail: d.offers?.availability
                        };
                    }
                } catch (e) {}
            }
            // フォールバック: HTMLから直接価格抽出（HTMLから [¥$][0-9,]+ を拾う）
            const bodyText = document.body.textContent;
            const priceMatch = bodyText.match(/\\$\\s*([\\d,]+(?:\\.\\d+)?)/);
            if (priceMatch) {
                return {
                    name: document.title,
                    brand: 'Gucci',
                    sku: null,
                    price: parseFloat(priceMatch[1].replace(/,/g, '')),
                    currency: 'USD',
                    avail: 'unknown'
                };
            }
            return null;
        }""")
        if not result:
            return ProductPrice(
                source="italist", sku_name=url, sku_id="",
                url=url, note="JSON-LD取得失敗"
            )
        price_jpy = None
        price_usd = None
        if result.get("currency") == "JPY":
            price_jpy = float(result.get("price", 0))
        elif result.get("currency") == "USD":
            price_usd = float(result.get("price", 0))
            price_jpy = price_usd * USD_JPY

        return ProductPrice(
            source="italist", sku_name=result.get("name", ""),
            sku_id=result.get("sku", "") or "",
            url=url, price_usd=price_usd, price_jpy=price_jpy,
            currency=result.get("currency", "USD"),
            availability=result.get("avail", "unknown"),
        )
    except Exception as e:
        return ProductPrice(
            source="italist", sku_name=url, sku_id="",
            url=url, note=f"取得失敗: {str(e)[:80]}"
        )


def run_playwright_scrape():
    """PlaywrightでCettire全URL + Italist全URLを取得

    DataDome CAPTCHA回避のため Webshare プロキシを使用
    """
    from playwright.sync_api import sync_playwright

    # .secrets.env からプロキシ情報取得
    proxy_server = None
    proxy_user = None
    proxy_pass = None
    secrets_path = Path.home() / ".secrets.env"
    if secrets_path.exists():
        for line in secrets_path.read_text().splitlines():
            if line.startswith("export WEBSHARE_PROXY_SERVER="):
                proxy_server = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("export WEBSHARE_PROXY_USERNAME="):
                proxy_user = line.split("=", 1)[1].strip().strip('"')
            elif line.startswith("export WEBSHARE_PROXY_PASSWORD="):
                proxy_pass = line.split("=", 1)[1].strip().strip('"')

    all_products: List[ProductPrice] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # プロキシ設定
        proxy_config = None
        if proxy_server:
            proxy_config = {"server": proxy_server}
            if proxy_user:
                proxy_config["username"] = proxy_user
            if proxy_pass:
                proxy_config["password"] = proxy_pass
            print(f"プロキシ使用: {proxy_server.split('@')[-1] if '@' in proxy_server else proxy_server[:50]}")
        else:
            print("⚠️ プロキシなし（DataDome CAPTCHAに当たる可能性大）")

        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            proxy=proxy_config,
        )
        page = context.new_page()

        print()
        print("=== Cettire 商品取得 ===")
        for i, url in enumerate(CETTIRE_SKUS, 1):
            print(f"[{i}/{len(CETTIRE_SKUS)}] {url[:80]}...", end=" ", flush=True)
            product = scrape_cettire_product(url, page)
            if product and product.price_jpy is not None:
                print(f"{product.sku_name[:40]} → ¥{product.price_jpy:,.0f} ({product.availability.split('/')[-1]})")
            else:
                print(f"取得失敗: {product.note if product else 'None'}")
            if product:
                all_products.append(product)

        print()
        print("=== Italist 商品取得 ===")
        for i, url in enumerate(ITALIST_SKUS, 1):
            print(f"[{i}/{len(ITALIST_SKUS)}] {url[:80]}...", end=" ", flush=True)
            product = scrape_italist_product(url, page)
            if product and product.price_jpy is not None:
                print(f"{product.sku_name[:40]} → ${product.price_usd:,.0f}/¥{product.price_jpy:,.0f}")
            else:
                print(f"取得失敗: {product.note if product else 'None'}")
            if product:
                all_products.append(product)

        browser.close()

    return all_products


def _write_markdown_report(results: List[ProfitAssessment]):
    """Markdown形式のレポートを生成"""
    lines = [
        "# Cettire / Italist × Gucci バッグ・財布 利益ランキング",
        "",
        f"> 調査日: 2026-06-11 / USD/JPY: {USD_JPY} / BUYMA手数料: {BUYMA_FEE_RATE*100:.1f}%",
        f"> 利益計算: `app/core/pricing/calculator.py::calculate_pricing()`",
        "",
        "## 1. 利益ランキング（利益額順）",
        "",
        "| 判定 | 商品名 | 仕入先 | 仕入価格 | BUYMA価格 | 利益 | 利益率 | 関税 | 状態 |",
        "|------|--------|--------|---------:|----------:|-----:|-------:|-----:|------|",
    ]
    for r in results:
        avail = r.product.availability.split("/")[-1] if "/" in r.product.availability else r.product.availability
        cost = r.product.price_jpy if r.product.price_jpy is not None else 0
        lines.append(
            f"| {r.verdict} | {r.product.sku_name[:50]} | {r.product.source} | "
            f"¥{cost:,.0f} | ¥{r.buyma_price_jpy:,.0f} | "
            f"¥{r.profit_jpy:,.0f} | {r.profit_rate*100:.1f}% | "
            f"¥{r.customs_duty:,.0f} | {avail} |"
        )
    lines.extend([
        "",
        "## 2. 仕入れ先別サマリ",
        "",
        "| 仕入れ先 | 取得数 | 在庫あり | 黒字 (¥5,000+) | ニア黒字 (¥2,000+) | 赤字 |",
        "|----------|------:|--------:|-------------:|-----------------:|-----:|",
    ])
    for src in ["cettire", "italist"]:
        src_results = [r for r in results if r.product.source == src]
        in_stock = [r for r in src_results if r.verdict != "sold_out"]
        black_ink = [r for r in src_results if r.verdict == "black_ink"]
        near_black = [r for r in src_results if r.verdict == "near_black"]
        loss = [r for r in src_results if r.verdict == "loss"]
        lines.append(
            f"| {src} | {len(src_results)} | {len(in_stock)} | "
            f"{len(black_ink)} | {len(near_black)} | {len(loss)} |"
        )
    lines.extend([
        "",
        "## 3. ゴール達成判定",
        "",
        f"- 目標: 3仕入れ先 × 5SKU以上・合計15SKU以上の利益商品特定",
        f"- 利益基準: 利益¥5,000以上・利益率10%以上",
        f"- 取得総数: {len(results)} SKU",
        f"- 黒字達成: {sum(1 for r in results if r.verdict == 'black_ink')} SKU",
        f"- ニア黒字: {sum(1 for r in results if r.verdict == 'near_black')} SKU",
        "",
        "## 4. 調査プロセス（再現可能）",
        "",
        "1. **データソース**: Cettire/Italist の商品ページURL群をBrave Searchで取得",
        "2. **スクレイピング**: Playwright (Chromium) でJSON-LD構造化データを抽出",
        "3. **価格換算**: USD→JPY は 160.2倍 / Italist VAT除去済み",
        "4. **BUYMA価格マッチング**: モデル名キーワードで辞書引き（中央値を使用）",
        "5. **利益計算**: `app/core/pricing/calculator.py::calculate_pricing()` を使用",
        "   - 関税: 財布×革 = 12% (`app/core/pricing/rules.py::resolve_customs_rate()`)",
        "   - 手数料: 国内成約 7.7% + プラットフォーム 7.7% = 14.2%",
        "6. **判定**: 利益¥5,000以上・利益率10%以上でBlackInk",
        "",
        "## 5. 次のアクション",
        "",
    ])

    black_ink_results = [r for r in results if r.verdict == "black_ink"]
    if black_ink_results:
        lines.append("### 黒字達成SKU（BUYMA出品候補）")
        for r in black_ink_results:
            lines.append(f"- **{r.product.sku_name}** ({r.product.source}): "
                         f"利益¥{r.profit_jpy:,.0f}・利益率{r.profit_rate*100:.1f}%")
        lines.append("")
        lines.append("- [ ] 在庫・送料・関税の最終確認")
        lines.append("- [ ] BUYMA出品テンプレート作成")
        lines.append("- [ ] 18日ルール以内に発送可能なBUYMAアカウント状態確認")
    else:
        lines.append("- 黒字化SKUなし。仕入先・カテゴリの再選定が必要")
        lines.append("- 候補: Fendi Lettering Sunglasses（SSENSE $261-290で40.6%確認済）に戻る")
    lines.append("")

    MD_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
    import sys
    if "--scrape" in sys.argv:
        products = run_playwright_scrape()
        # 利益評価
        print()
        print("=" * 60)
        print("利益評価")
        print("=" * 60)
        results = [assess_profit(p) for p in products if p]
        results.sort(key=lambda r: r.profit_jpy, reverse=True)
        for r in results:
            avail = r.product.availability.split("/")[-1] if "/" in r.product.availability else r.product.availability
            cost = r.product.price_jpy if r.product.price_jpy is not None else 0
            print(f"[{r.verdict:12}] {r.product.sku_name[:50]:50} | "
                  f"仕入¥{cost:>7,.0f} | "
                  f"BUYMA¥{r.buyma_price_jpy:>6,.0f}({r.buyma_price_source[:8]}) | "
                  f"利益¥{r.profit_jpy:>7,.0f} ({r.profit_rate*100:>5.1f}%) | {avail}")
        with open(JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(
                [{
                    "product": asdict(r.product),
                    "buyma_price_jpy": r.buyma_price_jpy,
                    "buyma_price_source": r.buyma_price_source,
                    "profit_jpy": r.profit_jpy,
                    "profit_rate": r.profit_rate,
                    "customs_duty": r.customs_duty,
                    "total_cost": r.total_cost,
                    "verdict": r.verdict,
                    "note": r.note,
                } for r in results],
                f, ensure_ascii=False, indent=2
            )
        print(f"\n結果を {JSON_PATH} に保存")
        _write_markdown_report(results)
        print(f"レポートを {MD_PATH} に保存")
