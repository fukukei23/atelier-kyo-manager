#!/usr/bin/env python3
"""
SSENSE Prada/Gucciサングラス スクレイピング & BUYMA利益計算スクリプト
Web Reader経由でSSENSE商品データを取得し、BUYMA実勢価格と照合して利益ランキングを作成。

日付: 2026-06-11
環境: WSL2 Ubuntu / atelier-kyo-manager venv
"""

import csv
import json
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# === 定数 ===
USD_JPY = 160.2
BUYMA_FEE_RATE = 0.142  # 14.2%（手数料+振込手数料込み）
SHIPPING_JPY = 1500  # 小物送料
OUTPUT_DIR = Path("/home/yn4416/projects/obsidian-ssot/30_RESEARCH/atelier-kyo")
CSV_PATH = Path("/home/yn4416/projects/atelier-kyo-manager/output/ssense_sunglasses_profit.csv")
MD_PATH = OUTPUT_DIR / "SSENSE_Prada_Gucci_サングラス利益ランキング_2026.md"


@dataclass
class SSENSEProduct:
    brand: str
    name: str
    price_usd: float
    sale_price_usd: float | None = None
    url: str = ""
    sku: str = ""
    gender: str = ""

    @property
    def effective_price_usd(self) -> float:
        return self.sale_price_usd if self.sale_price_usd else self.price_usd

    @property
    def cost_jpy(self) -> float:
        return self.effective_price_usd * USD_JPY


@dataclass
class BUYMAPrice:
    model_name: str
    min_price: float = 0  # 最安値（新品・送料込）
    median_price: float = 0  # 中央値近似
    max_price: float = 0
    sample_count: int = 0


@dataclass
class ProfitResult:
    product: SSENSEProduct
    buyma: BUYMAPrice
    profit: float = 0
    profit_rate: float = 0
    verdict: str = ""

    def calculate(self):
        """利益 = BUYMA価格 - (原価 + 送料 + BUYMA手数料)"""
        sell_price = self.buyma.median_price
        if sell_price <= 0:
            self.profit = 0
            self.profit_rate = 0
            self.verdict = "BUYMA価格不明"
            return

        cost = self.product.cost_jpy
        buyma_fee = sell_price * BUYMA_FEE_RATE
        total_cost = cost + SHIPPING_JPY + buyma_fee
        self.profit = sell_price - total_cost
        self.profit_rate = (self.profit / total_cost) * 100 if total_cost > 0 else 0

        if self.profit > 5000:
            self.verdict = "◎ 狙い目"
        elif self.profit > 2000:
            self.verdict = "○ 可能"
        elif self.profit > 0:
            self.verdict = "△ 微益"
        else:
            self.verdict = "✗ 損"


# ============================================================
# SSENSE商品データ（Web Readerで取得済みのデータをハードコード）
# ============================================================

SSENSE_PRADA_MEN = [
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 520, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 520, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Black Runway Sunglasses", "price": 605, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Gold 'Prada Eyewear Collection' Sunglasses", "price": 315, "sale": None},
    {"name": "Black 'Prada' Logo Sunglasses", "price": 490, "sale": None},
    {"name": "Black Symbole Sunglasses", "price": 550, "sale": None},
    {"name": "Tortoiseshell Rectangular Sunglasses", "price": 385, "sale": 550},
]

SSENSE_PRADA_WOMEN = [
    {"name": "Black Catwalk Butterfly Sunglasses", "price": 550, "sale": None},
    {"name": "Black Catwalk Rectangle Side Triangle Sunglasses", "price": 665, "sale": None},
    {"name": "Black Twisted Frame Sunglasses", "price": 286, "sale": 635},
    {"name": "Black Twisted Frame Square Sunglasses", "price": 254, "sale": 635},
    {"name": "Black Symbole Sunglasses", "price": 275, "sale": 550},
    {"name": "Black Triangle Logo Sunglasses", "price": 565, "sale": None},
    {"name": "Black Symbole Sunglasses", "price": 550, "sale": None},
    {"name": "Black Symbole Sunglasses", "price": 550, "sale": None},
    {"name": "Black Thick Cat Eye Acetate Sunglasses", "price": 520, "sale": None},
    {"name": "Black Pillow Cat Eye Acetate Sunglasses", "price": 520, "sale": None},
    {"name": "Black Square Shield Acetate Sunglasses", "price": 520, "sale": None},
    {"name": "Black Cat Eye Square Sunglasses", "price": 565, "sale": None},
    {"name": "Black 'Prada Symbole' Sunglasses", "price": 550, "sale": None},
    {"name": "Black Angular Rectangle Acetate Sunglasses", "price": 520, "sale": None},
    {"name": "White Symbole Sunglasses", "price": 248, "sale": 605},
    {"name": "Black Swing Sunglasses", "price": 309, "sale": 475},
    {"name": "Transparent Runway Sunglasses", "price": 242, "sale": 605},
    {"name": "Black Symbole Sunglasses", "price": 399, "sale": 570},
    {"name": "Brown Triangle Logo Sunglasses", "price": 565, "sale": None},
    {"name": "Black Rectangular Sunglasses", "price": 550, "sale": None},
]

SSENSE_GUCCI_MEN = [
    {"name": "Black & Grey Square Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black & Blue Square Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black & Grey Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black & Yellow Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black & Pink Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black Aviator Sunglasses", "price": 375, "sale": None},
    {"name": "Tortoiseshell & Grey Aviator Sunglasses", "price": 375, "sale": None},
    {"name": "Tortoiseshell & Brown Aviator Sunglasses", "price": 375, "sale": None},
    {"name": "Black Rectangular Sunglasses", "price": 350, "sale": None},
    {"name": "Tortoiseshell Rectangular Sunglasses", "price": 350, "sale": None},
    {"name": "Black Acetate Mask Sunglasses", "price": 1305, "sale": None},
    {"name": "Gold & Pink Square Glasses", "price": 435, "sale": None},
    {"name": "Tortoiseshell & Gold Aviator Sunglasses", "price": 505, "sale": None},
    {"name": "Silver Aviator Sunglasses", "price": 360, "sale": None},
    {"name": "Tortoiseshell Aviator Sunglasses", "price": 435, "sale": None},
    {"name": "Black Aviator Sunglasses", "price": 435, "sale": None},
    {"name": "Black & Orange Aviator Sunglasses", "price": 505, "sale": None},
    {"name": "Gold & Grey Aviator Sunglasses", "price": 360, "sale": None},
    {"name": "Gold Mask Sunglasses", "price": 795, "sale": None},
    {"name": "Black Rectangular Sunglasses", "price": 375, "sale": None},
    {"name": "Grey Rectangular Mirror Sunglasses", "price": 375, "sale": None},
    {"name": "Pink Oval Sunglasses", "price": 375, "sale": None},
    {"name": "Gold & Grey Rectangular Sunglasses", "price": 495, "sale": None},
    {"name": "Gold & Pink Rectangular Sunglasses", "price": 495, "sale": None},
    {"name": "Gold & Orange Rectangular Sunglasses", "price": 495, "sale": None},
    {"name": "Off-White Bold Rectangular Sunglasses", "price": 435, "sale": None},
    {"name": "Gold Skinny Flip-Up Sunglasses", "price": 725, "sale": None},
]

SSENSE_GUCCI_WOMEN = [
    {"name": "Gold Guillochet Square Sunglasses", "price": 550, "sale": None},
    {"name": "Gold & Pink Metal Aviator Sunglasses", "price": 435, "sale": None},
    {"name": "Gold Aviator Sunglasses", "price": 420, "sale": None},
    {"name": "Black Feminine Chic Sunglasses", "price": 450, "sale": None},
    {"name": "Tortoiseshell Feminine Chic Sunglasses", "price": 450, "sale": None},
    {"name": "Gold & Yellow Square Sunglasses", "price": 435, "sale": None},
    {"name": "Gold & Blue Square Sunglasses", "price": 435, "sale": None},
    {"name": "Black Urban Socks Round Sunglasses", "price": 375, "sale": None},
    {"name": "Tortoiseshell Round Striped Sunglasses", "price": 375, "sale": None},
    {"name": "Black Crystal Logo Cat-Eye Sunglasses", "price": 450, "sale": None},
    {"name": "Black Round Frame Sunglasses", "price": 305, "sale": None},
    {"name": "Black Pantos Sunglasses", "price": 275, "sale": None},
    {"name": "Gold Geometric Sunglasses", "price": 435, "sale": None},
    {"name": "Black Oversized Rectangular Sunglasses", "price": 350, "sale": None},
    {"name": "Black Sylvie Sunglasses", "price": 345, "sale": None},
    {"name": "Gold Square Frame Flip-Up Sunglasses", "price": 725, "sale": None},
    {"name": "Gold & Brown Round Vintage Web Sunglasses", "price": 415, "sale": None},
    {"name": "Black Round GG Sunglasses", "price": 350, "sale": None},
    {"name": "Gold & Green Wide Rectangle Sunglasses", "price": 505, "sale": None},
    {"name": "Black Aviator Sunglasses", "price": 485, "sale": None},
    {"name": "Pink Urban Fork Sunglasses", "price": 485, "sale": None},
    {"name": "Black Strass Logo Sunglasses", "price": 1305, "sale": None},
    {"name": "Black Round Glitter GG Sunglasses", "price": 375, "sale": None},
    {"name": "Multicolor Rainbow Sylvie Sunglasses", "price": 415, "sale": None},
]

# ============================================================
# BUYMA実勢価格データ（Web Reader/Exaで取得済み）
# キー: モデルキーワード（小文字） → (最安値, 中央値近似, 最高値)
# ============================================================

BUYMA_PRADA_PRICES = {
    # Symbole / PR26ZS系
    "symbole": (42310, 47500, 54648),
    "pr26z": (42310, 47000, 65900),
    "pr17ws": (45520, 55242, 68200),
    "pr14ws": (44946, 48000, 59400),
    "pr13ys": (33500, 35000, 40000),
    "pr22zs": (37800, 38192, 47740),
    "pr09zs": (48490, 48500, 72600),
    "prb56": (48490, 49480, 75900),
    "pr10z": (47500, 48000, 75900),
    "pr65zs": (51678, 52000, 72600),
    "pr08ys": (51990, 52000, 65000),
    "logo": (36800, 41900, 48700),
    "runway": (54430, 60000, 118000),
    "catwalk": (39990, 45000, 66500),
    "cat eye": (26800, 39990, 56500),
    "butterfly": (32500, 40000, 55000),
    "rectangular": (36000, 40000, 55000),
    "twisted": (25400, 28600, 63500),
    "swing": (30900, 35000, 47500),
    "triangle": (48490, 53000, 75900),
    "metal plaque": (48490, 52000, 81400),
    "acetate": (52000, 54000, 60500),
    "linea rossa": (39980, 42000, 55000),
}

BUYMA_GUCCI_PRICES = {
    "aviator": (28954, 36000, 50500),
    "gg0636sk": (21500, 22500, 33000),
    "gg0637sk": (21300, 23490, 36300),
    "gg0650sk": (28460, 30000, 57200),
    "gg0684s": (37990, 40000, 68200),
    "gg0830sk": (19780, 22000, 43000),
    "gg0077sk": (28700, 31767, 45100),
    "gg0208s": (18236, 20000, 59000),
    "gg0258s": (15181, 18000, 40700),
    "gg0488s": (23560, 25000, 44000),
    "gg0763s": (17991, 20000, 36300),
    "gg1116s": (22610, 24000, 34100),
    "gg1279s": (142100, 142000, 184400),
    "gg1285sa": (33060, 35000, 59400),
    "gg1338sk": (21185, 23000, 39600),
    "gg1339sk": (22610, 24000, 36300),
    "gg1409sk": (34280, 36000, 54000),
    "gg1471sj": (43510, 44000, 71500),
    "gg1662sa": (29480, 31000, 49500),
    "gg1667sk": (35114, 37000, 61600),
    "gg1680s": (52800, 55000, 67500),
    "gg1684sa": (41330, 43000, 73770),
    "gg1690sk": (37810, 40000, 64900),
    "gg1818sk": (24510, 26000, 44000),
    "gg1175sk": (32495, 34000, 57200),
    "gg1178s": (28490, 30000, 61600),
    "gg1180sk": (31160, 33000, 51700),
    "gg1158sk": (28310, 30000, 51700),
    "gg1152s": (30535, 31000, 31480),
    "gg1140sk": (22610, 24000, 42900),
    "gg1143s": (52100, 53000, 67500),
    "gg1087s": (82500, 83000, 87000),
    "gg0022s": (23560, 25000, 38500),
    "gg0259s": (17081, 19000, 41800),
    "gg0808s": (20710, 22000, 51700),
    "gg0163sn": (20710, 22000, 44000),
    "gg0847sk": (16880, 18000, 44000),
    "gg1029sa": (36575, 38000, 77000),
    "gg1550sk": (37810, 40000, 64900),
    "gg1470sj": (41610, 43000, 67100),
    "gg1498sk": (37500, 39000, 62700),
    "gg0563skn": (28950, 30000, 44000),
    "mask": (79500, 80000, 130500),
    "strass": (130500, 130000, 130500),
    "flip-up": (72500, 73000, 72500),
    "rectangular": (35000, 37500, 50500),
    "round": (30500, 35000, 41500),
    "feminine chic": (45000, 46000, 48000),
    "cat-eye": (28700, 36000, 45000),
    "sylvie": (34500, 38000, 41500),
    "horsebit": (32500, 34000, 47200),
    "oversized": (35000, 36000, 50500),
    "urban": (37500, 38000, 48500),
    "pantos": (27500, 28000, 35000),
    "geometric": (43500, 44000, 50500),
}


def find_buyma_price(product_name: str, brand: str) -> BUYMAPrice:
    """商品名からBUYMA価格を検索"""
    name_lower = product_name.lower()
    price_db = BUYMA_PRADA_PRICES if brand == "Prada" else BUYMA_GUCCI_PRICES

    best_match = None
    best_score = 0

    for key, (min_p, med_p, max_p) in price_db.items():
        # キーワードマッチスコア
        key_parts = key.split()
        match_count = sum(1 for part in key_parts if part in name_lower)
        score = match_count / len(key_parts) if key_parts else 0

        # 完全一致ボーナス
        if key in name_lower:
            score += 1

        if score > best_score:
            best_score = score
            best_match = (min_p, med_p, max_p)

    if best_match:
        return BUYMAPrice(
            model_name=product_name,
            min_price=best_match[0],
            median_price=best_match[1],
            max_price=best_match[2],
            sample_count=3,
        )

    # フォールバック: ブランド別中央値
    if brand == "Prada":
        return BUYMAPrice(model_name=product_name, min_price=40000, median_price=48000, max_price=65000, sample_count=0)
    else:
        return BUYMAPrice(model_name=product_name, min_price=28000, median_price=35000, max_price=50000, sample_count=0)


def build_products() -> list[SSENSEProduct]:
    """全商品データを構築"""
    products = []

    for item in SSENSE_PRADA_MEN:
        products.append(SSENSEProduct(
            brand="Prada", name=item["name"],
            price_usd=item["sale"] or item["price"],
            sale_price_usd=item["sale"],
            gender="Men"
        ))

    for item in SSENSE_PRADA_WOMEN:
        products.append(SSENSEProduct(
            brand="Prada", name=item["name"],
            price_usd=item["sale"] or item["price"],
            sale_price_usd=item["sale"],
            gender="Women"
        ))

    for item in SSENSE_GUCCI_MEN:
        products.append(SSENSEProduct(
            brand="Gucci", name=item["name"],
            price_usd=item["sale"] or item["price"],
            sale_price_usd=item["sale"],
            gender="Men"
        ))

    for item in SSENSE_GUCCI_WOMEN:
        products.append(SSENSEProduct(
            brand="Gucci", name=item["name"],
            price_usd=item["sale"] or item["price"],
            sale_price_usd=item["sale"],
            gender="Women"
        ))

    return products


def calculate_profits(products: list[SSENSEProduct]) -> list[ProfitResult]:
    """利益計算"""
    results = []
    for p in products:
        buyma = find_buyma_price(p.name, p.brand)
        result = ProfitResult(product=p, buyma=buyma)
        result.calculate()
        results.append(result)
    return results


def export_csv(results: list[ProfitResult]):
    """CSV出力"""
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "順位", "ブランド", "モデル名", "性別",
            "SSENSE価格(USD)", "原価(JPY)", "BUYMA最安値",
            "BUYMA中央値", "利益(JPY)", "利益率", "判定"
        ])
        sorted_results = sorted(results, key=lambda r: r.profit, reverse=True)
        for i, r in enumerate(sorted_results, 1):
            writer.writerow([
                i, r.product.brand, r.product.name, r.product.gender,
                f"${r.product.effective_price_usd:.0f}",
                f"¥{r.product.cost_jpy:,.0f}",
                f"¥{r.buyma.min_price:,.0f}",
                f"¥{r.buyma.median_price:,.0f}",
                f"¥{r.profit:,.0f}",
                f"{r.profit_rate:.1f}%",
                r.verdict,
            ])
    print(f"CSV saved: {CSV_PATH}")


def export_markdown(results: list[ProfitResult]):
    """マークダウン出力"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sorted_results = sorted(results, key=lambda r: r.profit, reverse=True)

    lines = [
        "# SSENSE Prada / Gucci サングラス 利益ランキング",
        "",
        f"> 取得日: 2026-06-11",
        f"> USD/JPY: {USD_JPY}",
        f"> BUYMA手数料: {BUYMA_FEE_RATE*100:.1f}%",
        f"> 小物送料: ¥{SHIPPING_JPY:,}",
        "",
        f"> 利益 = BUYMA価格 - (SSENSE価格 x {USD_JPY} + ¥{SHIPPING_JPY:,} + BUYMA手数料)",
        "",
        "## データソース",
        "- SSENSE: Web Reader (mcp__web_reader) で全ページ取得",
        "- BUYMA: Exa Web Fetch + Brave Search で実勢価格取得",
        "",
        "## 注意事項",
        "- SSENSEのSALE品はセール価格を原価に使用",
        "- BUYMA価格は2026年6月時点の出品価格（送料・関税込）",
        "- 「BUYMA価格不明」は推定ブランド中央値で計算",
        "- SSENSEは「SALE ONLY」表示があり、現在セール品のみの可能性",
        "",
        "## Prada サングラス",
        "",
        "| 順位 | モデル名 | 性別 | SSENSE(USD) | 原価(JPY) | BUYMA最安値 | BUYMA中央値 | 利益(JPY) | 利益率 | 判定 |",
        "|-----:|---------|------|------------:|----------:|-----------:|-----------:|----------:|-------:|------|",
    ]

    prada_results = [r for r in sorted_results if r.product.brand == "Prada"]
    for i, r in enumerate(prada_results, 1):
        sale_mark = " (SALE)" if r.product.sale_price_usd else ""
        lines.append(
            f"| {i} | {r.product.name} | {r.product.gender} | "
            f"${r.product.effective_price_usd:.0f}{sale_mark} | "
            f"¥{r.product.cost_jpy:,.0f} | "
            f"¥{r.buyma.min_price:,.0f} | "
            f"¥{r.buyma.median_price:,.0f} | "
            f"¥{r.profit:,.0f} | "
            f"{r.profit_rate:.1f}% | {r.verdict} |"
        )

    lines.extend([
        "",
        "## Gucci サングラス",
        "",
        "| 順位 | モデル名 | 性別 | SSENSE(USD) | 原価(JPY) | BUYMA最安値 | BUYMA中央値 | 利益(JPY) | 利益率 | 判定 |",
        "|-----:|---------|------|------------:|----------:|-----------:|-----------:|----------:|-------:|------|",
    ])

    gucci_results = [r for r in sorted_results if r.product.brand == "Gucci"]
    for i, r in enumerate(gucci_results, 1):
        sale_mark = " (SALE)" if r.product.sale_price_usd else ""
        lines.append(
            f"| {i} | {r.product.name} | {r.product.gender} | "
            f"${r.product.effective_price_usd:.0f}{sale_mark} | "
            f"¥{r.product.cost_jpy:,.0f} | "
            f"¥{r.buyma.min_price:,.0f} | "
            f"¥{r.buyma.median_price:,.0f} | "
            f"¥{r.profit:,.0f} | "
            f"{r.profit_rate:.1f}% | {r.verdict} |"
        )

    # 全体サマリー
    profit_results = [r for r in sorted_results if r.profit > 0]
    avg_profit = sum(r.profit for r in profit_results) / len(profit_results) if profit_results else 0
    best = sorted_results[0] if sorted_results else None

    lines.extend([
        "",
        "## 全体サマリー",
        "",
        f"- 総商品数: {len(sorted_results)}",
        f"- 利益プス商品数: {len(profit_results)}",
        f"- 平均利益（利益プスのみ）: ¥{avg_profit:,.0f}",
        "",
    ])

    if best:
        lines.extend([
            "## トップ5 狙い目商品",
            "",
            "| # | ブランド | モデル | 原価 | BUYMA中央値 | 利益 | 利益率 |",
            "|--:|---------|--------|-----:|-----------:|-----:|-------:|",
        ])
        for i, r in enumerate(sorted_results[:5], 1):
            lines.append(
                f"| {i} | {r.product.brand} | {r.product.name} | "
                f"¥{r.product.cost_jpy:,.0f} | ¥{r.buyma.median_price:,.0f} | "
                f"¥{r.profit:,.0f} | {r.profit_rate:.1f}% |"
            )

    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Markdown saved: {MD_PATH}")


def main():
    print("=== SSENSE Sunglasses Profit Calculator ===")
    print(f"USD/JPY: {USD_JPY}")
    print()

    products = build_products()
    print(f"Total products: {len(products)}")
    print(f"  Prada Men: {len(SSENSE_PRADA_MEN)}")
    print(f"  Prada Women: {len(SSENSE_PRADA_WOMEN)}")
    print(f"  Gucci Men: {len(SSENSE_GUCCI_MEN)}")
    print(f"  Gucci Women: {len(SSENSE_GUCCI_WOMEN)}")

    results = calculate_profits(products)

    # 利益でソート
    sorted_results = sorted(results, key=lambda r: r.profit, reverse=True)

    # 上位10件表示
    print("\n=== Top 10 Profit Ranking ===")
    for i, r in enumerate(sorted_results[:10], 1):
        sale_info = " (SALE)" if r.product.sale_price_usd else ""
        print(
            f"{i:2d}. [{r.product.brand}] {r.product.name} | "
            f"${r.product.effective_price_usd:.0f}{sale_info} -> "
            f"¥{r.product.cost_jpy:,.0f} | "
            f"BUYMA ¥{r.buyma.median_price:,.0f} | "
            f"利益 ¥{r.profit:,.0f} ({r.profit_rate:.1f}%) | {r.verdict}"
        )

    # 下位5件
    print("\n=== Bottom 5 ===")
    for i, r in enumerate(sorted_results[-5:], len(sorted_results) - 4):
        print(
            f"{i:2d}. [{r.product.brand}] {r.product.name} | "
            f"利益 ¥{r.profit:,.0f} ({r.profit_rate:.1f}%) | {r.verdict}"
        )

    # エクスポート
    export_csv(results)
    export_markdown(results)
    print("\nDone!")


if __name__ == "__main__":
    main()
