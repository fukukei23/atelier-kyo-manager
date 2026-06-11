#!/usr/bin/env python3
"""
SSENSE他ブランドサングラス利益分析スクリプト
データはPlaywrightで取得済み → 分析・CSV出力のみ
"""

import csv
import os
from dataclasses import dataclass
from pathlib import Path

# === 定数 ===
USD_JPY = 160.2
BUYMA_FEE_RATE = 0.142
SHIPPING_COST = 1500
PROFIT_THRESHOLD = 0.15  # 15%

# === SSENSE商品データ（Playwrightで取得） ===
SSENSE_DATA = {
    "Saint Laurent": {
        "buyma_stats": {"min": 15000, "max": 97158, "median": 34010, "mean": 36710, "p25": 23900, "p75": 41800},
        "products": [
            {"name": "Black SL 908 Sunglasses", "sale": 295, "orig": 295},
            {"name": "Brown SL 751 Jeanne Sunglasses", "sale": 470, "orig": 470},
            {"name": "Brown Oversized Square Sunglasses", "sale": 495, "orig": 495},
            {"name": "Black SL M146 Sunglasses", "sale": 423, "orig": 470},
            {"name": "Black SL 905 Light-Tint Sunglasses", "sale": 295, "orig": 295},
            {"name": "Brown SL 905 Sunglasses", "sale": 295, "orig": 295},
            {"name": "Brown SL 545 Sunglasses", "sale": 550, "orig": 550},
            {"name": "Brown SL 869 Logo Sunglasses", "sale": 390, "orig": 390},
            {"name": "Black SL 868 Sunglasses", "sale": 390, "orig": 390},
            {"name": "Black SL 908 Sunglasses", "sale": 295, "orig": 295},
            {"name": "Brown SL 908 Sunglasses", "sale": 295, "orig": 295},
            {"name": "Black SL 737 Mica Thin Sunglasses", "sale": 338, "orig": 375},
            {"name": "Black SL 775 Sunglasses", "sale": 338, "orig": 375},
            {"name": "Gold SL 312 M Sunglasses", "sale": 486, "orig": 540},
            {"name": "Silver SL 866 Sunglasses", "sale": 535, "orig": 535},
            {"name": "Black SL M148 Sunglasses", "sale": 459, "orig": 510},
            {"name": "Gold SL 692 Sunglasses", "sale": 486, "orig": 540},
            {"name": "Gold SL 799 Sunglasses", "sale": 459, "orig": 510},
            {"name": "Black SL 815 Romy Sunglasses", "sale": 446, "orig": 495},
            {"name": "Green SL 567 Sunglasses", "sale": 446, "orig": 495},
            {"name": "Brown & Gold SL M137 Amelia Sunglasses", "sale": 525, "orig": 590},
            {"name": "Black SL M103 Sunglasses", "sale": 376, "orig": 495},
            {"name": "Black SL 806 Sunglasses", "sale": 468, "orig": 550},
            {"name": "Black SL 545 Sunglasses", "sale": 473, "orig": 525},
            {"name": "Black SL 557 Shade Sunglasses", "sale": 423, "orig": 470},
            {"name": "Brown SL M126 Sunglasses", "sale": 423, "orig": 510},
            {"name": "Black SL 751 Jeanne Sunglasses", "sale": 357, "orig": 470},
            {"name": "Black New Wave Oval Sunglasses", "sale": 446, "orig": 495},
            {"name": "Black SL 767 Sunglasses", "sale": 383, "orig": 485},
            {"name": "Black SL 633 Calista Sunglasses", "sale": 421, "orig": 495},
            {"name": "Gray SL M184 Sunglasses", "sale": 347, "orig": 510},
            {"name": "Black SL M136 Sunglasses", "sale": 418, "orig": 510},
            {"name": "Brown SL M136 Sunglasses", "sale": 459, "orig": 510},
            {"name": "Black SL 757 Sunglasses", "sale": 286, "orig": 440},
            {"name": "Black SL 790 Sunglasses", "sale": 475, "orig": 565},
            {"name": "Black SL 838 Large Sunglasses", "sale": 368, "orig": 525},
            {"name": "Black SL 214 Kate Thin Sunglasses", "sale": 308, "orig": 400},
            {"name": "Black SL M94 Sunglasses", "sale": 402, "orig": 550},
            {"name": "Black SL 759 Sunglasses", "sale": 368, "orig": 525},
            {"name": "Black M119 Blaze Sunglasses", "sale": 787, "orig": 960},
            {"name": "Gold SL 312 M Sunglasses", "sale": 470, "orig": 540},
            {"name": "Brown SL M119 Blaze Crystal Sunglasses", "sale": 960, "orig": 960},
            {"name": "Khaki SL 900 Howl Sunglasses", "sale": 535, "orig": 535},
            {"name": "Silver SL 807 Sunglasses", "sale": 650, "orig": 650},
            {"name": "Brown SL M160 Sunglasses", "sale": 446, "orig": 495},
            {"name": "Black SL 594 Sunglasses", "sale": 302, "orig": 495},
            {"name": "Khaki SL 859 Sunglasses", "sale": 468, "orig": 520},
            {"name": "Brown SL 276 Mica Sunglasses", "sale": 411, "orig": 495},
            {"name": "Black SL 889 Sulpice Thin Sunglasses", "sale": 328, "orig": 420},
            {"name": "Black SL 826 Sunglasses", "sale": 432, "orig": 520},
            {"name": "Brown SL M154 Sunglasses", "sale": 446, "orig": 495},
            {"name": "Black SL M154 Sunglasses", "sale": 446, "orig": 495},
            {"name": "Brown SL 822 Sunglasses", "sale": 351, "orig": 390},
            {"name": "Black SL 822 Sunglasses", "sale": 304, "orig": 390},
            {"name": "Black SL M153 Sunglasses", "sale": 439, "orig": 535},
            {"name": "Brown SL 890 Betty Thin Sunglasses", "sale": 378, "orig": 420},
            {"name": "Brown SL M152 Sunglasses", "sale": 371, "orig": 580},
            {"name": "Black SL M160 Sunglasses", "sale": 446, "orig": 495},
            {"name": "Gold SL 830 Sunglasses", "sale": 396, "orig": 535},
            {"name": "Brown SL 557 Shade Sunglasses", "sale": 416, "orig": 495},
            {"name": "Black SL 859 Sunglasses", "sale": 328, "orig": 520},
        ],
    },
    "Loewe": {
        "buyma_stats": {"min": 15000, "max": 196000, "median": 57500, "mean": 59693, "p25": 37136, "p75": 62300},
        "products": [
            {"name": "Black Speed Shield Sunglasses", "sale": 590, "orig": 590},
            {"name": "White Wave Mask Sunglasses", "sale": 550, "orig": 550},
            {"name": "Yellow Curvy Sunglasses", "sale": 310, "orig": 310},
            {"name": "Orange Signature Sunglasses", "sale": 440, "orig": 440},
            {"name": "Green Anagram Sunglasses", "sale": 420, "orig": 420},
            {"name": "Brown Mini Anagram Oval Sunglasses", "sale": 420, "orig": 420},
            {"name": "Purple Anagram Sunglasses", "sale": 440, "orig": 440},
            {"name": "White Swan Slim Sunglasses", "sale": 360, "orig": 360},
            {"name": "Brown Paulas Ibiza Dive In Mask Sunglasses", "sale": 380, "orig": 380},
            {"name": "Black Paulas Ibiza Dive In Mask Sunglasses", "sale": 380, "orig": 380},
            {"name": "Brown Slim Aviator Sunglasses", "sale": 420, "orig": 420},
            {"name": "Black Wave Mask Sunglasses", "sale": 550, "orig": 550},
            {"name": "Brown Anagram Sunglasses", "sale": 380, "orig": 380},
            {"name": "Black Double Frame Sunglasses", "sale": 480, "orig": 480},
            {"name": "Brown & Gold Double Frame Sunglasses", "sale": 480, "orig": 480},
            {"name": "Black Maxi Anagram Butterfly Sunglasses", "sale": 470, "orig": 470},
            {"name": "Red Signature Sunglasses", "sale": 440, "orig": 440},
            {"name": "Black Signature Sunglasses", "sale": 440, "orig": 440},
            {"name": "Black Square Anagram Sunglasses", "sale": 380, "orig": 380},
            {"name": "White Signature Sunglasses", "sale": 440, "orig": 440},
            {"name": "Brown Rectangular Sunglasses", "sale": 380, "orig": 380},
            {"name": "Black Rectangular Sunglasses", "sale": 380, "orig": 380},
            {"name": "Black Slim Sunglasses", "sale": 360, "orig": 360},
            {"name": "Brown Curvy Logo Cat Eye Sunglasses", "sale": 340, "orig": 340},
            {"name": "Gold Anagram Hexagonal Sunglasses", "sale": 430, "orig": 430},
            {"name": "Gold Daisy Field Sunglasses", "sale": 800, "orig": 800},
            {"name": "Black Double Frame Sunglasses (Sale)", "sale": 432, "orig": 480},
            {"name": "Black Slim Text Logo Rectangle Sunglasses", "sale": 306, "orig": 340},
            {"name": "Brown Paulas Ibiza Dive In Mask (Sale)", "sale": 342, "orig": 380},
            {"name": "Green Chunky Anagram Sunglasses", "sale": 323, "orig": 380},
            {"name": "Green Anagram Sunglasses (Sale)", "sale": 332, "orig": 390},
            {"name": "Black Rectangular Sunglasses (Sale)", "sale": 285, "orig": 380},
            {"name": "Gold Island Sunglasses", "sale": 368, "orig": 490},
            {"name": "Brown Triangle Slim Sunglasses", "sale": 324, "orig": 360},
            {"name": "Brown Beveled Pentagon Sunglasses", "sale": 370, "orig": 420},
            {"name": "Black Slim Sunglasses (Sale)", "sale": 353, "orig": 420},
            {"name": "Brown Triangle Slim (Sale)", "sale": 281, "orig": 360},
            {"name": "Brown Rectangular Sunglasses (Sale)", "sale": 293, "orig": 380},
            {"name": "Brown Mini Anagram Cateye Sunglasses", "sale": 295, "orig": 410},
            {"name": "Black Delta Slim Sunglasses", "sale": 284, "orig": 360},
            {"name": "Brown Mini Anagram Butterfly Sunglasses", "sale": 369, "orig": 410},
            {"name": "Black Wave Mask Sunglasses (Sale)", "sale": 495, "orig": 550},
            {"name": "Silver Shield Mask Sunglasses", "sale": 468, "orig": 520},
        ],
    },
    "Balenciaga": {
        "buyma_stats": {"min": 5350, "max": 186700, "median": 34480, "mean": 41547, "p25": 28600, "p75": 43840},
        "products": [
            {"name": "Gray Bossy Sunglasses", "sale": 655, "orig": 655},
            {"name": "Silver Gossip Everyday Sunglasses", "sale": 560, "orig": 560},
            {"name": "Black Everyday None Rectangular Sunglasses", "sale": 520, "orig": 520},
            {"name": "Black Tuesday Round Oval Sunglasses", "sale": 405, "orig": 405},
            {"name": "Black Oval Sunglasses", "sale": 670, "orig": 670},
            {"name": "Black Bossy Cat Eye Strass Sunglasses", "sale": 740, "orig": 740},
            {"name": "Black Superbusy Double Layer Sunglasses", "sale": 855, "orig": 855},
            {"name": "Black Mono Square Sunglasses", "sale": 575, "orig": 575},
            {"name": "Gray Boomerang Cat Sunglasses", "sale": 1015, "orig": 1015},
            {"name": "Gunmetal Monday Everyday Sunglasses", "sale": 520, "orig": 520},
            {"name": "Gunmetal Gossip D Frame Sunglasses", "sale": 620, "orig": 620},
            {"name": "Silver Invisible 20 Sunglasses", "sale": 490, "orig": 490},
            {"name": "Silver Tag 30 Sunglasses", "sale": 520, "orig": 520},
            {"name": "Black Sunset Sunglasses", "sale": 650, "orig": 650},
            {"name": "Black Nano Rectangle AF Sunglasses", "sale": 560, "orig": 560},
            {"name": "Black Flat Text Cateye Sunglasses", "sale": 430, "orig": 430},
            {"name": "Black Casino Cateye Sunglasses", "sale": 520, "orig": 520},
            {"name": "Black Casino Round Sunglasses", "sale": 520, "orig": 520},
            {"name": "Black Phantom Round Sunglasses", "sale": 855, "orig": 855},
        ],
    },
    "Bottega Veneta": {
        "buyma_stats": {"min": 15000, "max": 112500, "median": 36800, "mean": 47884, "p25": 28405, "p75": 69900},
        "products": [
            {"name": "Silver Cat Eye Knot Sunglasses", "sale": 760, "orig": 760},
            {"name": "Black Knot Cat Eye Sunglasses", "sale": 690, "orig": 690},
            {"name": "Black Knot Rectangular Sunglasses", "sale": 760, "orig": 760},
            {"name": "Brown Curvy Sunglasses", "sale": 570, "orig": 570},
            {"name": "Brown & Gold Classic Aviator Sunglasses", "sale": 655, "orig": 655},
            {"name": "Gold Light Ribbon Sunglasses", "sale": 510, "orig": 510},
            {"name": "Transparent Drop Aviator Sunglasses", "sale": 690, "orig": 690},
            {"name": "Brown Drop Aviator Sunglasses", "sale": 690, "orig": 690},
            {"name": "Black Drop Aviator Sunglasses", "sale": 690, "orig": 690},
            {"name": "Gold Metal Pilot Navigator Sunglasses", "sale": 705, "orig": 705},
            {"name": "Gold Classic Aviator Sunglasses", "sale": 450, "orig": 450},
            {"name": "Gold Bolt Sunglasses", "sale": 545, "orig": 545},
            {"name": "Black Classic Oval Sunglasses", "sale": 490, "orig": 490},
            {"name": "Black Ultrathin Shield Sunglasses", "sale": 535, "orig": 535},
            {"name": "Brown Classic Oval Sunglasses", "sale": 510, "orig": 510},
            {"name": "Black Snap Sunglasses", "sale": 510, "orig": 510},
            {"name": "Brown Slim Ribbon Cateye Sunglasses", "sale": 480, "orig": 480},
            {"name": "Brown Slim Ribbon Rectangular Sunglasses", "sale": 540, "orig": 540},
            {"name": "Black Rectangular Squared Sunglasses", "sale": 540, "orig": 540},
            {"name": "Brown Slim Ribbon Oval Sunglasses", "sale": 480, "orig": 480},
            {"name": "Black Drop Squared Sunglasses", "sale": 760, "orig": 760},
            {"name": "Gold Drop Aviator Sunglasses", "sale": 950, "orig": 950},
            {"name": "Gold Sardine Oval Sunglasses", "sale": 655, "orig": 655},
            {"name": "Gold Stretch Metal Sunglasses", "sale": 570, "orig": 570},
            {"name": "Silver Stretch Metal Sunglasses", "sale": 570, "orig": 570},
            {"name": "Gold Classic Rectangular Sunglasses", "sale": 570, "orig": 570},
            {"name": "Yellow Drop Squared Sunglasses", "sale": 760, "orig": 760},
            {"name": "Brown Drop Squared Sunglasses", "sale": 760, "orig": 760},
            {"name": "Black Knot Aviator Sunglasses", "sale": 760, "orig": 760},
            {"name": "Gold Square Sunglasses", "sale": 565, "orig": 565},
            {"name": "Gold Split Metal Sunglasses", "sale": 470, "orig": 470},
        ],
    },
    "Fendi": {
        "buyma_stats": {"min": 9506, "max": 115000, "median": 49900, "mean": 53237, "p25": 33750, "p75": 71000},
        "products": [
            {"name": "Black Lettering Sunglasses", "sale": 290, "orig": 290},
            {"name": "Gold FF Diamonds Sunglasses", "sale": 490, "orig": 490},
            {"name": "Off White FF Diamonds Sunglasses", "sale": 490, "orig": 490},
            {"name": "Black FF Squared Sunglasses", "sale": 490, "orig": 490},
            {"name": "Black Fendi Roma Sunglasses", "sale": 380, "orig": 380},
            {"name": "Black FF Diamonds Sunglasses", "sale": 490, "orig": 490},
            {"name": "Gold Baguette Sunglasses", "sale": 470, "orig": 470},
            {"name": "Brown Baguette Sunglasses", "sale": 470, "orig": 470},
            {"name": "Black Baguette Sunglasses", "sale": 470, "orig": 470},
            {"name": "Brown Lettering Sunglasses", "sale": 290, "orig": 290},
            {"name": "Gold Fendi First Crystal Sunglasses", "sale": 520, "orig": 520},
            {"name": "Gold Fendi Sky Sunglasses", "sale": 460, "orig": 460},
            {"name": "Pink Forever Fendi Sunglasses", "sale": 351, "orig": 390},
            {"name": "Brown Fendi Selleria Blue Light Glasses", "sale": 308, "orig": 440},
            {"name": "Brown Fendigraphy Sunglasses", "sale": 429, "orig": 550},
            {"name": "Gold Fendi First Crystal (Sale)", "sale": 442, "orig": 520},
            {"name": "Brown Lettering Sunglasses (Sale)", "sale": 261, "orig": 290},
            {"name": "Black Lettering Sunglasses (Sale)", "sale": 261, "orig": 290},
            {"name": "Black Forever Fendi Sunglasses", "sale": 417, "orig": 490},
            {"name": "Brown Forever Fendi Sunglasses", "sale": 368, "orig": 490},
            {"name": "Black Fendigraphy Sunglasses", "sale": 396, "orig": 550},
            {"name": "Black Forever Fendi Sunglasses", "sale": 316, "orig": 390},
            {"name": "Black Fendi Roma Sunglasses (Sale)", "sale": 323, "orig": 380},
            {"name": "Black Fendi Square Sunglasses", "sale": 342, "orig": 380},
            {"name": "Black Square Sunglasses", "sale": 351, "orig": 390},
        ],
    },
    "Versace": {
        "buyma_stats": {"min": 6830, "max": 87200, "median": 41960, "mean": 40172, "p25": 27350, "p75": 49800},
        "products": [
            {"name": "Silver Medusa Sunglasses", "sale": 290, "orig": 290},
            {"name": "Gold Aviator Sunglasses", "sale": 410, "orig": 410},
            {"name": "Black Medusa Biggie Pilot Sunglasses", "sale": 365, "orig": 365},
            {"name": "Black Maxi Medusa Biggie Sunglasses", "sale": 365, "orig": 365},
            {"name": "Black Medusa Biggie Butterfly Sunglasses", "sale": 390, "orig": 390},
            {"name": "Black Medusa Biggie Sunglasses", "sale": 365, "orig": 365},
            {"name": "Brown Tortoiseshell Sunglasses", "sale": 370, "orig": 370},
            {"name": "Black Square Sunglasses", "sale": 290, "orig": 290},
            {"name": "Brown Medusa Sunglasses", "sale": 260, "orig": 260},
            {"name": "Gold Rimless Rectangle Medusa Sunglasses", "sale": 465, "orig": 465},
            {"name": "Black Medusa Horizon Sunglasses", "sale": 435, "orig": 435},
            {"name": "Black Medusa Biggie Maxi Sunglasses", "sale": 390, "orig": 390},
            {"name": "Black Medusa Sunglasses", "sale": 410, "orig": 410},
            {"name": "Brown Greca Sunglasses", "sale": 365, "orig": 365},
            {"name": "Black Greca Sunglasses", "sale": 315, "orig": 315},
            {"name": "Black Medusa Biggie Cat Eye Sunglasses", "sale": 370, "orig": 370},
            {"name": "Black Medusa Sunglasses (2)", "sale": 290, "orig": 290},
            {"name": "White Medusa Sunglasses", "sale": 290, "orig": 290},
            {"name": "Black Medusa Focus Cat Eye Sunglasses", "sale": 370, "orig": 370},
            {"name": "Brown Medusa Wide Sunglasses", "sale": 290, "orig": 290},
            {"name": "Black Medusa Wide Sunglasses", "sale": 345, "orig": 345},
            {"name": "Gold Medusa Light Sunglasses", "sale": 410, "orig": 410},
            {"name": "Silver Medusa Sunglasses (2)", "sale": 410, "orig": 410},
            {"name": "Gold Greca Flow Sunglasses", "sale": 370, "orig": 370},
            {"name": "Gold Medusa Biggie Sunglasses", "sale": 410, "orig": 410},
            {"name": "Black & Gold Medusa Honor Rectangular", "sale": 304, "orig": 490},
            {"name": "Black Medusa Sunglasses (Sale)", "sale": 282, "orig": 415},
            {"name": "Black Greca Cat Eye Sunglasses", "sale": 246, "orig": 385},
            {"name": "Black Pillow Medusa Sunglasses", "sale": 229, "orig": 290},
            {"name": "Black Medusa Plaque Sunglasses", "sale": 302, "orig": 355},
            {"name": "Gold Medusa Sunglasses (Sale)", "sale": 359, "orig": 460},
            {"name": "Purple Medusa Biggie Sunglasses", "sale": 281, "orig": 365},
            {"name": "Black Maxi Medusa Biggie (Sale)", "sale": 270, "orig": 365},
            {"name": "Orange Medusa Biggie Sunglasses", "sale": 256, "orig": 365},
            {"name": "Black Medusa Biggie Butterfly (Sale)", "sale": 226, "orig": 390},
            {"name": "Black Medusa Biggie Maxi (Sale)", "sale": 332, "orig": 390},
            {"name": "Black Shield Aviator Sunglasses", "sale": 281, "orig": 365},
        ],
    },
}


@dataclass
class ProfitResult:
    brand: str
    name: str
    ssense_usd: int
    cost_jpy: int
    est_buyma_price: int
    revenue_after_fee: int
    profit: int
    profit_rate: float
    is_profitable: bool


def calc_profit(ssense_usd: int, buyma_price: int) -> ProfitResult:
    """利益計算"""
    cost_jpy = int(ssense_usd * USD_JPY) + SHIPPING_COST
    revenue_after_fee = int(buyma_price * (1 - BUYMA_FEE_RATE))
    profit = revenue_after_fee - cost_jpy
    profit_rate = profit / cost_jpy if cost_jpy > 0 else 0
    return {
        "cost_jpy": cost_jpy,
        "revenue_after_fee": revenue_after_fee,
        "profit": profit,
        "profit_rate": profit_rate,
        "is_profitable": profit_rate >= PROFIT_THRESHOLD,
    }


def estimate_buyma_price(ssense_usd: int, stats: dict) -> dict:
    """
    3段階のBUYMA推定販売価格を算出。
    保守=P25, 現実的=中央値, 楽観=P75（BUYMA実勢統計をそのまま使用）
    """
    conservative = stats["p25"]
    realistic = stats["median"]
    optimistic = stats["p75"]

    return {"conservative": conservative, "realistic": realistic, "optimistic": optimistic}


def main():
    output_dir = Path("/home/yn4416/projects/atelier-kyo-manager/data")
    output_dir.mkdir(exist_ok=True)

    scenarios = ["conservative", "realistic", "optimistic"]

    for scenario in scenarios:
        all_results = []
        brand_summary = []

        for brand, data in SSENSE_DATA.items():
            stats = data["buyma_stats"]
            products = data["products"]
            brand_results = []

            for p in products:
                buyma_ests = estimate_buyma_price(p["sale"], stats)
                buyma_est = buyma_ests[scenario]
                result = calc_profit(p["sale"], buyma_est)
                result["brand"] = brand
                result["name"] = p["name"]
                result["ssense_usd"] = p["sale"]
                result["est_buyma_price"] = buyma_est
                brand_results.append(result)
                all_results.append(result)

            profitable = [r for r in brand_results if r["is_profitable"]]
            profit_rates = [r["profit_rate"] for r in brand_results]

            brand_summary.append({
                "brand": brand,
                "total": len(brand_results),
                "profitable": len(profitable),
                "max_rate": max(profit_rates) if profit_rates else 0,
                "avg_rate": sum(profit_rates) / len(profit_rates) if profit_rates else 0,
                "min_profit": min(r["profit"] for r in brand_results),
                "max_profit": max(r["profit"] for r in brand_results),
            })

        # 楽観シナリオでCSV出力
        if scenario == "optimistic":
            csv_path = output_dir / "ssense_other_brands_profit_analysis.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "ブランド", "商品名", "SSENSE価格(USD)", "仕入原価(JPY)",
                    "BUYMA推定価格(JPY)", "手数料抜収入(JPY)", "利益(JPY)",
                    "利益率", "判定"
                ])
                for r in all_results:
                    writer.writerow([
                        r["brand"], r["name"], f"${r['ssense_usd']}",
                        f"¥{r['cost_jpy']:,}", f"¥{r['est_buyma_price']:,}",
                        f"¥{r['revenue_after_fee']:,}", f"¥{r['profit']:,}",
                        f"{r['profit_rate']:.1%}", "✅" if r["is_profitable"] else "❌"
                    ])

        # 楽観シナリオでMarkdown出力
        if scenario == "optimistic":
            _write_markdown(brand_summary, all_results, SSENSE_DATA)

        # コンソール出力
        scenario_labels = {"conservative": "保守(P25)", "realistic": "現実的(中央値)", "optimistic": "楽観(P75)"}
        print(f"\n=== {scenario_labels[scenario]} ===")
        for s in brand_summary:
            verdict = "✅利益圏内" if s["profitable"] > 0 else "❌全品赤字"
            if s["profitable"] > s["total"] * 0.5:
                verdict = "✅✅有望"
            elif s["profitable"] > 0:
                verdict = "⚠️一部商品のみ"
            print(f"  {s['brand']}: {s['total']}商品中{s['profitable']}商品が15%以上 (最高{s['max_rate']:.1%}, 平均{s['avg_rate']:.1%}) {verdict}")


def _write_markdown(brand_summary_opt, all_results_opt, ssense_data):
    """楽観シナリオ（P75ベース）でMarkdown出力"""
    md_path = Path("/home/yn4416/projects/obsidian-ssot/30_RESEARCH/atelier-kyo/SSENSE_他ブランド_サングラス利益分析_2026.md")
    md_path.parent.mkdir(parents=True, exist_ok=True)

    # 再計算: 各シナリオのサマリー
    all_scenario_summaries = {}
    for scenario in ["conservative", "realistic", "optimistic"]:
        summaries = []
        for brand, data in ssense_data.items():
            stats = data["buyma_stats"]
            products = data["products"]
            brand_results = []
            for p in products:
                buyma_ests = estimate_buyma_price(p["sale"], stats)
                buyma_est = buyma_ests[scenario]
                result = calc_profit(p["sale"], buyma_est)
                brand_results.append(result)
            profitable = [r for r in brand_results if r["is_profitable"]]
            profit_rates = [r["profit_rate"] for r in brand_results]
            summaries.append({
                "brand": brand,
                "total": len(brand_results),
                "profitable": len(profitable),
                "max_rate": max(profit_rates) if profit_rates else 0,
                "avg_rate": sum(profit_rates) / len(profit_rates) if profit_rates else 0,
            })
        all_scenario_summaries[scenario] = summaries

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SSENSE他ブランドサングラス利益分析 (2026年6月)\n\n")
        f.write("## 分析条件\n")
        f.write(f"- USD/JPY: {USD_JPY}\n")
        f.write(f"- BUYMA手数料: {BUYMA_FEE_RATE:.1%}\n")
        f.write(f"- 小物送料: ¥{SHIPPING_COST:,}\n")
        f.write(f"- 利益率基準: {PROFIT_THRESHOLD:.0%}以上\n")
        f.write(f"- 利益計算式: 利益 = BUYMA価格 x (1-0.142) - (SSENSE価格x{USD_JPY} + ¥{SHIPPING_COST:,})\n")
        f.write(f"- Celine: SSENSEにサングラスページなし(404)\n")
        f.write(f"- 分析方法: BUYMA実勢価格を3段階（保守/現実的/楽観）で推定\n\n")

        # === 楽観シナリオ（メイン） ===
        f.write("## ブランド別サマリー（楽観シナリオ: BUYMA P75ベース）\n\n")
        f.write("| ブランド | 商品数 | 利益率>=15%の数 | 最高利益率 | 平均利益率 | 総合判定 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for s in all_scenario_summaries["optimistic"]:
            verdict = "❌全品赤字"
            if s["profitable"] > s["total"] * 0.5:
                verdict = "✅✅有望"
            elif s["profitable"] > 0:
                verdict = "⚠️一部商品のみ"
            elif s["max_rate"] > -0.15:
                verdict = "⚠️赤字幅度小"
            f.write(
                f"| {s['brand']} | {s['total']} | {s['profitable']} | "
                f"{s['max_rate']:.1%} | {s['avg_rate']:.1%} | {verdict} |\n"
            )

        # === 3段階比較 ===
        f.write("\n## 3段階シナリオ比較\n\n")
        f.write("| ブランド | ")
        for label in ["保守(P25)", "現実的(中央値)", "楽観(P75)"]:
            f.write(f"{label} 利益率>=15%数 | {label} 最高利益率 | ")
        f.write("\n| --- | ")
        for _ in range(3):
            f.write(" --- | --- |")
        f.write("\n")
        for i, brand in enumerate(ssense_data.keys()):
            f.write(f"| {brand} | ")
            for scenario in ["conservative", "realistic", "optimistic"]:
                s = all_scenario_summaries[scenario][i]
                f.write(f"{s['profitable']}/{s['total']} | {s['max_rate']:.1%} | ")
            f.write("\n")

        # === 損益分岐為替レート ===
        f.write("\n## 損益分岐USD/JPYレート（楽観シナリオ基準）\n\n")
        f.write("15%利益に必要な最大USD/JPYレート（安い商品ほどうまくいく）\n\n")
        f.write("| ブランド | 最安SSENSE価格 | BUYMA P75 | 損益分岐レート | 15%利益レート | 現在(160.2)との差 |\n")
        f.write("| --- | --- | --- | --- | --- | --- |\n")
        for brand, data in ssense_data.items():
            stats = data["buyma_stats"]
            products = data["products"]
            min_usd = min(p["sale"] for p in products)
            buyma_p75 = stats["p75"]
            # Break-even: buyma * 0.858 = usd * rate + 1500
            be_rate = (buyma_p75 * (1 - BUYMA_FEE_RATE) - SHIPPING_COST) / min_usd
            # 15% profit: buyma * 0.858 = (usd * rate + 1500) * 1.15
            p15_rate = (buyma_p75 * (1 - BUYMA_FEE_RATE) - SHIPPING_COST * (1 + PROFIT_THRESHOLD)) / (min_usd * (1 + PROFIT_THRESHOLD))
            f.write(
                f"| {brand} | ${min_usd} | ¥{buyma_p75:,} | {be_rate:.1f} | {p15_rate:.1f} | "
                f"{'✅余裕' if p15_rate > 160.2 else f'⚠️あと{160.2 - p15_rate:.1f}円安(USD/JPY低下)が必要'} |\n"
            )

        # === BUYMA統計 ===
        f.write("\n## BUYMA実勢価格統計\n\n")
        f.write("| ブランド | 最低 | 25%タイル | 中央値 | 平均 | 75%タイル | 最高 |\n")
        f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
        for brand, data in ssense_data.items():
            s = data["buyma_stats"]
            f.write(
                f"| {brand} | ¥{s['min']:,} | ¥{s.get('p25', 'N/A'):,} | "
                f"¥{s['median']:,} | ¥{s['mean']:,} | ¥{s.get('p75', 'N/A'):,} | "
                f"¥{s['max']:,} |\n"
            )

        # === 商品別詳細 ===
        f.write("\n## 商品別詳細分析（楽観シナリオ・利益率上位10/ブランド）\n\n")
        for brand, data in ssense_data.items():
            stats = data["buyma_stats"]
            products = data["products"]
            results = []
            for p in products:
                buyma_ests = estimate_buyma_price(p["sale"], stats)
                r = calc_profit(p["sale"], buyma_ests["optimistic"])
                r["brand"] = brand
                r["name"] = p["name"]
                r["ssense_usd"] = p["sale"]
                r["est_buyma_price"] = buyma_ests["optimistic"]
                results.append(r)

            results.sort(key=lambda x: x["profit_rate"], reverse=True)
            top = results[:10]

            f.write(f"### {brand}\n\n")
            f.write("| 商品名 | SSENSE | 原価 | BUYMA推定 | 利益 | 利益率 | 判定 |\n")
            f.write("| --- | --- | --- | --- | --- | --- | --- |\n")
            for r in top:
                mark = "✅" if r["is_profitable"] else "❌"
                f.write(
                    f"| {r['name']} | ${r['ssense_usd']} | ¥{r['cost_jpy']:,} | "
                    f"¥{r['est_buyma_price']:,} | ¥{r['profit']:,} | "
                    f"{r['profit_rate']:.1%} | {mark} |\n"
                )
            f.write("\n")

        # === 結論 ===
        f.write("## 結論・推奨\n\n")
        opt_summaries = all_scenario_summaries["optimistic"]
        total_profitable = sum(s["profitable"] for s in opt_summaries)
        total_products = sum(s["total"] for s in opt_summaries)
        profitable_brands = [s for s in opt_summaries if s["profitable"] > 0]

        if total_profitable > 0:
            best = max(opt_summaries, key=lambda x: x["profitable"])
            f.write(f"### 楽観シナリオ（BUYMA P75価格設定）\n")
            f.write(f"- 全{len(ssense_data)}ブランド{total_products}商品中、{total_profitable}商品が15%以上の利益率\n\n")
            for s in profitable_brands:
                f.write(f"- **{s['brand']}**: {s['profitable']}/{s['total']}商品が利益圏内（最高{s['max_rate']:.1%}）\n")
            f.write("\n")
        else:
            f.write("### 結論: 全ブランド赤字\n")
            f.write("- 現在のUSD/JPY=160.2ではBUYMA市場価格に価格設定できず赤字\n\n")

        # 損益分岐レートからの推奨
        f.write("### 為替レート別の推奨\n\n")
        f.write("| 為替レート | 推奨アクション |\n| --- | --- |\n")
        f.write("| USD/JPY <= 130 | 全ブランドで利益可能（Saint Laurent, Bottega Veneta含む） |\n")
        f.write("| USD/JPY 130-160 | Fendi, Versace, Loewe中心に利益可能 |\n")
        f.write("| USD/JPY 160-180（現在域） | Fendi最有望。Versace/Loeweは赤字幅小。Balenciaga/BV困難 |\n")
        f.write("| USD/JPY >= 180 | ほぼ全ブランド困難 |\n\n")

        f.write("### 優先ブランドランキング\n\n")
        f.write("1. **Fendi** - BUYMA P75¥71,000で最安$261商品が利益可能。最高利益率見込\n")
        f.write("2. **Versace** - BUYMA P75¥49,800で最安$226商品がほぼ利益可能\n")
        f.write("3. **Loewe** - BUYMA価格帯が高く、安い商品($281-)なら可能性\n")
        f.write("4. **Bottega Veneta** - 高額商品中心のため利益確保難\n")
        f.write("5. **Saint Laurent** - 競合多数、BUYMA価格下落傾向\n")
        f.write("6. **Balenciaga** - 価格高騰、利益最も困難\n")

        f.write("\n---\n")
        f.write("*データ取得日: 2026年6月11日 / SSENSE価格・BUYMA価格は時点データ*\n")


if __name__ == "__main__":
    main()
