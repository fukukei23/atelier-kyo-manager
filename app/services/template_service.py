from __future__ import annotations

import csv
import io
import json

from app.models.listing_template import ListingTemplate
from app.utils.validation import sanitize_csv_cell


def generate_listing_text(product, template=None):
    """Product から出品文を生成。template未指定ならデフォルトテンプレートを使用。"""

    if template is None:
        template = ListingTemplate.query.filter_by(is_default=True).first()
    if template is None:
        template = _fallback_template()
    return template.render(product)


def generate_buyma_csv(products) -> tuple[str, int]:
    """BUYMA CSV形式で出品データを生成して (CSV文字列, スキップ件数) を返す。

    - 画像パス: processed_images(JSON)を優先、空ならimage_urlにフォールバック
    - 説明文なし商品はスキップ
    - UTF-8 BOM付き（Excel互換）
    """
    headers = [
        "商品名",
        "ブランド",
        "カラー",
        "サイズ",
        "素材",
        "定価",
        "販売価格",
        "仕入地域",
        "説明文",
        "カテゴリ",
        "画像パス",
        "pipeline_status",
    ]
    rows: list[list] = []
    skipped = 0

    for p in products:
        if not p.description:
            skipped += 1
            continue

        # 画像パス: processed_images(JSON配列) → カンマ区切り
        image_path = p.image_url or ""
        if p.processed_images:
            try:
                images = json.loads(p.processed_images)
                if isinstance(images, list) and images:
                    image_path = ",".join(str(img) for img in images)
            except (json.JSONDecodeError, TypeError):
                pass

        rows.append(
            [
                sanitize_csv_cell(p.name or ""),
                sanitize_csv_cell(p.brand or ""),
                sanitize_csv_cell(p.color or ""),
                sanitize_csv_cell(p.size or ""),
                sanitize_csv_cell(p.material or ""),
                int(p.retail_price or 0),
                int(p.selling_price or 0),
                sanitize_csv_cell(p.source_region or ""),
                sanitize_csv_cell(p.description or ""),
                sanitize_csv_cell(p.item_category or ""),
                sanitize_csv_cell(image_path),
                sanitize_csv_cell(p.pipeline_status or ""),
            ]
        )

    buf = io.StringIO()
    buf.write("\ufeff")  # UTF-8 BOM
    writer = csv.writer(buf, quoting=csv.QUOTE_ALL)
    writer.writerow(headers)
    writer.writerows(rows)
    return buf.getvalue(), skipped


def _fallback_template():
    """テンプレート未登録時のフォールバック"""
    t = ListingTemplate()
    t.template_text = (
        "【{{brand}}】{{productName}}\n"
        "カラー: {{color}}\n"
        "素材: {{material}}\n"
        "サイズ: {{size}}\n"
        "定価: {{retailPrice}}\n"
        "仕入国: {{sourceCountry}}\n"
        "状態: 新品未使用・タグ付き\n"
    )
    return t
