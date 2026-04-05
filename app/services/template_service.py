from __future__ import annotations

import csv
import io


def generate_listing_text(product, template=None):
    """Product から出品文を生成。template未指定ならデフォルトテンプレートを使用。"""
    from app.models.listing_template import ListingTemplate

    if template is None:
        template = ListingTemplate.query.filter_by(is_default=True).first()
    if template is None:
        template = _fallback_template()
    return template.render(product)


def generate_buyma_csv(products) -> str:
    """BUYMA CSV形式で出品データを生成して文字列で返す。"""
    headers = [
        "商品名", "ブランド", "カラー", "サイズ", "素材",
        "定価", "販売価格", "仕入地域", "説明文", "カテゴリ",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=headers)
    writer.writeheader()
    for p in products:
        writer.writerow({
            "商品名": p.name or "",
            "ブランド": p.brand or "",
            "カラー": p.color or "",
            "サイズ": p.size or "",
            "素材": p.material or "",
            "定価": int(p.retail_price or 0),
            "販売価格": int(p.selling_price or 0),
            "仕入地域": p.source_region or "",
            "説明文": p.description or "",
            "カテゴリ": "",
        })
    return buf.getvalue()


def _fallback_template():
    """テンプレート未登録時のフォールバック"""
    from app.models.listing_template import ListingTemplate
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
