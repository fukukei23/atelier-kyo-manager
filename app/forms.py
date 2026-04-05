# =============================================================================
# ファイル   : app/forms.py
# 目的       : /manage と /auto-research で使う WTForms を定義
# 使い方     : 既存の forms.py をこの内容で全置換 → 保存
# 依存       : Flask-WTF (CSRF有効), WTForms
# =============================================================================
from __future__ import annotations
from flask_wtf import FlaskForm
from wtforms import (
    StringField, FloatField, BooleanField, SubmitField,
    SelectField, TextAreaField,
)
from wtforms.validators import DataRequired, Optional, URL, NumberRange

class ProductForm(FlaskForm):
    # 基本
    name = StringField("商品名", validators=[DataRequired(message="商品名は必須です")])
    brand = StringField("ブランド", validators=[Optional()])

    # 価格・費用（/manage の NOT NULL に合わせて必須2つ）
    purchase_price = FloatField(
        "仕入価格",
        validators=[DataRequired(message="仕入価格は必須です"), NumberRange(min=0)],
    )
    selling_price = FloatField(
        "販売価格",
        validators=[DataRequired(message="販売価格は必須です"), NumberRange(min=0)],
    )
    transaction_fee = FloatField("取引手数料", validators=[Optional(), NumberRange(min=0)])
    shipping_cost = FloatField("送料・梱包費", validators=[Optional(), NumberRange(min=0)])
    customs_duty = FloatField("関税・輸入消費税", validators=[Optional(), NumberRange(min=0)])
    procurement_fee = FloatField("買付代行料", validators=[Optional(), NumberRange(min=0)])

    # URL・在庫
    supplier_url = StringField("仕入先URL", validators=[Optional(), URL(require_tld=False, message="URLの形式で入力してください")])
    image_url = StringField("画像URL", validators=[Optional(), URL(require_tld=False, message="URLの形式で入力してください")])
    stock_status = BooleanField("在庫あり")

    # --- BUYMA拡張フィールド (F02) ---
    source_type = SelectField(
        "仕入種別",
        choices=[("", "---"), ("domestic", "国内"), ("overseas", "海外")],
        validators=[Optional()],
    )
    source_region = StringField("仕入地域", validators=[Optional()])
    color = StringField("カラー", validators=[Optional()])
    size = StringField("サイズ", validators=[Optional()])
    material = StringField("素材", validators=[Optional()])
    description = TextAreaField("説明文", validators=[Optional()])
    retail_price = FloatField("定価", validators=[Optional(), NumberRange(min=0)])
    target_profit_rate = FloatField(
        "目標利益率(%)",
        validators=[Optional(), NumberRange(min=0, max=100)],
        default=10.0,
    )
    listing_status = SelectField(
        "出品ステータス",
        choices=[
            ("draft", "下書き"), ("listed", "出品中"),
            ("sold", "売約済"), ("archived", "アーカイブ"),
        ],
        validators=[Optional()],
        default="draft",
    )

    # 送信
    submit = SubmitField("保存")

class AutoResearchForm(FlaskForm):
    """自動リサーチ画面のCSRF用（必要ならパラメータを追加可）"""
    submit = SubmitField("実行")
