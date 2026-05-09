# =============================================================================
# ファイル   : app/forms.py
# 目的       : /manage と /auto-research で使う WTForms を定義
# 使い方     : 既存の forms.py をこの内容で全置換 → 保存
# 依存       : Flask-WTF (CSRF有効), WTForms
# =============================================================================
from __future__ import annotations

from flask_wtf import FlaskForm
from wtforms import (
    BooleanField,
    FloatField,
    IntegerField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import URL, DataRequired, Email, Length, NumberRange, Optional


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

    # --- FR-005 実ベース利益計算フィールド ---
    warehouse_shipping_cost = FloatField("転送倉庫送料", validators=[Optional(), NumberRange(min=0)])
    original_currency = SelectField(
        "仕入れ通貨",
        choices=[("JPY", "JPY"), ("USD", "USD"), ("EUR", "EUR"), ("GBP", "GBP"), ("CNY", "CNY"), ("KRW", "KRW")],
        validators=[Optional()],
        default="JPY",
    )
    exchange_rate = FloatField("為替レート", validators=[Optional(), NumberRange(min=0)])
    item_category = StringField("品目カテゴリ", validators=[Optional()])

    # URL・在庫
    supplier_url = StringField(
        "仕入先URL", validators=[Optional(), URL(require_tld=False, message="URLの形式で入力してください")]
    )
    image_url = StringField(
        "画像URL", validators=[Optional(), URL(require_tld=False, message="URLの形式で入力してください")]
    )
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
            ("draft", "下書き"),
            ("listed", "出品中"),
            ("sold", "売約済"),
            ("archived", "アーカイブ"),
        ],
        validators=[Optional()],
        default="draft",
    )

    # 送信
    submit = SubmitField("保存")


class AutoResearchForm(FlaskForm):
    """自動リサーチ画面のCSRF用（必要ならパラメータを追加可）"""

    submit = SubmitField("実行")


# ---- F05: 注文管理 ----
class OrderForm(FlaskForm):
    order_number = StringField("注文番号", validators=[DataRequired(), Length(max=64)])
    product_name = StringField("商品名", validators=[DataRequired(), Length(max=255)])
    customer_name = StringField("顧客名", validators=[Optional()])
    order_date = StringField("注文日", validators=[Optional()])
    selling_price = FloatField("販売価格", validators=[DataRequired(), NumberRange(min=0)])
    purchase_cost = FloatField("仕入原価", validators=[DataRequired(), NumberRange(min=0)])
    customs_duty = FloatField("関税", validators=[Optional(), NumberRange(min=0)])
    payment_method = SelectField(
        "決済方法",
        choices=[
            ("credit_card", "クレジット"),
            ("rakuten_pay", "楽天ペイ"),
            ("d_pay", "d払い"),
            ("au_pay", "auかんたん決済"),
            ("paidy", "Paidy"),
            ("bank_transfer", "銀行振込"),
            ("convenience", "コンビニ"),
            ("paypay", "PayPay"),
            ("amazon_pay", "Amazon Pay"),
        ],
        validators=[Optional()],
    )
    source_type = SelectField(
        "仕入区分",
        choices=[
            ("domestic", "国内"),
            ("overseas", "海外"),
        ],
        validators=[Optional()],
    )
    status = SelectField(
        "ステータス",
        choices=[
            ("pending", "保留"),
            ("shipped", "発送済"),
            ("completed", "完了"),
            ("cancelled", "キャンセル"),
        ],
        validators=[Optional()],
    )
    notes = TextAreaField("メモ", validators=[Optional()])
    submit = SubmitField("保存")


# ---- F06: パートナー管理 ----
class PartnerForm(FlaskForm):
    name = StringField("名前", validators=[DataRequired()])
    email = StringField("メール", validators=[Optional(), Email()])
    phone = StringField("電話番号", validators=[Optional()])
    active_regions = StringField("活動地域", validators=[Optional()])
    specialty_brands = StringField("得意ブランド", validators=[Optional()])
    priority_level = SelectField(
        "優先度",
        choices=[
            ("low", "低"),
            ("medium", "中"),
            ("high", "高"),
        ],
        validators=[Optional()],
    )
    status = SelectField(
        "ステータス",
        choices=[
            ("active", "有効"),
            ("inactive", "無効"),
            ("suspended", "停止"),
        ],
        validators=[Optional()],
    )
    notes = TextAreaField("メモ", validators=[Optional()])
    submit = SubmitField("保存")


# ---- F13: リピーター管理 ----
class CustomerForm(FlaskForm):
    customer_name = StringField("顧客名", validators=[DataRequired()])
    email = StringField("メール", validators=[Optional()])
    phone = StringField("電話番号", validators=[Optional()])
    total_orders = IntegerField("総注文数", validators=[Optional()], default=0)
    total_spent = FloatField("総購入額", validators=[Optional()], default=0)
    first_order_date = StringField("初回注文日", validators=[Optional()])
    last_order_date = StringField("最終注文日", validators=[Optional()])
    notes = TextAreaField("メモ", validators=[Optional()])
    submit = SubmitField("保存")


# ---- F03: 出品テンプレート ----
class ListingTemplateForm(FlaskForm):
    name = StringField("テンプレート名", validators=[DataRequired()])
    template_text = TextAreaField("本文", validators=[DataRequired()])
    category = SelectField(
        "カテゴリ",
        choices=[
            ("general", "一般"),
            ("fashion", "ファッション"),
            ("luxury", "ラグジュアリー"),
        ],
        validators=[Optional()],
    )
    is_default = BooleanField("デフォルトにする")
    submit = SubmitField("保存")


# ---- F10: 在庫チェック ----
class StockCheckForm(FlaskForm):
    product_id = IntegerField("商品ID", validators=[DataRequired()])
    source_url = StringField("仕入先URL", validators=[Optional(), URL(require_tld=False)])
    current_price = FloatField("現在価格", validators=[Optional(), NumberRange(min=0)])
    in_stock = BooleanField("在庫あり")
    notes = TextAreaField("メモ", validators=[Optional()])
    submit = SubmitField("保存")


# ---- F07: 品出し進捗 ----
class ListingProgressForm(FlaskForm):
    record_date = StringField("記録日", validators=[Optional()])
    listings_count = IntegerField("出品数", validators=[Optional()], default=0)
    target_daily = IntegerField("目標(日)", validators=[Optional()], default=20)
    target_monthly = IntegerField("目標(月)", validators=[Optional()], default=600)
    cumulative_monthly = IntegerField("累計(月)", validators=[Optional()], default=0)
    notes = TextAreaField("メモ", validators=[Optional()])
    submit = SubmitField("保存")


# ---- F11: 人気度トラッキング ----
class PopularityForm(FlaskForm):
    product_id = IntegerField("商品ID", validators=[DataRequired()])
    views = IntegerField("閲覧数", validators=[Optional()], default=0)
    favorites = IntegerField("お気に入り数", validators=[Optional()], default=0)
    inquiries = IntegerField("問い合わせ数", validators=[Optional()], default=0)
    sold_count = IntegerField("販売数", validators=[Optional()], default=0)
    submit = SubmitField("保存")


# ---- F12: 地域レコメンド ----
class RegionForm(FlaskForm):
    region = StringField("地域コード", validators=[DataRequired()])
    region_name = StringField("地域名", validators=[Optional()])
    avg_profit_rate = FloatField("平均利益率", validators=[Optional(), NumberRange(min=0)])
    avg_shipping_days = IntegerField("平均配送日数", validators=[Optional()])
    avg_customs_rate = FloatField("平均関税率", validators=[Optional(), NumberRange(min=0)])
    risk_score = FloatField("リスクスコア", validators=[Optional(), NumberRange(min=0, max=100)], default=50)
    reliability_score = FloatField("信頼性スコア", validators=[Optional(), NumberRange(min=0, max=100)], default=50)
    submit = SubmitField("保存")
