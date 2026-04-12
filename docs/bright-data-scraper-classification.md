# Bright Data スクレーパーライブラリ分類

**日付**: 2026-04-12
**分析手法**: GLM-5.1
**目的**: atelier-kyo-manager（BUYMA転売ツール）で利用するドメインを206種から選定

## カテゴリ1: 仕入れ先（海外EC）
実際に商品を仕入れるサイト。Collector作成推奨度: ★3=必須, ★2=推奨, ★1=任意

| サイト | 特徴 | 推奨度 |
|--------|------|--------|
| ssense.com | 既存Collectorあり (`c_mnub8vs31ch29pinx1`) | ★★★ |
| farfetch.com | セレクトショップ集約、ラグジュアリー幅広い | ★★★ |
| nordstrom.com | 総合デパート、ブランド幅広い | ★★★ |
| saksfifthavenue.com | 高級デパート | ★★★ |
| neimanmarcus.com | ラグジュアリー特化 | ★★★ |
| matchesfashion.com | ラグジュアリーセレクト | ★★☆ |
| bloomingdales.com | 高級デパート | ★★☆ |
| macys.com | 総合デパート、品揃え最大級 | ★★☆ |
| zara.com | ファストファッション | ★☆☆ |
| asos.com | カジュアル中心 | ★☆☆ |

## カテゴリ2: 価格調査先（市場比較）
BUYMA出品価格の設定参考用

| サイト | 用途 |
|--------|------|
| amazon.com | 価格下限チェック |
| ebay.com | 中古価格・オークション相場 |
| yahoo.co.jp (shopping) | 日本国内価格比較 |
| rakuten.co.jp | 日本国内価格比較 |
| coupang.com | 韓国EC（K-beauty等） |

## カテゴリ3: ブランド公式サイト
正規品証明・在庫確認用。BUYMA出品で「正規品」アピールに有効

| サイト | ブランド |
|--------|----------|
| hermes.com | エルメス |
| chanel.com | シャネル |
| dior.com | ディオール |
| prada.com | プラダ |
| fendi.com | フェンディ |
| gucci.com | グッチ |
| louisvuitton.com | ルイ・ヴィトン |
| burberry.com | バーバリー |
| balenciaga.com | バレンシアガ |
| givenchy.com | ジバンシー |
| valentino.com | バレンティノ |
| bottegaveneta.com | ボッテガ・ヴェネタ |
| ysl.com | サンローラン |
| cartier.com | カルティエ |
| celine.com | セリーヌ |
| loewe.com | ロエベ |
| moncler.com | モンクレー |
| off-white.com | オフホワイト |
| acnestudios.com | アクネストゥディオズ |
| nike.com | ナイキ |
| adidas.com | アディダス |
| newbalance.com | ニューバランス |

## カテゴリ4: リセール市場
価格相場把握用。BUYMA価格の適正設定に活用

| サイト | 用途 |
|--------|------|
| ebay.com | グローバル中古相場 |
| vinted.com | 欧州リセール相場 |
| poshmark.com | 米国リセール相場 |
| mercari.com | 日本フリマ相場 |
| therealreal.com | 高級品リセール相場 |
| stockx.com | スニーカー・ストリート相場 |

## 実装優先順位

1. **P0**: ssense.com（既存）、farfetch.com、nordstrom.com — 仕入れ先スクレーパー
2. **P1**: ブランド公式（hermes, chanel, dior等）— 在庫・正規品確認
3. **P2**: mercari.com、yahoo.co.jp — 日本国内価格比較
4. **P3**: stockx.com、ebay.com — リセール相場確認

## 注意
- BUYMAはBright Dataライブラリにない → Scraper Studio AIでカスタムCollector作成
- 画像URLフィールドの追加が必要（SSENSE既存Collectorで未取得）
