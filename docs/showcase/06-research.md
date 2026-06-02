---
title: リサーチ
parent: 機能ショーケース
nav_order: 6
---

# リサーチ

![リサーチ]({{ site.baseurl }}/screenshots/06-research.png)

## 概要

海外仕入先サイトからPlaywrightで価格・在庫を自動取得。国別にリサーチ結果を商品管理ページに反映します。

## 対応国・サイト

| 国 | サイト | 通貨 |
|---|---|---|
| 🇺🇸 アメリカ | Farfetch, SSENSE, END. | USD |
| 🇬🇧 イギリス | Farfetch UK, END. UK | GBP |
| 🇮🇹 イタリア | Farfetch IT | EUR |
| 🇫🇷 フランス | Farfetch FR, Back Market | EUR |
| 🇩🇪 ドイツ | Farfetch DE, Zalando | EUR |

## 操作フロー

<p align="center">
  <img src="{{ site.baseurl }}/screenshots/scraping-demo.gif" width="600" alt="リサーチ操作デモ">
</p>

1. **国を選択**: アメリカ・イギリス・イタリア等他から仕入先を選択
2. **URLを入力**: 商品ページのURLを入力
3. **スクレイピング実行**: 「取得」ボタンでPlaywrightが自動実行
4. **結果を確認**: 商品名・価格・在庫状況を一覧表示
5. **商品管理に反映**: 結果をそのまま商品データベースに登録

## 価格スクレイピングの詳細

![価格スクレイピング]({{ site.baseurl }}/screenshots/price-scraping.png)

- **Playwright ヘッドレス**: サイトのご都合主義的反スクレイピング対策
- **BeautifulSoup 解析**: HTMLをパースして商品名・価格・在庫を抽出
- **為替自動適用**: EUR/USD → JPYに自動換算
- **24時間キャッシュ**: 同一URLの結果は24時間キャッシュで高速化
- **在庫判定**: 「売切れ」「在庫切れ」「Out of Stock」等のキーワードで自動判定

## 技術的詳細

- **ブラウザ自動化**: PlaywrightのヘッドレスモードでJavaScript描画も取得
- **DOM解析**: BeautifulSoup4 + lxml で高速HTML解析
- **レート制限**: 同一サイトへの連続アクセスを2秒間隔に制限
- **エラーハンドリング**: タイムアウト・接続エラー時のリトライ機構（最大3回）
