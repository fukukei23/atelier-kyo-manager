---
title: ダッシュボード
parent: 機能ショーケース
nav_order: 1
---

# ダッシュボード

![ダッシュボード]({{ site.baseurl }}/screenshots/01-dashboard.png)

## 概要

売上推移・未処理注文数・在庫アラートを一覧表示する統合管理画面。BUYMAとBuyandshipの両方のデータを集約し、一目でビジネス全体を把握できます。

## 機能のポイント

- **売上推移グラフ**: 月次・週次での売上トレンドを可視化
- **未処理注文アラート**: 対応が必要な注文数をリアルタイム表示
- **在庫アラート**: 在庫切れ予測・補充タイミングを通知
- **BUYMA / Buyandship統合**: 両プラットフォームのデータを一画面に集約

## 技術的詳細

- **データ集約**: SQLAlchemy ORMで複数テーブル（orders, products, partners）をJOIN集計
- **チャート描画**: Chart.js で売上グラフ・カテゴリ別円グラフをフロントエンド描画
- **キャッシュ**: 集計クエリ結果を15分キャッシュし、ダッシュボードの高速表示を実現

## データフロー

```
BUYMA API ──┐
            ├──→ SQLAlchemy ──→ 集計Service ──→ Jinja2 Template ──→ Chart.js
Buyandship ─┘
```
