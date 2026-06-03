---
title: 概要
nav_order: 1
---

# Atelier Kyo Manager

> 📂 **[GitHub リポジトリ →](https://github.com/fukukei23/atelier-kyo-manager)**{: .btn .btn-blue } — ソースコード・テスト・技術詳細はこちら

BUYMA × Buyandshipを利用した転売管理システム（個人用）。出品パイプライン、注文ステートマシン、AIチャットボット、LLMルーティングを統合したFlaskアプリケーション。

## 操作デモ

<p align="center">
  <img src="{{ site.baseurl }}/screenshots/demo-flow.gif" width="600" alt="操作デモ">
</p>

> ログイン → ダッシュボード（売上・注文状況を一覧表示） → 商品管理（出品パイプライン実行） → 注文管理（18日ルール） → キャッシュフロー（月次利益予測）の流れ。

## 特徴

- **出品パイプライン自動化**: 画像収集 → AI背景除去 → AI説明文生成 → 出品テキスト生成
- **注文ステートマシン**: pending → sourcing → shipped → completed の状態遷移
- **AIチャットボット**: FAQテンプレート → AI回答 → エスカレーションの3段階分類
- **LLMルーティング**: OpenAI / Gemini / Local LLM を統一管理
- **価格スクレイピング**: Playwrightで仕入先価格を自動取得
- **18日ルール**: 決済方法別の延長期限自動計算

## 技術スタック

| カテゴリ | 技術 |
|---|---|
| 言語 | Python 3.x |
| フレームワーク | Flask + SQLAlchemy |
| スクレイピング | Playwright / Selenium |
| 画像処理 | Pillow / OpenCV / rembg |
| LLM | OpenAI / Gemini / Local LLM |
| テスト | pytest (2,070 tests) |

---

> 👉 各機能の詳細はサイドバーの **機能ショーケース** をご覧ください。
