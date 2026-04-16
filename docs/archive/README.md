# Atelier Kyo Manager

EC商品管理・AI自動化リサーチシステム

## プロジェクト概要

Atelier Kyo Managerは、ECサイトの商品情報取得、利益分析、AI自動リサーチのための統合システムです。

## 主要機能

- **LLM統合**: MiniMax / GLM 対応
- **スクレイピング**: SSENSE, GUCCI, PRADA, MONCLER 対応
- **Flaskダッシュボード**: 商品管理・利益分析
- **Playwrightブラウザ自動化**: 遅延読み込み対応

## ドキュメント

- [セットアップコマンド](docs/setup_commands.md) - 環境構築手順
- [開発計画](docs/DEVELOPMENT_PLAN.md) - P0-P2優先事項
- [完了レポート](docs/completion_reports/) - 作業履歴

## 開発

```bash
# テスト実行
python -m pytest tests/

# Flaskアプリ起動
flask run
```

## アーキテクチャ

```
app/
├── agents/          # AIエージェント
│   ├── browser/      # ブラウザ自動化
│   └── plugins/      # サイト別スクレイピング戦略
├── config/           # 設定
├── core/             # コア機能
├── models/           # データベースモデル
├── extractors/       # 抽出ロジック
└── web/             # Flaskダッシュボード
```
